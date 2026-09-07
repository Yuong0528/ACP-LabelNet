"""Datasets, fold construction, handcrafted features, and augmentation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import CANCER_TYPES


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDE_INDEX = {
    first + second: index
    for index, (first, second) in enumerate(
        (a, b) for a in AMINO_ACIDS for b in AMINO_ACIDS
    )
}
RARE_RESIDUES = re.compile(r"[UZOBJ]")

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    "X": 0.0,
}
MOLECULAR_WEIGHT = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.03, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
    "X": 110.0,
}
BLOSUM62_CONSERVATIVE = {
    "A": "S", "R": "K", "N": "D", "D": "E", "C": "S",
    "Q": "E", "E": "D", "G": "A", "H": "N", "I": "V",
    "L": "I", "K": "R", "M": "L", "F": "Y", "P": "A",
    "S": "T", "T": "S", "W": "Y", "Y": "F", "V": "I",
}


def clean_sequence(sequence: str) -> str:
    return RARE_RESIDUES.sub("X", str(sequence).upper().strip())


def tokenizer_needs_spaces(tokenizer) -> bool:
    """Detect ProtBERT-style tokenizers without relying on a class name."""
    return len(tokenizer("KWKL", add_special_tokens=False)["input_ids"]) < 4


def tokenize_sequence(tokenizer, sequence: str, max_length: int) -> dict[str, torch.Tensor]:
    sequence = clean_sequence(sequence)
    if tokenizer_needs_spaces(tokenizer):
        sequence = " ".join(sequence)
    encoded = tokenizer(
        sequence,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {name: value.squeeze(0) for name, value in encoded.items()}


def physicochemical_features(sequence: str, max_length: int = 64) -> np.ndarray:
    """Six normalized descriptors used in the reported Stage 2 model."""
    sequence = clean_sequence(sequence)
    length = len(sequence)
    if length == 0:
        return np.zeros(6, dtype=np.float32)
    gravy = np.mean([KYTE_DOOLITTLE.get(aa, 0.0) for aa in sequence]) / 4.5
    positive = sum(1.0 if aa in "RK" else 0.1 if aa == "H" else 0.0 for aa in sequence)
    negative = sum(aa in "DE" for aa in sequence)
    net_charge = (positive - negative) / 10.0
    cationic_fraction = sum(aa in "RK" for aa in sequence) / length
    aromatic_fraction = sum(aa in "FWY" for aa in sequence) / length
    mass_kda = sum(MOLECULAR_WEIGHT.get(aa, 110.0) for aa in sequence) / 1000.0
    mass_normalized = float(np.clip(mass_kda / 10.0, 0.0, 1.0))
    length_normalized = float(np.clip(length / max_length, 0.0, 1.0))
    return np.asarray(
        [gravy, net_charge, cationic_fraction, aromatic_fraction,
         mass_normalized, length_normalized],
        dtype=np.float32,
    )


def handcrafted_features(sequence: str, max_length: int = 64) -> np.ndarray:
    """Return physicochemical (6) + AAC (20) + DPC (400) features."""
    sequence = clean_sequence(sequence)
    canonical = "".join(aa for aa in sequence if aa in AMINO_ACIDS)
    length = len(canonical)
    aac = np.zeros(20, dtype=np.float32)
    dpc = np.zeros(400, dtype=np.float32)
    if length:
        aac = np.asarray([canonical.count(aa) for aa in AMINO_ACIDS], dtype=np.float32) / length
    if length > 1:
        for left, right in zip(canonical[:-1], canonical[1:]):
            dpc[DIPEPTIDE_INDEX[left + right]] += 1
        dpc /= length - 1
    return np.concatenate([physicochemical_features(sequence, max_length), aac, dpc])


class BinaryPeptideDataset(Dataset):
    def __init__(self, csv_path: str | Path, tokenizer, max_length: int = 64):
        self.frame = pd.read_csv(csv_path)
        _require_columns(self.frame, ("sequence", "label"), csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        item = tokenize_sequence(self.tokenizer, row["sequence"], self.max_length)
        item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item


class MultiLabelPeptideDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int = 64):
        self.frame = frame.reset_index(drop=True)
        _require_columns(self.frame, ("sequence", *CANCER_TYPES), "data frame")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        item = tokenize_sequence(self.tokenizer, row["sequence"], self.max_length)
        item["labels"] = torch.tensor(
            [int(row[label]) for label in CANCER_TYPES], dtype=torch.float32
        )
        item["handcrafted"] = torch.from_numpy(
            handcrafted_features(row["sequence"], self.max_length)
        )
        return item


def make_folds(
    frame: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 0,
    stratify_column: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct deterministic folds; Stage 2 stratifies on cervical labels."""
    rng = np.random.RandomState(seed)
    if stratify_column is None:
        validation_parts = np.array_split(rng.permutation(len(frame)), n_folds)
    else:
        _require_columns(frame, (stratify_column,), "data frame")
        positive = np.flatnonzero(frame[stratify_column].to_numpy() == 1)
        negative = np.flatnonzero(frame[stratify_column].to_numpy() == 0)
        rng.shuffle(positive)
        rng.shuffle(negative)
        positive_parts = np.array_split(positive, n_folds)
        negative_parts = np.array_split(negative, n_folds)
        validation_parts = [
            np.concatenate((positive_parts[i], negative_parts[i])) for i in range(n_folds)
        ]

    all_indices = np.arange(len(frame))
    folds = []
    for validation in validation_parts:
        validation = np.sort(validation)
        training = np.setdiff1d(all_indices, validation)
        folds.append((training, validation))
    _validate_folds(folds, len(frame), frame, stratify_column)
    return folds


def conditional_cooccurrence(frame: pd.DataFrame) -> np.ndarray:
    """Compute P(label_j=1 | label_i=1) using training-fold annotations only."""
    labels = frame.loc[:, CANCER_TYPES].to_numpy(dtype=np.float32)
    result = np.zeros((len(CANCER_TYPES), len(CANCER_TYPES)), dtype=np.float32)
    for i in range(len(CANCER_TYPES)):
        positives = labels[:, i] == 1
        if positives.any():
            result[i] = labels[positives].mean(axis=0)
    return result


def augment_cervical_positives(
    frame: pd.DataFrame,
    seed: int,
    substitution_probability: float = 0.15,
    minimum_subsequence_fraction: float = 0.80,
) -> pd.DataFrame:
    """Create one augmented example per cervical-positive training peptide."""
    rng = np.random.RandomState(seed)
    augmented = []
    positives = frame[frame["cervical"] == 1]
    for _, row in positives.iterrows():
        sequence = clean_sequence(row["sequence"])
        sequence = _conservative_substitution(sequence, substitution_probability, rng)
        sequence = _subsequence_sample(sequence, minimum_subsequence_fraction, rng)
        copied = row.to_dict()
        copied["sequence"] = sequence
        augmented.append(copied)
    if not augmented:
        return frame.copy()
    return pd.concat([frame, pd.DataFrame(augmented)], ignore_index=True)


def _conservative_substitution(sequence: str, probability: float, rng) -> str:
    residues = list(sequence)
    eligible = [index for index, residue in enumerate(residues) if residue in BLOSUM62_CONSERVATIVE]
    if not eligible:
        return sequence
    count = min(max(1, int(len(residues) * probability)), len(eligible))
    for index in rng.choice(eligible, size=count, replace=False):
        residues[index] = BLOSUM62_CONSERVATIVE[residues[index]]
    return "".join(residues)


def _subsequence_sample(sequence: str, minimum_fraction: float, rng) -> str:
    minimum_length = max(6, int(len(sequence) * minimum_fraction))
    if minimum_length >= len(sequence):
        return sequence
    sampled_length = rng.randint(minimum_length, len(sequence) + 1)
    start = rng.randint(0, len(sequence) - sampled_length + 1)
    return sequence[start:start + sampled_length]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], source) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {source}: {', '.join(missing)}")


def _validate_folds(folds, sample_count, frame, stratify_column) -> None:
    validation = np.concatenate([indices for _, indices in folds])
    if len(validation) != sample_count or len(np.unique(validation)) != sample_count:
        raise RuntimeError("Validation folds must cover every training sample exactly once")
    for fold, (training, held_out) in enumerate(folds):
        if not len(training) or not len(held_out) or np.intersect1d(training, held_out).size:
            raise RuntimeError(f"Invalid train/validation partition in fold {fold}")
        if stratify_column and int(frame.iloc[held_out][stratify_column].sum()) == 0:
            raise RuntimeError(f"Fold {fold} has no positive {stratify_column} examples")
