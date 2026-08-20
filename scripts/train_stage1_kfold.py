#!/usr/bin/env python3
"""Stage 1: adapt ESM2-650M to binary ACP recognition with five folds."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acp_protfusion.config import ExperimentConfig
from acp_protfusion.data import BinaryPeptideDataset, make_folds
from acp_protfusion.metrics import best_f1_threshold, binary_metrics
from acp_protfusion.model import BinaryACPClassifier, load_tokenizer
from acp_protfusion.training import (
    build_optimizer,
    cosine_warmup_scheduler,
    freeze_lower_backbone_layers,
    load_fold_results,
    mean_and_std,
    save_json,
    set_seed,
)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probabilities.append(torch.softmax(output["logits"], dim=-1)[:, 1].cpu().numpy())
        labels.append(batch["labels"].numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def train_fold(args, config: ExperimentConfig) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_path = Path(args.data_dir) / f"{args.dataset}_train.csv"
    test_path = Path(args.data_dir) / f"{args.dataset}_test.csv"
    frame = pd.read_csv(train_path)
    folds = make_folds(frame, args.n_folds, args.fold_seed)
    training_indices, validation_indices = folds[args.fold]

    tokenizer = load_tokenizer(args.backbone)
    complete_dataset = BinaryPeptideDataset(train_path, tokenizer, config.max_length)
    test_dataset = BinaryPeptideDataset(test_path, tokenizer, config.max_length)
    training_loader = DataLoader(
        Subset(complete_dataset, training_indices), batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, drop_last=True,
    )
    validation_loader = DataLoader(
        Subset(complete_dataset, validation_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    model = BinaryACPClassifier(
        args.backbone, config.attention_heads, config.attention_dropout,
        config.head_hidden_dim, config.head_dropout,
    ).to(device)
    freeze_lower_backbone_layers(model, config.trainable_backbone_layers)
    optimizer = build_optimizer(
        model, args.lr_backbone, args.lr_head, args.weight_decay,
        config.layerwise_lr_decay,
        backbone_misc_lr=(
            args.lr_backbone
            * config.layerwise_lr_decay ** int(model.backbone.config.num_hidden_layers)
        ),
    )
    total_steps = args.epochs * len(training_loader)
    scheduler = cosine_warmup_scheduler(optimizer, int(config.warmup_ratio * total_steps), total_steps)
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    results_dir = Path(args.output_dir) / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"stage1_{args.dataset}_fold{args.fold}.pt"
    best_validation_f1 = -1.0
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in training_loader:
            output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(output["logits"], batch["labels"].to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())

        validation_probabilities, validation_labels = predict(model, validation_loader, device)
        threshold = best_f1_threshold(validation_labels, validation_probabilities)
        validation = binary_metrics(validation_labels, validation_probabilities, threshold)
        print(
            f"fold={args.fold} epoch={epoch}/{args.epochs} "
            f"loss={total_loss / max(1, len(training_loader)):.4f} "
            f"val_f1={validation['f1']:.4f} threshold={threshold:.3f}"
        )
        if validation["f1"] > best_validation_f1:
            best_validation_f1 = validation["f1"]
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "backbone": args.backbone,
                "dataset": args.dataset,
                "fold": args.fold,
                "epoch": epoch,
                "validation_metrics": validation,
                "threshold": threshold,
            }, checkpoint_path)
        else:
            patience += 1
            if patience >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_probabilities, test_labels = predict(model, test_loader, device)
    test_metrics = binary_metrics(test_labels, test_probabilities, checkpoint["threshold"])
    save_json(results_dir / f"stage1_{args.dataset}_fold{args.fold}.json", {
        "fold": args.fold,
        "best_epoch": checkpoint["epoch"],
        "validation_f1": checkpoint["validation_metrics"]["f1"],
        **test_metrics,
    })
    print(f"Independent test F1: {test_metrics['f1']:.4f}")


def aggregate(args) -> None:
    results = load_fold_results(
        Path(args.output_dir) / "results", f"stage1_{args.dataset}_fold{{fold}}.json", args.n_folds
    )
    keys = ["accuracy", "precision", "recall", "f1", "fpr", "auroc", "auprc"]
    summary = mean_and_std(results, keys)
    save_json(Path(args.output_dir) / "results" / f"stage1_{args.dataset}_summary.json", summary)

    # Stage 2 uses the Stage 1 fold chosen strictly by Stage 1 validation F1.
    best_fold = max(results, key=lambda row: row["validation_f1"])["fold"]
    source = Path(args.output_dir) / "checkpoints" / f"stage1_{args.dataset}_fold{best_fold}.pt"
    destination = Path(args.output_dir) / "checkpoints" / f"stage1_{args.dataset}_best.pt"
    shutil.copy2(source, destination)
    print(f"Saved validation-selected Stage 1 checkpoint: {destination}")
    for key, values in summary.items():
        print(f"{key}: {values['mean']:.4f} +/- {values['std']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("set1", "set2"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--data-dir", default="data/stage1")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--backbone", default=ExperimentConfig().backbone)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.aggregate_only:
        aggregate(arguments)
    elif arguments.fold is None:
        raise SystemExit("--fold is required unless --aggregate-only is used")
    else:
        train_fold(arguments, ExperimentConfig(backbone=arguments.backbone))
