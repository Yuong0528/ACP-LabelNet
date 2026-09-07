"""Shared training utilities with no project- or cluster-specific paths."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_lower_backbone_layers(model, trainable_layers: int) -> None:
    """Freeze embeddings and all but the upper N Transformer layers."""
    total_layers = int(model.backbone.config.num_hidden_layers)
    first_trainable = total_layers - trainable_layers
    for name, parameter in model.backbone.named_parameters():
        if "embeddings" in name or "embed_tokens" in name or name.endswith("embed.weight"):
            parameter.requires_grad = False
            continue
        layer_index = _layer_index(name)
        parameter.requires_grad = layer_index is None or layer_index >= first_trainable
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {trainable:,}/{total:,} ({100 * trainable / total:.2f}%)")


def build_optimizer(
    model,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    layerwise_decay: float,
    sequence_projection_lr: float | None = None,
    backbone_misc_lr: float | None = None,
) -> AdamW:
    """Apply layer-wise decay to trainable backbone blocks and head_lr elsewhere."""
    backbone_groups: dict[int, list[torch.nn.Parameter]] = {}
    backbone_misc_parameters = []
    sequence_projection_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            layer_index = _layer_index(name)
            if layer_index is not None:
                backbone_groups.setdefault(layer_index, []).append(parameter)
                continue
            if backbone_misc_lr is not None:
                backbone_misc_parameters.append(parameter)
                continue
        if sequence_projection_lr is not None and "label_graph.sequence_projections" in name:
            sequence_projection_parameters.append(parameter)
            continue
        head_parameters.append(parameter)

    groups = []
    if backbone_groups:
        uppermost = max(backbone_groups)
        for layer_index, parameters in sorted(backbone_groups.items()):
            groups.append({
                "params": parameters,
                "lr": backbone_lr * layerwise_decay ** (uppermost - layer_index),
            })
    if head_parameters:
        groups.append({"params": head_parameters, "lr": head_lr})
    if sequence_projection_parameters:
        groups.append({"params": sequence_projection_parameters, "lr": sequence_projection_lr})
    if backbone_misc_parameters:
        groups.append({"params": backbone_misc_parameters, "lr": backbone_misc_lr})
    return AdamW(groups, weight_decay=weight_decay)


def cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, multiplier)


def save_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def load_fold_results(results_dir: str | Path, pattern: str, n_folds: int) -> list[dict]:
    paths = [Path(results_dir) / pattern.format(fold=fold) for fold in range(n_folds)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing fold results: " + ", ".join(missing))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def mean_and_std(rows: list[dict], keys: list[str]) -> dict[str, dict[str, float]]:
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows])),
        }
        for key in keys
    }


def _layer_index(name: str) -> int | None:
    for marker in ("encoder.layer.", "transformer.blocks."):
        if marker in name:
            try:
                return int(name.split(marker, 1)[1].split(".", 1)[0])
            except (IndexError, ValueError):
                return None
    return None
