#!/usr/bin/env python3
"""Stage 2: train ACP-LabelNet for five-label cancer-type prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acp_labelnet.config import CANCER_TYPES, ExperimentConfig
from acp_labelnet.data import (
    MultiLabelPeptideDataset,
    augment_cervical_positives,
    conditional_cooccurrence,
    make_folds,
)
from acp_labelnet.losses import GradNormBalancer, asymmetric_loss_per_label
from acp_labelnet.metrics import best_youden_thresholds, multilabel_metrics, selection_score
from acp_labelnet.model import ACPLabelNet, load_stage1_encoder, load_tokenizer
from acp_labelnet.training import (
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
        output = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["handcrafted"].to(device),
        )
        probabilities.append(torch.sigmoid(output["logits"]).cpu().numpy())
        labels.append(batch["labels"].numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def build_model(args, config, adjacency, device):
    model = ACPLabelNet(
        backbone=args.backbone,
        attention_heads=config.attention_heads,
        attention_dropout=config.attention_dropout,
        handcrafted_projection_dim=config.handcrafted_projection_dim,
        handcrafted_dropout=config.handcrafted_dropout,
        label_dim=config.label_embedding_dim,
        gnn_layers=config.gnn_layers,
        num_labels=len(CANCER_TYPES),
    ).to(device)
    load_stage1_encoder(model, args.stage1_checkpoint, device)
    model.label_graph.set_adjacency(
        torch.as_tensor(adjacency, dtype=torch.float32, device=device), args.gnn_threshold
    )
    freeze_lower_backbone_layers(model, config.trainable_backbone_layers)
    return model


def train_fold(args, config: ExperimentConfig) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir = Path(args.data_dir)
    development = pd.read_csv(data_dir / "train.csv")
    independent_test = pd.read_csv(data_dir / "test.csv")
    folds = make_folds(development, args.n_folds, args.fold_seed, stratify_column="cervical")
    training_indices, validation_indices = folds[args.fold]
    training_frame = development.iloc[training_indices].reset_index(drop=True)
    validation_frame = development.iloc[validation_indices].reset_index(drop=True)

    # Augmentation and graph estimation are confined to the training portion of this fold.
    if not args.no_augmentation:
        training_frame = augment_cervical_positives(training_frame, seed=args.fold)
    adjacency = conditional_cooccurrence(training_frame)

    tokenizer = load_tokenizer(args.backbone)
    training_dataset = MultiLabelPeptideDataset(training_frame, tokenizer, config.max_length)
    validation_dataset = MultiLabelPeptideDataset(validation_frame, tokenizer, config.max_length)
    test_dataset = MultiLabelPeptideDataset(independent_test, tokenizer, config.max_length)
    training_loader = DataLoader(
        training_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args, config, adjacency, device)
    balancer = GradNormBalancer(len(CANCER_TYPES), alpha=args.gradnorm_alpha).to(device)
    model_optimizer = build_optimizer(
        model, args.lr_backbone, args.lr_head, args.weight_decay,
        config.layerwise_lr_decay, args.lr_sequence_projection,
    )
    weight_optimizer = torch.optim.Adam([balancer.weights], lr=args.gradnorm_lr)
    total_steps = args.epochs * len(training_loader)
    scheduler = cosine_warmup_scheduler(
        model_optimizer, int(config.warmup_ratio * total_steps), total_steps
    )

    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    results_dir = Path(args.output_dir) / "results"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"stage2_fold{args.fold}.pt"
    best_score = -1.0
    patience = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in training_loader:
            output = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["handcrafted"].to(device),
            )
            task_losses = asymmetric_loss_per_label(
                output["logits"],
                batch["labels"].to(device),
                args.asl_gamma_positive,
                args.asl_gamma_negative,
                args.asl_clip,
            )
            balancer.initialize(task_losses)
            total_loss = balancer.weighted_loss(task_losses)

            model_optimizer.zero_grad()
            weight_optimizer.zero_grad()
            total_loss.backward(retain_graph=True)
            gradnorm_loss = balancer.gradnorm_objective(task_losses, output["pooled"])
            weight_gradient = torch.autograd.grad(gradnorm_loss, balancer.weights)[0]
            balancer.weights.grad = weight_gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            model_optimizer.step()
            weight_optimizer.step()
            balancer.renormalize()
            scheduler.step()
            running_loss += float(total_loss.detach().item())

        validation_probabilities, validation_labels = predict(model, validation_loader, device)
        thresholds = best_youden_thresholds(validation_labels, validation_probabilities)
        validation_metrics = multilabel_metrics(
            validation_labels, validation_probabilities, thresholds
        )
        score = selection_score(validation_metrics)
        print(
            f"fold={args.fold} epoch={epoch}/{args.epochs} "
            f"loss={running_loss / max(1, len(training_loader)):.4f} "
            f"val_macro_f1={validation_metrics['macro']['f1']:.4f} score={score:.4f}"
        )
        if score > best_score:
            best_score = score
            patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "task_weights": balancer.weights.detach().cpu(),
                "backbone": args.backbone,
                "fold": args.fold,
                "epoch": epoch,
                "thresholds": dict(zip(CANCER_TYPES, thresholds.tolist())),
                "validation_metrics": validation_metrics,
                "selection_score": score,
                "conditional_cooccurrence": adjacency,
            }, checkpoint_path)
        else:
            patience += 1
            if patience >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    thresholds = np.asarray([checkpoint["thresholds"][name] for name in CANCER_TYPES])
    test_probabilities, test_labels = predict(model, test_loader, device)
    test_metrics = multilabel_metrics(test_labels, test_probabilities, thresholds)
    result = {
        "fold": args.fold,
        "best_epoch": checkpoint["epoch"],
        "validation_selection_score": checkpoint["selection_score"],
        "macro_f1": test_metrics["macro"]["f1"],
        "macro_auroc": test_metrics["macro"]["auroc"],
        "macro_auprc": test_metrics["macro"]["auprc"],
    }
    result.update({f"{name}_f1": test_metrics[name]["f1"] for name in CANCER_TYPES})
    save_json(results_dir / f"stage2_fold{args.fold}.json", result)
    np.savez_compressed(
        results_dir / f"stage2_fold{args.fold}_predictions.npz",
        probabilities=test_probabilities,
        labels=test_labels,
        thresholds=thresholds,
        cancer_types=np.asarray(CANCER_TYPES),
    )
    print(f"Independent test macro-F1: {result['macro_f1']:.4f}")


def aggregate(args) -> None:
    results = load_fold_results(Path(args.output_dir) / "results", "stage2_fold{fold}.json", args.n_folds)
    keys = [
        *(f"{name}_f1" for name in CANCER_TYPES),
        "macro_f1", "macro_auroc", "macro_auprc",
    ]
    summary = mean_and_std(results, list(keys))
    save_json(Path(args.output_dir) / "results" / "stage2_summary.json", summary)
    for key, values in summary.items():
        print(f"{key}: {values['mean']:.4f} +/- {values['std']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--data-dir", default="data/stage2")
    parser.add_argument("--stage1-checkpoint", default="outputs/checkpoints/stage1_set2_best.pt")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--backbone", default=ExperimentConfig().backbone)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=5e-4)
    parser.add_argument("--lr-sequence-projection", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gnn-threshold", type=float, default=0.4)
    parser.add_argument("--asl-gamma-positive", type=float, default=1.0)
    parser.add_argument("--asl-gamma-negative", type=float, default=4.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--gradnorm-alpha", type=float, default=1.5)
    parser.add_argument("--gradnorm-lr", type=float, default=0.025)
    parser.add_argument("--no-augmentation", action="store_true")
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
