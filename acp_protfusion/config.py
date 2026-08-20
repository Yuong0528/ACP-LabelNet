"""Central, public configuration for the two-stage training pipeline."""

from __future__ import annotations

from dataclasses import dataclass


CANCER_TYPES = ("breast", "lung", "colon", "cervical", "skin")


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters used for the experiments reported in the manuscript."""

    backbone: str = "facebook/esm2_t33_650M_UR50D"
    max_length: int = 64
    attention_heads: int = 4
    attention_dropout: float = 0.2
    head_hidden_dim: int = 256
    head_dropout: float = 0.5
    label_embedding_dim: int = 128
    gnn_layers: int = 2
    gnn_threshold: float = 0.4
    handcrafted_projection_dim: int = 64
    handcrafted_dropout: float = 0.3
    trainable_backbone_layers: int = 4
    layerwise_lr_decay: float = 0.9
    warmup_ratio: float = 0.1
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    num_workers: int = 4
    fold_seed: int = 0
    training_seed: int = 42

