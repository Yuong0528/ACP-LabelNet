"""Neural-network modules used by both training stages."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


def load_tokenizer(backbone: str):
    return AutoTokenizer.from_pretrained(backbone)


class AttentionPool(nn.Module):
    """Aggregate residue embeddings with a learned single-query attention pool."""

    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_size)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_size = hidden.shape

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        query = split_heads(self.query.expand(batch_size, -1, -1))
        key = split_heads(self.key(hidden))
        value = split_heads(self.value(hidden))
        scores = (query @ key.transpose(-1, -2)) * self.scale
        mask = attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(F.softmax(scores, dim=-1))
        pooled = (weights @ value).transpose(1, 2).reshape(batch_size, 1, hidden_size)
        pooled = self.norm(self.output(pooled.squeeze(1)))
        return pooled, weights.mean(dim=1).squeeze(1)


class PeptideEncoder(nn.Module):
    def __init__(self, backbone: str, attention_heads: int = 4, attention_dropout: float = 0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone)
        self.hidden_size = self.backbone.config.hidden_size
        self.pool = AttentionPool(self.hidden_size, attention_heads, attention_dropout)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled, residue_attention = self.pool(hidden, attention_mask)
        return {"pooled": pooled, "residue_attention": residue_attention}


class BinaryACPClassifier(PeptideEncoder):
    """Stage 1 binary ACP-recognition model."""

    def __init__(
        self,
        backbone: str,
        attention_heads: int = 4,
        attention_dropout: float = 0.2,
        head_hidden_dim: int = 256,
        head_dropout: float = 0.5,
    ):
        super().__init__(backbone, attention_heads, attention_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden_dim, 2),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(input_ids, attention_mask)
        encoded["logits"] = self.classifier(encoded["pooled"])
        return encoded


class LabelGraphHead(nn.Module):
    """Two-layer label GCN followed by cancer-specific sequence projections."""

    def __init__(
        self,
        sequence_dim: int,
        num_labels: int = 5,
        label_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label_embeddings = nn.Parameter(torch.randn(num_labels, label_dim) * 0.02)
        self.graph_layers = nn.ModuleList(nn.Linear(label_dim, label_dim) for _ in range(num_layers))
        self.sequence_projections = nn.ModuleList(
            nn.Linear(sequence_dim, label_dim) for _ in range(num_labels)
        )
        self.register_buffer("adjacency", torch.eye(num_labels))

    @torch.no_grad()
    def set_adjacency(self, conditional_cooccurrence: torch.Tensor, threshold: float = 0.4) -> None:
        """Threshold the directional graph, add self-loops, and degree-normalize."""
        adjacency = (conditional_cooccurrence >= threshold).float()
        adjacency = torch.maximum(
            adjacency,
            torch.eye(self.num_labels, device=adjacency.device, dtype=adjacency.dtype),
        )
        inverse_sqrt_degree = adjacency.sum(dim=1).clamp_min(1e-6).pow(-0.5)
        normalized = inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]
        self.adjacency.copy_(normalized)

    def forward(self, sequence_embedding: torch.Tensor) -> torch.Tensor:
        label_embedding = self.label_embeddings
        for layer in self.graph_layers:
            label_embedding = F.relu(layer(self.adjacency @ label_embedding))
        logits = [
            (projection(sequence_embedding) * label_embedding[index]).sum(dim=-1)
            for index, projection in enumerate(self.sequence_projections)
        ]
        return torch.stack(logits, dim=-1)


class ACPProtFusion(PeptideEncoder):
    """Stage 2 multi-label model described in the manuscript."""

    def __init__(
        self,
        backbone: str,
        attention_heads: int = 4,
        attention_dropout: float = 0.2,
        handcrafted_dim: int = 426,
        handcrafted_projection_dim: int = 64,
        handcrafted_dropout: float = 0.3,
        label_dim: int = 128,
        gnn_layers: int = 2,
        num_labels: int = 5,
    ):
        super().__init__(backbone, attention_heads, attention_dropout)
        self.handcrafted_projection = nn.Sequential(
            nn.Linear(handcrafted_dim, handcrafted_projection_dim),
            nn.ReLU(),
            nn.Dropout(handcrafted_dropout),
        )
        shared_dim = self.hidden_size + handcrafted_projection_dim
        self.label_graph = LabelGraphHead(shared_dim, num_labels, label_dim, gnn_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        handcrafted: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode(input_ids, attention_mask)
        explicit = self.handcrafted_projection(handcrafted)
        shared = torch.cat((encoded["pooled"], explicit), dim=-1)
        encoded.update({"shared": shared, "logits": self.label_graph(shared)})
        return encoded


def load_stage1_encoder(model: ACPProtFusion, checkpoint_path: str, device: torch.device) -> None:
    """Warm-start the Stage 2 encoder and attention pool from a Stage 1 checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = checkpoint.get("model_state_dict", checkpoint)
    compatible = {
        key: value
        for key, value in source.items()
        if key.startswith(("backbone.", "pool."))
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected Stage 1 checkpoint keys: {unexpected}")
    if not compatible or not any(key.startswith("backbone.") for key in compatible):
        raise RuntimeError("The Stage 1 checkpoint contains no compatible encoder weights")
    print(f"Loaded {len(compatible)} Stage 1 tensors; {len(missing)} Stage 2 tensors initialized anew")

