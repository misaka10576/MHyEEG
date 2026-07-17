"""Cross-modal attention fusion for modality-level feature vectors."""

from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn


class CrossModalAttentionBlock(nn.Module):
    """Synchronously update modality tokens using only the other modalities."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return updated tokens and per-head cross-attention weights."""
        modality_count = tokens.size(1)
        if modality_count < 2:
            raise ValueError("Cross-attention requires at least two modalities.")

        normalized_queries = self.query_norm(tokens)
        normalized_context = self.context_norm(tokens)
        updated: List[torch.Tensor] = []
        weights: List[torch.Tensor] = []

        # Every query reads the same pre-update state, so modality order does not
        # introduce sequential update bias.
        for query_index in range(modality_count):
            context_indices = [
                index for index in range(modality_count) if index != query_index
            ]
            query = normalized_queries[:, query_index : query_index + 1, :]
            context = normalized_context[:, context_indices, :]
            attended, attention = self.attention(
                query=query,
                key=context,
                value=context,
                need_weights=True,
                average_attn_weights=False,
            )
            token = tokens[:, query_index : query_index + 1, :]
            token = token + self.attention_dropout(attended)
            token = token + self.ffn(self.ffn_norm(token))
            updated.append(token.squeeze(1))
            weights.append(attention.squeeze(2))

        return torch.stack(updated, dim=1), torch.stack(weights, dim=1)


class CrossModalAttentionFusion(nn.Module):
    """Fuse differently sized modality vectors into one representation."""

    def __init__(
        self,
        input_dims: Sequence[int],
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        output_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError("input_dims must contain at least two modalities.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        self.input_dims = tuple(input_dims)
        self.modality_count = len(input_dims)
        self.projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for input_dim in input_dims
        )
        self.modality_embedding = nn.Parameter(
            torch.empty(1, self.modality_count, d_model)
        )
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)

        self.blocks = nn.ModuleList(
            CrossModalAttentionBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        )
        self.gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.output = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        modality_features: Sequence[torch.Tensor],
        return_attention: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        if len(modality_features) != self.modality_count:
            raise ValueError(
                f"Expected {self.modality_count} modalities, "
                f"received {len(modality_features)}."
            )

        projected = []
        batch_size = modality_features[0].size(0)
        for index, (feature, expected_dim, projection) in enumerate(
            zip(modality_features, self.input_dims, self.projections)
        ):
            if feature.ndim != 2 or feature.size(0) != batch_size:
                raise ValueError(
                    f"Modality {index} must have shape [batch, feature]."
                )
            if feature.size(1) != expected_dim:
                raise ValueError(
                    f"Modality {index} must have {expected_dim} features, "
                    f"received {feature.size(1)}."
                )
            projected.append(projection(feature))

        tokens = torch.stack(projected, dim=1) + self.modality_embedding
        layer_attention = []
        for block in self.blocks:
            tokens, attention = block(tokens)
            layer_attention.append(attention)

        pooling_weights = torch.softmax(self.gate(tokens), dim=1)
        pooled = torch.sum(tokens * pooling_weights, dim=1)
        fused = self.output(pooled)
        if not return_attention:
            return fused

        diagnostics = {
            # [layer, batch, query_modality, head, other_modality]
            "cross_attention": torch.stack(layer_attention, dim=0),
            # [batch, modality]
            "pooling_weights": pooling_weights.squeeze(-1),
        }
        return fused, diagnostics
