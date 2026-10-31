"""EEG-only neural baselines for SEED-IV differential-entropy features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adjacency_logits: torch.Tensor) -> torch.Tensor:
    """Return a finite, symmetric, degree-normalized adjacency matrix."""
    if adjacency_logits.ndim != 2 or adjacency_logits.shape[0] != adjacency_logits.shape[1]:
        raise ValueError("adjacency_logits must be a square matrix.")

    adjacency = F.softplus(
        (adjacency_logits + adjacency_logits.transpose(0, 1)) * 0.5
    )
    adjacency = adjacency + torch.eye(
        adjacency.shape[0],
        dtype=adjacency.dtype,
        device=adjacency.device,
    )
    inverse_sqrt_degree = adjacency.sum(dim=1).clamp_min(1e-8).rsqrt()
    return (
        inverse_sqrt_degree[:, None]
        * adjacency
        * inverse_sqrt_degree[None, :]
    )


class ChebyshevGraphConvolution(nn.Module):
    """A compact order-K Chebyshev-style graph convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        order: int = 2,
    ) -> None:
        super().__init__()
        if order < 1:
            raise ValueError("order must be at least 1.")
        self.order = order
        self.projections = nn.ModuleList(
            nn.Linear(in_channels, out_channels, bias=False)
            for _ in range(order)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        for projection in self.projections:
            nn.init.xavier_normal_(projection.weight)

    def forward(
        self,
        features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        supports = [features]
        if self.order > 1:
            supports.append(torch.matmul(adjacency, features))
        for _ in range(2, self.order):
            supports.append(
                2.0 * torch.matmul(adjacency, supports[-1]) - supports[-2]
            )

        output = sum(
            projection(support)
            for projection, support in zip(self.projections, supports)
        )
        return output + self.bias


class SeedIVDGCNN(nn.Module):
    """DGCNN baseline for aligned SEED-IV DE features shaped ``[B, 62, 5]``."""

    def __init__(
        self,
        in_channels: int = 5,
        num_electrodes: int = 62,
        graph_channels: int = 32,
        chebyshev_order: int = 2,
        classifier_dim: int = 64,
        dropout: float = 0.3,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_electrodes = num_electrodes
        self.feature_norm = nn.BatchNorm1d(in_channels)
        self.adjacency_logits = nn.Parameter(
            torch.empty(num_electrodes, num_electrodes)
        )
        nn.init.xavier_normal_(self.adjacency_logits)
        self.graph_convolution = ChebyshevGraphConvolution(
            in_channels=in_channels,
            out_channels=graph_channels,
            order=chebyshev_order,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(num_electrodes * graph_channels, classifier_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, num_classes),
        )

    def normalized_adjacency(self) -> torch.Tensor:
        return normalize_adjacency(self.adjacency_logits)

    def forward(
        self,
        eeg: torch.Tensor,
        eye: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del eye
        expected_shape = (self.num_electrodes, self.in_channels)
        if eeg.ndim != 3 or tuple(eeg.shape[1:]) != expected_shape:
            raise ValueError(
                f"EEG input must have shape [B, {expected_shape[0]}, "
                f"{expected_shape[1]}], got {tuple(eeg.shape)}."
            )

        features = self.feature_norm(eeg.transpose(1, 2)).transpose(1, 2)
        features = self.graph_convolution(
            features,
            self.normalized_adjacency(),
        )
        features = self.dropout(F.gelu(features))
        return self.classifier(features.flatten(start_dim=1))
