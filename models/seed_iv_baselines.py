"""Simple, reproducible SEED-IV unimodal and concatenation baselines."""

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class SeedIVBaseline(nn.Module):
    """EEG-only, eye-only, or feature-concatenation baseline."""

    valid_modes = ("eeg", "eye", "concat")

    def __init__(
        self,
        mode: str = "concat",
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        dropout: float = 0.3,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        if mode not in self.valid_modes:
            raise ValueError(f"mode must be one of {self.valid_modes}.")
        self.mode = mode
        self.eeg_encoder = (
            FeatureEncoder(
                input_dim=62 * 5,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
            if mode in ("eeg", "concat")
            else None
        )
        self.eye_encoder = (
            FeatureEncoder(
                input_dim=31,
                hidden_dim=hidden_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
            if mode in ("eye", "concat")
            else None
        )

        classifier_input = embedding_dim * 2 if mode == "concat" else embedding_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, eeg: torch.Tensor, eye: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[1:] != (62, 5):
            raise ValueError(f"EEG input must have shape [B, 62, 5], got {eeg.shape}.")
        if eye.ndim != 2 or eye.shape[1] != 31:
            raise ValueError(f"Eye input must have shape [B, 31], got {eye.shape}.")

        if self.mode == "eeg":
            fused = self.eeg_encoder(eeg.flatten(start_dim=1))
        elif self.mode == "eye":
            fused = self.eye_encoder(eye)
        else:
            fused = torch.cat(
                (
                    self.eeg_encoder(eeg.flatten(start_dim=1)),
                    self.eye_encoder(eye),
                ),
                dim=1,
            )
        return self.classifier(fused)
