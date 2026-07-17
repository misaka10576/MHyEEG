"""H2 encoders with cross-modal attention replacing concatenation fusion."""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cross_attention_fusion import CrossModalAttentionFusion
from models.h2 import ECGPHBase, EEGPHBase, GSRPHBase, eyePHBase
from models.hypercomplex_layers import PHMLinear


class CrossAttentionH2(nn.Module):
    """H2 with modality-level cross-attention and three-class logits."""

    modality_names = ("eye", "gsr", "eeg", "ecg")
    encoder_output_dims = (256, 130, 2040, 1026)

    def __init__(
        self,
        dropout_rate: float = 0.5,
        units: int = 1024,
        n: int = 4,
        n_eye: int = 4,
        n_gsr: int = 1,
        n_eeg: int = 10,
        n_ecg: int = 3,
        attention_dim: int = 256,
        attention_heads: int = 8,
        attention_layers: int = 2,
        attention_dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.eye = eyePHBase(n=n_eye)
        self.gsr = GSRPHBase(n=n_gsr)
        self.eeg = EEGPHBase(n=n_eeg)
        self.ecg = ECGPHBase(n=n_ecg)

        # This replaces H2's cat([eye, gsr, eeg, ecg]) + D2 operation.
        self.fusion = CrossModalAttentionFusion(
            input_dims=self.encoder_output_dims,
            d_model=attention_dim,
            num_heads=attention_heads,
            num_layers=attention_layers,
            output_dim=units,
            dropout=attention_dropout,
        )

        # Keep the remaining H2 PHM classifier unchanged.
        self.BN2 = nn.BatchNorm1d(units)
        self.drop1 = nn.Dropout(dropout_rate)
        self.D3 = PHMLinear(n, units, units // 2)
        self.BN3 = nn.BatchNorm1d(units // 2)
        self.drop2 = nn.Dropout(dropout_rate)
        self.D4 = PHMLinear(n, units // 2, units // 4)
        self.drop3 = nn.Dropout(dropout_rate)
        self.out_3 = nn.Linear(units // 4, num_classes)

    def encode_modalities(
        self,
        eye: torch.Tensor,
        gsr: torch.Tensor,
        eeg: torch.Tensor,
        ecg: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.eye(eye), self.gsr(gsr), self.eeg(eeg), self.ecg(ecg)

    def classify(self, fused: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.BN2(fused))
        x = self.drop1(x)
        x = F.relu(self.BN3(self.D3(x)))
        x = self.drop2(x)
        x = F.relu(self.D4(x))
        x = self.drop3(x)
        return self.out_3(x)

    def get_features(self, eye, gsr, eeg, ecg, level: str = "fusion"):
        """Extract per-modality, fused, or pre-logit classifier features."""
        if level not in ("encoder", "fusion", "classifier"):
            raise ValueError("level must be encoder, fusion, or classifier.")

        features = self.encode_modalities(eye, gsr, eeg, ecg)
        if level == "encoder":
            return dict(zip(self.modality_names, features))

        fused = self.fusion(features)
        if level == "fusion":
            return fused

        x = F.relu(self.BN2(fused))
        x = F.relu(self.BN3(self.D3(x)))
        return F.relu(self.D4(x))

    def forward(
        self,
        eye,
        gsr,
        eeg,
        ecg,
        return_attention: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        features = self.encode_modalities(eye, gsr, eeg, ecg)
        if return_attention:
            fused, diagnostics = self.fusion(features, return_attention=True)
            return self.classify(fused), diagnostics
        return self.classify(self.fusion(features))
