"""BERN branches and vector-token cross-attention."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .data import eeg_to_grid


RESNET18_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
RESNET18_HASH = "f37072fd"


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Use running statistics for a singleton final batch."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values_per_channel = inputs.numel() // inputs.shape[1]
        if self.training and values_per_channel == 1:
            return F.batch_norm(
                inputs, self.running_mean, self.running_var, self.weight, self.bias,
                False, self.momentum, self.eps,
            )
        return super().forward(inputs)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample: nn.Module | None = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return self.relu(output + identity)


class ResNet18ResidualLayers(nn.Module):
    """The four residual stages of ResNet-18, excluding its image stem and head."""

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self._initialize()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [BasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        layers.extend(BasicBlock(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.layer1(inputs)
        output = self.layer2(output)
        output = self.layer3(output)
        return self.layer4(output)

    def load_imagenet_weights(
        self,
        cache_dir: str | Path = "BERN/cache",
        weights_path: str | Path | None = None,
        progress: bool = True,
    ) -> dict[str, Any]:
        if weights_path is None:
            cache = Path(cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            state = torch.hub.load_state_dict_from_url(
                RESNET18_URL,
                model_dir=str(cache),
                progress=progress,
                check_hash=True,
                map_location="cpu",
            )
            source = RESNET18_URL
        else:
            state = torch.load(weights_path, map_location="cpu", weights_only=True)
            source = str(weights_path)
        expected = self.state_dict()
        # Some official torchvision checkpoints predate serialization of the
        # BatchNorm batch counters. They are non-parameter integer buffers and
        # are intentionally left at their local zero initialization.
        required = {
            key: value
            for key, value in expected.items()
            if not key.endswith(".num_batches_tracked")
        }
        filtered = {key: value for key, value in state.items() if key in expected}
        missing = sorted(set(required) - set(filtered))
        mismatched = sorted(
            key for key, value in filtered.items() if value.shape != expected[key].shape
        )
        if missing or mismatched:
            raise RuntimeError(
                f"Incompatible ResNet-18 residual weights: missing={missing}, "
                f"shape_mismatch={mismatched}"
            )
        merged = dict(expected)
        merged.update(filtered)
        self.load_state_dict(merged, strict=True)
        return {
            "source": source,
            "sha256_prefix": RESNET18_HASH,
            "loaded_keys": len(set(required) & set(filtered)),
            "expected_keys": len(required),
            "initialized_batch_counters": sorted(set(expected) - set(required)),
        }


class EEGEncoder(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        cache_dir: str | Path = "BERN/cache",
        weights_path: str | Path | None = None,
    ):
        super().__init__()
        self.frequency_spatial_conv = nn.Conv3d(
            1, 64, kernel_size=(5, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False
        )
        self.bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.residual_layers = ResNet18ResidualLayers()
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(512, 128)
        self.pretrained_report: dict[str, Any] | None = None
        if pretrained:
            self.pretrained_report = self.residual_layers.load_imagenet_weights(
                cache_dir=cache_dir, weights_path=weights_path
            )

    def forward_features(
        self, eeg: torch.Tensor, return_shapes: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        grid = eeg_to_grid(eeg)
        resized = F.interpolate(grid, size=(112, 112), mode="bilinear", align_corners=False)
        conv3d = self.relu(self.bn(self.frequency_spatial_conv(resized.unsqueeze(1))))
        stem = conv3d.squeeze(2)
        residual = self.residual_layers(self.pool(stem))
        embedding = self.projection(torch.flatten(self.average_pool(residual), 1))
        if return_shapes:
            return embedding, {
                "grid": tuple(grid.shape),
                "resized": tuple(resized.shape),
                "conv3d": tuple(conv3d.shape),
                "stem": tuple(stem.shape),
                "residual": tuple(residual.shape),
                "embedding": tuple(embedding.shape),
            }
        return embedding

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.forward_features(eeg)


class EyeResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(31, 15)
        self.bn1 = SafeBatchNorm1d(15)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(15, 31)
        self.bn2 = SafeBatchNorm1d(31)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.sigmoid(self.bn2(self.linear2(self.relu(self.bn1(self.linear1(inputs))))))
        return inputs + residual


class EyeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(EyeResidualBlock() for _ in range(4))
        self.conv = nn.Conv1d(31, 128, kernel_size=1, bias=False)
        self.bn = SafeBatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)

    def forward_features(
        self, eye: torch.Tensor, return_shapes: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, tuple[int, ...]]]:
        if eye.ndim != 2 or eye.shape[1] != 31:
            raise ValueError(f"Expected eye shape [B,31], received {tuple(eye.shape)}")
        output = eye
        shapes: dict[str, tuple[int, ...]] = {}
        for index, block in enumerate(self.blocks, start=1):
            output = block(output)
            shapes[f"block{index}"] = tuple(output.shape)
        embedding = self.relu(self.bn(self.conv(output.unsqueeze(-1)))).squeeze(-1)
        shapes["embedding"] = tuple(embedding.shape)
        if return_shapes:
            return embedding, shapes
        return embedding

    def forward(self, eye: torch.Tensor) -> torch.Tensor:
        return self.forward_features(eye)


class VectorTokenCrossAttention(nn.Module):
    """Pre-norm multi-head cross-attention over vector-valued tokens."""

    def __init__(self, token_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.query_norm = nn.LayerNorm(token_dim)
        self.context_norm = nn.LayerNorm(token_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        query_tokens: torch.Tensor,
        eye_tokens: torch.Tensor,
        return_attention: bool = False,
    ):
        normalized_query = self.query_norm(query_tokens)
        normalized_eye = self.context_norm(eye_tokens)
        cross, weights = self.attention(
            normalized_query,
            normalized_eye,
            normalized_eye,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        output = self.output_norm(query_tokens + cross)
        return (output, weights) if return_attention else output


class BERN(nn.Module):
    """BERN(mode="eye"|"eeg"|"fusion") with four-class logits."""

    MODES = ("eye", "eeg", "fusion")

    def __init__(
        self,
        mode: str = "fusion",
        pretrained: bool = True,
        cache_dir: str | Path = "BERN/cache",
        weights_path: str | Path | None = None,
        fusion_num_tokens: int = 8,
        fusion_token_dim: int = 32,
        fusion_num_heads: int = 4,
        fusion_variant: str = "vector_token_v2",
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, received {mode!r}")
        if fusion_variant != "vector_token_v2":
            raise ValueError(
                "Only fusion_variant='vector_token_v2' is supported, received "
                f"{fusion_variant!r}"
            )
        if fusion_num_tokens < 1 or fusion_token_dim < 1:
            raise ValueError("Fusion token count and dimension must be positive.")
        if fusion_token_dim % fusion_num_heads:
            raise ValueError("fusion_token_dim must be divisible by fusion_num_heads.")
        self.mode = mode
        self.fusion_num_tokens = fusion_num_tokens
        self.fusion_token_dim = fusion_token_dim
        self.fusion_num_heads = fusion_num_heads
        self.fusion_variant = fusion_variant
        self.eeg_encoder = (
            EEGEncoder(pretrained, cache_dir, weights_path) if mode in ("eeg", "fusion") else None
        )
        self.eye_encoder = EyeEncoder() if mode in ("eye", "fusion") else None
        if mode == "fusion":
            token_features = fusion_num_tokens * fusion_token_dim
            self.eeg_token_projection = nn.Linear(128, token_features)
            self.eye_token_projection = nn.Linear(128, token_features)
            self.cross_attention = nn.ModuleList(
                VectorTokenCrossAttention(fusion_token_dim, fusion_num_heads)
                for _ in range(2)
            )
            self.fusion_projection = nn.Linear(token_features, 128)
        else:
            self.eeg_token_projection = None
            self.eye_token_projection = None
            self.cross_attention = None
            self.fusion_projection = None
        self.classifier = nn.Linear(128, 4)

    def forward(
        self,
        eeg: torch.Tensor,
        eye: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if return_attention and self.mode != "fusion":
            raise ValueError("Attention weights are available only in fusion mode.")
        attention_weights: list[torch.Tensor] = []
        if self.mode == "eeg":
            features = self.eeg_encoder(eeg)
        elif self.mode == "eye":
            features = self.eye_encoder(eye)
        else:
            eeg_features = self.eeg_encoder(eeg)
            eye_features = self.eye_encoder(eye)
            features = self.eeg_token_projection(eeg_features).reshape(
                -1, self.fusion_num_tokens, self.fusion_token_dim
            )
            eye_tokens = self.eye_token_projection(eye_features).reshape(
                -1, self.fusion_num_tokens, self.fusion_token_dim
            )
            for layer in self.cross_attention:
                if return_attention:
                    features, weights = layer(
                        features, eye_tokens, return_attention=True
                    )
                    attention_weights.append(weights)
                else:
                    features = layer(features, eye_tokens)
            features = self.fusion_projection(torch.flatten(features, 1))
        logits = self.classifier(features)
        return (logits, attention_weights) if return_attention else logits
