"""SEED-IV adaptation of the Multimodal Adaptive Emotion Transformer.

This implementation is adapted from MAET:
"Multimodal Adaptive Emotion Transformer with Flexible Modality Inputs on
A Novel Dataset with Continuous Labels". The original implementation in
``MAET/model.py`` is distributed under the MIT License; see ``MAET/LICENSE``.

The adaptation removes the timm dependency and changes the inputs and output
head to the aligned SEED-IV EEG/eye features and four emotion classes.

Portions copyright (c) 2023 Weibang Jiang.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class DropPath(nn.Module):
    """Drop residual paths independently per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ReverseLayerF(Function):
    """Identity in the forward pass and gradient reversal in the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("q_bias", None)
            self.register_parameter("v_bias", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (
                    self.q_bias,
                    torch.zeros_like(self.q_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            channels // self.num_heads,
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale).float() @ key.float().transpose(-2, -1)
        attention = self.attn_drop(attention.softmax(dim=-1).type_as(x))
        x = (attention @ value).transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            channels,
        )
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    """Shared attention with modality-specific FFNs and late Mix-FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        norm_layer,
        with_mixffn: bool,
        layer_scale_init_values: float | None,
        eeg_token_count: int,
    ) -> None:
        super().__init__()
        self.eeg_token_count = eeg_token_count
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        hidden_dim = int(dim * mlp_ratio)
        self.norm2_eeg = norm_layer(dim)
        self.norm2_eye = norm_layer(dim)
        self.mlp_eeg = Mlp(dim, hidden_dim, drop)
        self.mlp_eye = Mlp(dim, hidden_dim, drop)
        self.norm2_mix = norm_layer(dim) if with_mixffn else None
        self.mlp_mix = Mlp(dim, hidden_dim, drop) if with_mixffn else None
        if layer_scale_init_values is None:
            self.gamma_1 = 1.0
            self.gamma_2 = 1.0
        else:
            self.gamma_1 = nn.Parameter(
                layer_scale_init_values * torch.ones(dim)
            )
            self.gamma_2 = nn.Parameter(
                layer_scale_init_values * torch.ones(dim)
            )

    def forward(
        self,
        x: torch.Tensor,
        modality_type: str | None,
    ) -> torch.Tensor:
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        if modality_type == "eeg":
            update = self.mlp_eeg(self.norm2_eeg(x))
            return x + self.drop_path(self.gamma_2 * update)
        if modality_type == "eye":
            update = self.mlp_eye(self.norm2_eye(x))
            return x + self.drop_path(self.gamma_2 * update)
        if self.mlp_mix is not None:
            update = self.mlp_mix(self.norm2_mix(x))
            return x + self.drop_path(self.gamma_2 * update)

        eeg = x[:, : self.eeg_token_count]
        eye = x[:, self.eeg_token_count :]
        eeg = eeg + self.drop_path(
            self.gamma_2 * self.mlp_eeg(self.norm2_eeg(eeg))
        )
        eye = eye + self.drop_path(
            self.gamma_2 * self.mlp_eye(self.norm2_eye(eye))
        )
        return torch.cat((eeg, eye), dim=1)


class MultiViewEmbedding(nn.Module):
    """Project one feature vector into several gated view tokens."""

    def __init__(self, input_dim: int, output_dim: int, views: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.views = views
        self.transform = nn.Linear(input_dim, output_dim)
        self.gates = nn.Linear(input_dim, output_dim * views)
        self.batch_norm = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        content = self.transform(x).unsqueeze(1).expand(-1, self.views, -1)
        gates = self.gates(x).sigmoid().reshape(
            batch_size,
            self.views,
            self.output_dim,
        )
        x = content * gates
        return self.batch_norm(x.transpose(1, 2)).transpose(1, 2)


class AdaptiveFusion(nn.Module):
    """Learn a sample-specific convex combination of EEG and eye CLS tokens."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, 1))

    def forward(
        self,
        eeg: torch.Tensor,
        eye: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.cat((eeg @ self.weight, eye @ self.weight), dim=1)
        weights = scores.softmax(dim=1)
        return eeg * weights[:, :1] + eye * weights[:, 1:]


class SeedIVMAET(nn.Module):
    """MAET for SEED-IV EEG ``[B, 62, 5]`` and eye ``[B, 31]`` features."""

    valid_modes = ("eeg", "eye", "concat")

    def __init__(
        self,
        mode: str = "concat",
        eeg_dim: int = 310,
        eye_dim: int = 31,
        num_classes: int = 4,
        embed_dim: int = 32,
        depth: int = 3,
        eeg_seq_len: int = 5,
        eye_seq_len: int = 5,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_scale_init_values: float | None = 0.1,
        mixffn_start_layer_index: int = 2,
        domain_generalization: bool = False,
        num_domains: int = 13,
    ) -> None:
        super().__init__()
        if mode not in self.valid_modes:
            raise ValueError(f"mode must be one of {self.valid_modes}.")
        if depth <= 0:
            raise ValueError("depth must be positive.")
        if not 0 <= mixffn_start_layer_index <= depth:
            raise ValueError("mixffn_start_layer_index must be in [0, depth].")
        if domain_generalization and num_domains <= 1:
            raise ValueError("num_domains must be greater than one for DG.")

        self.mode = mode
        self.eeg_dim = eeg_dim
        self.eye_dim = eye_dim
        self.eeg_seq_len = eeg_seq_len
        self.eye_seq_len = eye_seq_len
        self.domain_generalization = domain_generalization

        self.eeg_transform = MultiViewEmbedding(
            eeg_dim,
            embed_dim,
            eeg_seq_len,
        )
        self.eye_transform = MultiViewEmbedding(
            eye_dim,
            embed_dim,
            eye_seq_len,
        )
        self.eeg_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eye_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eeg_pos_embed = nn.Parameter(
            torch.zeros(1, eeg_seq_len + 1, embed_dim)
        )
        self.eye_pos_embed = nn.Parameter(
            torch.zeros(1, eye_seq_len + 1, embed_dim)
        )
        self.eeg_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eye_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        drop_paths = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_paths[index],
                    norm_layer=norm_layer,
                    with_mixffn=index >= mixffn_start_layer_index,
                    layer_scale_init_values=layer_scale_init_values,
                    eeg_token_count=eeg_seq_len + 1,
                )
                for index in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.fusion = AdaptiveFusion(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_eeg = nn.Linear(embed_dim, num_classes)
        self.head_eye = nn.Linear(embed_dim, num_classes)
        self.domain_classifier = (
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_domains),
            )
            if domain_generalization
            else None
        )

        for parameter in (
            self.eeg_cls_token,
            self.eye_cls_token,
            self.eeg_pos_embed,
            self.eye_pos_embed,
            self.eeg_type_embed,
            self.eye_type_embed,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "eeg_pos_embed",
            "eye_pos_embed",
            "eeg_cls_token",
            "eye_cls_token",
            "eeg_type_embed",
            "eye_type_embed",
        }

    def _embed_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[1:] != (62, 5):
            raise ValueError(
                f"EEG input must have shape [B, 62, 5], got {tuple(eeg.shape)}."
            )
        if eeg.shape[1] * eeg.shape[2] != self.eeg_dim:
            raise ValueError("Configured eeg_dim does not match the EEG input.")
        tokens = self.eeg_transform(eeg.flatten(start_dim=1))
        cls = self.eeg_cls_token.expand(eeg.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        return tokens + self.eeg_pos_embed + self.eeg_type_embed

    def _embed_eye(self, eye: torch.Tensor) -> torch.Tensor:
        if eye.ndim != 2 or eye.shape[1] != self.eye_dim:
            raise ValueError(
                f"Eye input must have shape [B, {self.eye_dim}], "
                f"got {tuple(eye.shape)}."
            )
        tokens = self.eye_transform(eye)
        cls = self.eye_cls_token.expand(eye.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        return tokens + self.eye_pos_embed + self.eye_type_embed

    def forward_features(
        self,
        eeg: torch.Tensor | None,
        eye: torch.Tensor | None,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.mode == "eeg":
            if eeg is None:
                raise ValueError("EEG input is required in eeg mode.")
            x = self._embed_eeg(eeg)
            modality_type = "eeg"
        elif self.mode == "eye":
            if eye is None:
                raise ValueError("Eye input is required in eye mode.")
            x = self._embed_eye(eye)
            modality_type = "eye"
        else:
            if eeg is None or eye is None:
                raise ValueError("Both EEG and eye inputs are required in concat mode.")
            if eeg.shape[0] != eye.shape[0]:
                raise ValueError("EEG and eye batch sizes must match.")
            x = torch.cat((self._embed_eeg(eeg), self._embed_eye(eye)), dim=1)
            modality_type = None

        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x, modality_type)
        x = self.norm(x)

        domain_logits = None
        if self.domain_classifier is not None:
            domain_logits = self.domain_classifier(
                ReverseLayerF.apply(x[:, 0], alpha)
            )

        if modality_type == "eeg":
            emotion_logits = self.head_eeg(x[:, 1:].mean(dim=1))
        elif modality_type == "eye":
            emotion_logits = self.head_eye(x[:, 1:].mean(dim=1))
        else:
            eeg_cls = x[:, 0]
            eye_cls = x[:, self.eeg_seq_len + 1]
            emotion_logits = self.head(self.fusion(eeg_cls, eye_cls))
        return emotion_logits, domain_logits

    def forward(
        self,
        eeg: torch.Tensor | None,
        eye: torch.Tensor | None,
        alpha: float = 0.0,
    ):
        emotion_logits, domain_logits = self.forward_features(eeg, eye, alpha)
        if domain_logits is None:
            return emotion_logits
        return emotion_logits, domain_logits
