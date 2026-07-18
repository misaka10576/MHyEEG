"""Uncertainty-aware attention for aligned SEED-IV EEG and eye features."""

import math
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


EYE_GROUP_SLICES = (
    slice(0, 12),   # pupil diameter
    slice(12, 16),  # dispersion
    slice(16, 18),  # fixation duration
    slice(18, 22),  # saccade
    slice(22, 31),  # event statistics
)


def _inverse_softplus(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


class IntervalType2FuzzyReliability(nn.Module):
    """Differentiable IT2 Gaussian rules with Karnik-Mendel type reduction."""

    def __init__(
        self,
        num_rules: int = 8,
        uncertainty_penalty: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_rules < 2:
            raise ValueError("num_rules must be at least 2.")
        if uncertainty_penalty < 0:
            raise ValueError("uncertainty_penalty must be non-negative.")

        self.num_rules = num_rules
        self.uncertainty_penalty = uncertainty_penalty
        self.epsilon = epsilon

        initial_centers = torch.linspace(0.1, 0.9, num_rules)
        initial_centers = torch.logit(initial_centers).repeat(3, 1)
        self.center_logits = nn.Parameter(initial_centers)
        self.sigma_raw = nn.Parameter(
            torch.full((3, num_rules), _inverse_softplus(0.15))
        )
        self.center_delta_logits = nn.Parameter(
            torch.full((3, num_rules), torch.logit(torch.tensor(0.12)))
        )
        self.sigma_delta_logits = nn.Parameter(
            torch.full((3, num_rules), torch.logit(torch.tensor(0.25)))
        )
        initial_consequents = torch.linspace(0.05, 0.95, num_rules)
        self.consequent_logits = nn.Parameter(torch.logit(initial_consequents))

    def _normalize_statistics(self, statistics: torch.Tensor) -> torch.Tensor:
        minimum = statistics.amin(dim=1, keepdim=True)
        maximum = statistics.amax(dim=1, keepdim=True)
        return (statistics - minimum) / (maximum - minimum + self.epsilon)

    def _membership_intervals(
        self,
        normalized: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        centers = torch.sigmoid(self.center_logits)
        sigma = F.softplus(self.sigma_raw) + 0.03
        center_delta = 0.25 * torch.sigmoid(self.center_delta_logits)
        sigma_delta = 0.5 * sigma * torch.sigmoid(self.sigma_delta_logits)

        center_a = centers - center_delta
        center_b = centers + center_delta
        sigma_a = (sigma - sigma_delta).clamp_min(0.02)
        sigma_b = sigma + sigma_delta

        values = normalized.unsqueeze(-1)
        membership_a = torch.exp(
            -0.5 * ((values - center_a) / sigma_a).square()
        )
        membership_b = torch.exp(
            -0.5 * ((values - center_b) / sigma_b).square()
        )
        lower = torch.minimum(membership_a, membership_b)
        upper = torch.maximum(membership_a, membership_b)
        return lower, upper

    def _km_endpoint(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        consequents: torch.Tensor,
        left_endpoint: bool,
    ) -> torch.Tensor:
        sorted_consequents, order = torch.sort(consequents)
        lower = lower.index_select(-1, order)
        upper = upper.index_select(-1, order)

        midpoint_weights = 0.5 * (lower + upper)
        output = (
            (midpoint_weights * sorted_consequents).sum(dim=-1)
            / midpoint_weights.sum(dim=-1).clamp_min(self.epsilon)
        )
        rule_indices = torch.arange(
            self.num_rules,
            device=lower.device,
        )

        # At most R updates are needed for a stable switch point.
        for _ in range(self.num_rules):
            switch = (
                (sorted_consequents <= output.unsqueeze(-1)).sum(dim=-1) - 1
            ).clamp(0, self.num_rules - 1)
            lower_side = rule_indices <= switch.unsqueeze(-1)
            if left_endpoint:
                weights = torch.where(lower_side, upper, lower)
            else:
                weights = torch.where(lower_side, lower, upper)
            output = (
                (weights * sorted_consequents).sum(dim=-1)
                / weights.sum(dim=-1).clamp_min(self.epsilon)
            )
        return output

    def forward(
        self,
        statistics: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if statistics.ndim != 3 or statistics.size(-1) != 3:
            raise ValueError(
                "statistics must have shape [batch, token, 3]."
            )
        normalized = self._normalize_statistics(statistics)
        lower_membership, upper_membership = self._membership_intervals(
            normalized
        )
        lower_firing = lower_membership.prod(dim=2)
        upper_firing = upper_membership.prod(dim=2)
        consequents = torch.sigmoid(self.consequent_logits)

        left = self._km_endpoint(
            lower_firing,
            upper_firing,
            consequents,
            left_endpoint=True,
        )
        right = self._km_endpoint(
            lower_firing,
            upper_firing,
            consequents,
            left_endpoint=False,
        )
        interval_lower = torch.minimum(left, right)
        interval_upper = torch.maximum(left, right)
        midpoint = 0.5 * (interval_lower + interval_upper)
        uncertainty = interval_upper - interval_lower
        reliability = torch.softmax(
            midpoint - self.uncertainty_penalty * uncertainty,
            dim=1,
        )
        return reliability, {
            "interval_lower": interval_lower,
            "interval_upper": interval_upper,
            "uncertainty": uncertainty,
            "midpoint": midpoint,
        }


class ReliabilityGuidedSelfAttention(nn.Module):
    """Multi-head self-attention with an optional fuzzy key prior."""

    def __init__(self, dimension: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dimension % num_heads != 0:
            raise ValueError("dimension must be divisible by num_heads.")
        self.dimension = dimension
        self.num_heads = num_heads
        self.head_dimension = dimension // num_heads
        self.scale = self.head_dimension ** -0.5
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(
            batch,
            tokens,
            self.num_heads,
            self.head_dimension,
        ).transpose(1, 2)

    def forward(
        self,
        tokens: torch.Tensor,
        reliability: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self._split_heads(self.query(tokens))
        key = self._split_heads(self.key(tokens))
        value = self._split_heads(self.value(tokens))
        logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if reliability is not None:
            if reliability.shape != tokens.shape[:2]:
                raise ValueError(
                    "reliability must have shape [batch, token]."
                )
            logits = logits + torch.log(
                reliability.clamp_min(1e-6)
            )[:, None, None, :]

        attention = torch.softmax(logits, dim=-1)
        attended = torch.matmul(self.dropout(attention), value)
        attended = attended.transpose(1, 2).contiguous().view(
            tokens.size(0),
            tokens.size(1),
            self.dimension,
        )
        return self.output(attended), attention


class MultimodalAttentionBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        num_heads: int,
        ffn_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = ReliabilityGuidedSelfAttention(
            dimension,
            num_heads,
            dropout,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, ffn_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        reliability: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attended, attention = self.attention(
            self.attention_norm(tokens),
            reliability,
        )
        tokens = tokens + self.attention_dropout(attended)
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens, attention


class SeedIVFuzzyAttention(nn.Module):
    """Joint EEG/eye attention, optionally guided by IT2 fuzzy reliability."""

    def __init__(
        self,
        dimension: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        ffn_dimension: int = 128,
        dropout: float = 0.2,
        num_classes: int = 4,
        use_fuzzy: bool = True,
        fuzzy_rules: int = 8,
        uncertainty_penalty: float = 1.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        self.use_fuzzy = use_fuzzy
        self.eeg_projection = nn.Sequential(
            nn.Linear(62, dimension),
            nn.LayerNorm(dimension),
            nn.GELU(),
        )
        self.eye_projections = nn.ModuleList(
            nn.Sequential(
                nn.Linear(group.stop - group.start, dimension),
                nn.LayerNorm(dimension),
                nn.GELU(),
            )
            for group in EYE_GROUP_SLICES
        )
        self.eeg_token_embedding = nn.Parameter(torch.empty(1, 5, dimension))
        self.eye_token_embedding = nn.Parameter(torch.empty(1, 5, dimension))
        nn.init.normal_(self.eeg_token_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.eye_token_embedding, mean=0.0, std=0.02)

        if use_fuzzy:
            self.eeg_fuzzy = IntervalType2FuzzyReliability(
                fuzzy_rules,
                uncertainty_penalty,
            )
            self.eye_fuzzy = IntervalType2FuzzyReliability(
                fuzzy_rules,
                uncertainty_penalty,
            )
        else:
            self.eeg_fuzzy = None
            self.eye_fuzzy = None

        self.blocks = nn.ModuleList(
            MultimodalAttentionBlock(
                dimension,
                num_heads,
                ffn_dimension,
                dropout,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(dimension)
        self.classifier = nn.Sequential(
            nn.Linear(dimension * 2, dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dimension, num_classes),
        )

    @staticmethod
    def _statistics(groups: List[torch.Tensor]) -> torch.Tensor:
        statistics = []
        for group in groups:
            mean_absolute = group.abs().mean(dim=1)
            standard_deviation = group.std(dim=1, unbiased=False)
            energy = group.square().mean(dim=1)
            statistics.append(
                torch.stack(
                    (mean_absolute, standard_deviation, energy),
                    dim=-1,
                )
            )
        return torch.stack(statistics, dim=1)

    def _tokenize(
        self,
        eeg: torch.Tensor,
        eye: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        eeg_groups = [eeg[:, :, band] for band in range(5)]
        eye_groups = [eye[:, group] for group in EYE_GROUP_SLICES]
        eeg_tokens = self.eeg_projection(eeg.transpose(1, 2))
        eye_tokens = torch.stack(
            [
                projection(group)
                for projection, group in zip(self.eye_projections, eye_groups)
            ],
            dim=1,
        )
        return (
            eeg_tokens + self.eeg_token_embedding,
            eye_tokens + self.eye_token_embedding,
            eeg_groups,
            eye_groups,
        )

    def forward(
        self,
        eeg: torch.Tensor,
        eye: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if eeg.ndim != 3 or eeg.shape[1:] != (62, 5):
            raise ValueError(f"EEG input must have shape [B, 62, 5], got {eeg.shape}.")
        if eye.ndim != 2 or eye.shape[1] != 31:
            raise ValueError(f"Eye input must have shape [B, 31], got {eye.shape}.")

        eeg_tokens, eye_tokens, eeg_groups, eye_groups = self._tokenize(eeg, eye)
        diagnostics = {}
        if self.use_fuzzy:
            eeg_weights, eeg_info = self.eeg_fuzzy(
                self._statistics(eeg_groups)
            )
            eye_weights, eye_info = self.eye_fuzzy(
                self._statistics(eye_groups)
            )
            # Each modality has mean prior 1, so neither receives a size bias.
            reliability_prior = torch.cat(
                (5.0 * eeg_weights, 5.0 * eye_weights),
                dim=1,
            )
            diagnostics.update(
                {
                    "eeg_reliability": eeg_weights,
                    "eye_reliability": eye_weights,
                    "eeg_uncertainty": eeg_info["uncertainty"],
                    "eye_uncertainty": eye_info["uncertainty"],
                    "eeg_interval": torch.stack(
                        (
                            eeg_info["interval_lower"],
                            eeg_info["interval_upper"],
                        ),
                        dim=-1,
                    ),
                    "eye_interval": torch.stack(
                        (
                            eye_info["interval_lower"],
                            eye_info["interval_upper"],
                        ),
                        dim=-1,
                    ),
                }
            )
        else:
            eeg_weights = eye_weights = None
            reliability_prior = None

        tokens = torch.cat((eeg_tokens, eye_tokens), dim=1)
        attentions = []
        for block in self.blocks:
            tokens, attention = block(tokens, reliability_prior)
            attentions.append(attention)
        tokens = self.final_norm(tokens)

        if self.use_fuzzy:
            eeg_pooled = torch.sum(
                tokens[:, :5] * eeg_weights.unsqueeze(-1),
                dim=1,
            )
            eye_pooled = torch.sum(
                tokens[:, 5:] * eye_weights.unsqueeze(-1),
                dim=1,
            )
        else:
            eeg_pooled = tokens[:, :5].mean(dim=1)
            eye_pooled = tokens[:, 5:].mean(dim=1)
        logits = self.classifier(torch.cat((eeg_pooled, eye_pooled), dim=1))

        if not return_diagnostics:
            return logits
        diagnostics["attention"] = torch.stack(attentions, dim=0)
        return logits, diagnostics
