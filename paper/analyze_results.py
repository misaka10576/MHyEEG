"""Validate formal summaries and run paired subject-level comparisons."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "paper" / "results"
SUMMARIES = {
    "EEG MLP": ROOT
    / "checkpoints/seed_iv_baselines/mlp/seed_0/loso_summary.json",
    "DGCNN": ROOT
    / "checkpoints/seed_iv_baselines/dgcnn/seed_0/loso_summary.json",
    "MAET-Concat": ROOT
    / "checkpoints/seed_iv_maet/concat/vanilla/seed_0/loso_summary.json",
    "Eye MLP": ROOT
    / "checkpoints/seed_iv_paper/SeedIVEye_seed0/loso_summary.json",
    "Concat MLP": ROOT
    / "checkpoints/seed_iv_paper/SeedIVConcat_seed0/loso_summary.json",
    "Attention": ROOT
    / "checkpoints/seed_iv_paper/SeedIVAttention_seed0/loso_summary.json",
    "IT2 Fuzzy": ROOT
    / "checkpoints/seed_iv_paper/SeedIVIT2FuzzyAttention_seed0/loso_summary.json",
}
COMPARISON_PAIRS = (
    ("IT2 Fuzzy", "Eye MLP", "reliability_analysis"),
    ("IT2 Fuzzy", "Concat MLP", "reliability_analysis"),
    ("IT2 Fuzzy", "Attention", "reliability_analysis"),
    ("IT2 Fuzzy", "MAET-Concat", "reliability_analysis"),
    ("Eye MLP", "Concat MLP", "negative_transfer_analysis"),
    ("Eye MLP", "Attention", "negative_transfer_analysis"),
    ("Eye MLP", "MAET-Concat", "negative_transfer_analysis"),
)


def _read_complete(path):
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete"):
        return None
    subjects = [fold["test_subject"] for fold in payload["folds"]]
    if sorted(subjects) != list(range(1, 16)):
        raise ValueError(f"Invalid test-subject coverage in {path}: {subjects}")
    return payload


def _metric_by_subject(payload, metric):
    field = metric
    legacy_field = {"accuracy_pct": "test_accuracy", "macro_f1": "test_f1"}[
        metric
    ]
    return np.asarray(
        [
            fold[field] if field in fold else fold[legacy_field]
            for fold in sorted(
                payload["folds"],
                key=lambda item: item["test_subject"],
            )
        ],
        dtype=np.float64,
    )


def _bootstrap_mean_ci(values, seed=0, repetitions=10000):
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        values,
        size=(repetitions, len(values)),
        replace=True,
    )
    means = samples.mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def _holm_adjust(rows):
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_value"])
    adjusted = np.empty(len(rows), dtype=np.float64)
    running_maximum = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * rows[index]["p_value"])
        running_maximum = max(running_maximum, candidate)
        adjusted[index] = running_maximum
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = float(value)


def paired_comparisons(summaries, metric):
    rows = []
    for model_a, model_b, family in COMPARISON_PAIRS:
        if model_a not in summaries or model_b not in summaries:
            continue
        values_a = _metric_by_subject(summaries[model_a], metric)
        values_b = _metric_by_subject(summaries[model_b], metric)
        differences = values_a - values_b
        result = wilcoxon(
            values_a,
            values_b,
            alternative="two-sided",
            method="auto",
        )
        lower, upper = _bootstrap_mean_ci(differences)
        rows.append(
            {
                "metric": metric,
                "family": family,
                "model_a": model_a,
                "model_b": model_b,
                "mean_difference_a_minus_b": float(differences.mean()),
                "difference_95pct_bootstrap_ci": [lower, upper],
                "wilcoxon_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    _holm_adjust(rows)
    return rows


def _markdown_summary(summaries, comparisons):
    lines = [
        "# SEED-IV正式结果统计",
        "",
        "主结果为15名测试被试的均值±样本标准差。",
        "",
        "| 模型 | Accuracy/% | Macro-F1 |",
        "|---|---:|---:|",
    ]
    for name, payload in summaries.items():
        accuracy = _metric_by_subject(payload, "accuracy_pct")
        macro_f1 = _metric_by_subject(payload, "macro_f1")
        lines.append(
            f"| {name} | {accuracy.mean():.2f} ± {accuracy.std(ddof=1):.2f} "
            f"| {macro_f1.mean():.4f} ± {macro_f1.std(ddof=1):.4f} |"
        )
    if comparisons:
        lines.extend(
            [
                "",
                "## 逐被试配对比较",
                "",
                "| 指标 | 比较 | A−B均值差 | 95% bootstrap CI | Wilcoxon p "
                "| Holm校正p |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in comparisons:
            lower, upper = row["difference_95pct_bootstrap_ci"]
            lines.append(
                f"| {row['metric']} "
                f"| {row['model_a']} − {row['model_b']} "
                f"| {row['mean_difference_a_minus_b']:.4f} "
                f"| [{lower:.4f}, {upper:.4f}] "
                f"| {row['p_value']:.4g} "
                f"| {row['holm_adjusted_p']:.4g} |"
            )
    else:
        lines.extend(
            [
                "",
                "> 预设比较所需模型尚未完成，暂不执行显著性检验。",
            ]
        )
    return "\n".join(lines) + "\n"


def main():
    summaries = {
        name: payload
        for name, path in SUMMARIES.items()
        if (payload := _read_complete(path)) is not None
    }
    comparisons = []
    for metric in ("accuracy_pct", "macro_f1"):
        comparisons.extend(paired_comparisons(summaries, metric))
    payload = {
        "complete_models": list(summaries),
        "comparisons": comparisons,
        "note": (
            "Wilcoxon tests are paired by held-out subject. The comparison "
            "pairs are fixed in COMPARISON_PAIRS; Holm correction is applied "
            "jointly to all listed comparisons within each metric."
        ),
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIRECTORY / "statistical_tests.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIRECTORY / "statistical_tests.md").write_text(
        _markdown_summary(summaries, comparisons),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
