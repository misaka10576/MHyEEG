"""Generate paper-ready result figures from completed formal summaries."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIRECTORY = ROOT / "paper" / "figures"
MODEL_SUMMARIES = {
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
CLASS_NAMES = ("Neutral", "Sad", "Fear", "Happy")


def _read_completed_summaries():
    summaries = {}
    for name, path in MODEL_SUMMARIES.items():
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("complete"):
            summaries[name] = payload
    return summaries


def _aggregate(payload, metric):
    aggregate = payload["aggregate"]
    if metric in aggregate:
        return aggregate[metric]["mean"], aggregate[metric]["std"]
    legacy = {
        "accuracy_pct": "test_accuracy",
        "macro_f1": "test_f1",
    }[metric]
    return aggregate[legacy]["mean"], aggregate[legacy]["std"]


def _fold_metric(fold, metric):
    if metric in fold:
        return fold[metric]
    return fold[{"accuracy_pct": "test_accuracy", "macro_f1": "test_f1"}[metric]]


def _pooled_normalized_matrix(payload):
    if "paper_aggregate" in payload:
        return np.asarray(
            payload["paper_aggregate"]["pooled_test_samples"][
                "normalized_confusion_matrix"
            ],
            dtype=np.float64,
        )
    matrices = [
        np.asarray(fold["confusion_matrix"], dtype=np.int64)
        for fold in payload["folds"]
        if "confusion_matrix" in fold
    ]
    if len(matrices) != len(payload["folds"]):
        return None
    matrix = np.sum(matrices, axis=0)
    row_sum = matrix.sum(axis=1, keepdims=True)
    return np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sum != 0,
    )


def _save_figure(figure, stem):
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIRECTORY / f"{stem}.png"
    svg_path = FIGURE_DIRECTORY / f"{stem}.svg"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return [str(png_path.relative_to(ROOT)), str(svg_path.relative_to(ROOT))]


def plot_main_results(summaries):
    names = list(summaries)
    accuracy = [_aggregate(summaries[name], "accuracy_pct") for name in names]
    macro_f1 = [_aggregate(summaries[name], "macro_f1") for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    positions = np.arange(len(names))
    axes[0].bar(
        positions,
        [value[0] for value in accuracy],
        yerr=[value[1] for value in accuracy],
        capsize=3,
        color="#4C78A8",
    )
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    axes[1].bar(
        positions,
        [value[0] for value in macro_f1],
        yerr=[value[1] for value in macro_f1],
        capsize=3,
        color="#F58518",
    )
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.set_xticks(positions, names, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("SEED-IV cross-subject results (mean ± subject-level SD)")
    figure.tight_layout()
    return _save_figure(figure, "main_results")


def plot_subject_macro_f1(summaries):
    figure, axis = plt.subplots(figsize=(10, 5))
    subjects = np.arange(1, 16)
    for name, payload in summaries.items():
        folds = sorted(payload["folds"], key=lambda item: item["test_subject"])
        values = [_fold_metric(fold, "macro_f1") for fold in folds]
        axis.plot(subjects, values, marker="o", linewidth=1.5, label=name)
    axis.set_xlabel("Test subject")
    axis.set_ylabel("Macro-F1")
    axis.set_xticks(subjects)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    return _save_figure(figure, "subject_macro_f1")


def plot_confusion_matrices(summaries):
    available = [
        (name, matrix)
        for name, payload in summaries.items()
        if (matrix := _pooled_normalized_matrix(payload)) is not None
    ]
    if not available:
        return []
    columns = min(3, len(available))
    rows = int(np.ceil(len(available) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.1 * columns, 3.5 * rows),
        squeeze=False,
    )
    image = None
    for axis, (name, matrix) in zip(axes.flat, available):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axis.set_title(name)
        axis.set_xticks(range(4), CLASS_NAMES, rotation=30, ha="right")
        axis.set_yticks(range(4), CLASS_NAMES)
        for row in range(4):
            for column in range(4):
                color = "white" if matrix[row, column] >= 0.55 else "black"
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
    for axis in axes.flat[len(available) :]:
        axis.axis("off")
    figure.supxlabel("Predicted", y=0.02)
    figure.supylabel("True", x=0.02)
    figure.subplots_adjust(
        left=0.07,
        right=0.88,
        bottom=0.08,
        top=0.95,
        wspace=0.28,
        hspace=0.48,
    )
    colorbar_axis = figure.add_axes((0.91, 0.15, 0.018, 0.7))
    figure.colorbar(image, cax=colorbar_axis)
    return _save_figure(figure, "normalized_confusion_matrices")


def main():
    summaries = _read_completed_summaries()
    if not summaries:
        raise RuntimeError("No completed formal summaries were found.")
    generated = []
    generated.extend(plot_main_results(summaries))
    generated.extend(plot_subject_macro_f1(summaries))
    generated.extend(plot_confusion_matrices(summaries))
    manifest = {
        "completed_models": list(summaries),
        "generated_files": generated,
    }
    FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIRECTORY / "plot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
