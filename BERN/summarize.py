"""Validate completed runs and generate BERN reproduction summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import make_trial_split
from .training import classification_metrics


MODES = ("eye", "eeg", "fusion")
PAPER_ACCURACY_PCT = {"eye": 73.07, "eeg": 81.33, "fusion": 83.33}


def _assert_finite(value: Any, context: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{context} contains non-finite value {value}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{context}[{index}]")


def validate_run(path: Path, metrics: dict) -> None:
    mode = metrics["mode"]
    subject, session = int(metrics["subject"]), int(metrics["session"])
    if mode not in MODES or subject not in range(1, 16) or session not in range(1, 4):
        raise ValueError(f"Invalid run identity in {path}")
    expected = make_trial_split(session)
    expected_fields = {
        "selection_train_trials": list(expected.selection_train),
        "validation_trials": list(expected.validation),
        "final_train_trials": list(expected.final_train),
        "test_trials": list(expected.test),
        "selection_scaler_fit_trials": list(expected.selection_train),
        "final_scaler_fit_trials": list(expected.final_train),
    }
    for field, value in expected_fields.items():
        if metrics.get(field) != value:
            raise ValueError(f"{path}: {field} does not match the fixed protocol")
    if metrics.get("test_evaluations") != 1:
        raise ValueError(f"{path}: test must be evaluated exactly once")
    if metrics.get("sample_count", 0) < 1:
        raise ValueError(f"{path}: empty test predictions")
    _assert_finite(metrics, str(path))
    predictions_path = path.parent / "predictions.json"
    if not predictions_path.is_file():
        raise ValueError(f"{path}: missing predictions.json")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if len(predictions) != metrics["sample_count"]:
        raise ValueError(f"{path}: prediction count mismatch")
    if {int(row["trial"]) for row in predictions} - set(expected.test):
        raise ValueError(f"{path}: predictions contain non-test trials")
    _assert_finite(predictions, str(predictions_path))


def load_results(output_dir: str | Path, strict: bool = False) -> list[dict]:
    output = Path(output_dir)
    results: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for mode in MODES:
        for path in sorted((output / mode).glob("subject_*/session_*/metrics.json")):
            metrics = json.loads(path.read_text(encoding="utf-8"))
            validate_run(path, metrics)
            identity = (metrics["mode"], int(metrics["subject"]), int(metrics["session"]))
            if identity in seen:
                raise ValueError(f"Duplicate result: {identity}")
            seen.add(identity)
            results.append(metrics)
    if strict:
        expected = {
            (mode, subject, session)
            for mode in MODES
            for subject in range(1, 16)
            for session in range(1, 4)
        }
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        if missing or extra:
            raise ValueError(
                f"Expected exactly 135 runs; missing={len(missing)}, extra={len(extra)}. "
                f"First missing: {missing[:5]}"
            )
    return sorted(results, key=lambda item: (MODES.index(item["mode"]), item["subject"], item["session"]))


def _mean_std(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()) if array.size else 0.0,
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def build_summary(results: list[dict]) -> dict:
    modes: dict[str, dict] = {}
    for mode in MODES:
        rows = [row for row in results if row["mode"] == mode]
        by_subject: list[dict] = []
        for subject in range(1, 16):
            subject_rows = sorted(
                (row for row in rows if row["subject"] == subject),
                key=lambda row: row["session"],
            )
            if not subject_rows:
                continue
            by_subject.append(
                {
                    "subject": subject,
                    "sessions": {
                        str(row["session"]): row["accuracy_pct"] for row in subject_rows
                    },
                    "accuracy_pct": _mean_std(
                        [row["accuracy_pct"] for row in subject_rows]
                    )["mean"],
                    "macro_f1": _mean_std([row["macro_f1"] for row in subject_rows])["mean"],
                }
            )
        by_session: list[dict] = []
        for session in range(1, 4):
            session_rows = [row for row in rows if row["session"] == session]
            if session_rows:
                by_session.append(
                    {
                        "session": session,
                        "accuracy_pct": _mean_std(
                            [row["accuracy_pct"] for row in session_rows]
                        ),
                        "macro_f1": _mean_std([row["macro_f1"] for row in session_rows]),
                    }
                )
        subject_accuracies = [row["accuracy_pct"] for row in by_subject]
        pooled_matrix = np.sum(
            [np.asarray(row["confusion_matrix"], dtype=np.int64) for row in rows],
            axis=0,
        ) if rows else np.zeros((4, 4), dtype=np.int64)
        modes[mode] = {
            "completed_runs": len(rows),
            "run_level_accuracy_pct": _mean_std([row["accuracy_pct"] for row in rows]),
            "run_level_macro_f1": _mean_std([row["macro_f1"] for row in rows]),
            "paper_protocol_subject_accuracy_pct": _mean_std(subject_accuracies),
            "paper_accuracy_pct": PAPER_ACCURACY_PCT[mode],
            "absolute_gap_to_paper_pct_points": (
                abs(float(np.mean(subject_accuracies)) - PAPER_ACCURACY_PCT[mode])
                if subject_accuracies else None
            ),
            "per_subject": by_subject,
            "per_session": by_session,
            "pooled_metrics": _metrics_from_matrix(pooled_matrix),
        }
    return {
        "reimplementation": "paper-constrained, not author-code replication",
        "total_completed_runs": len(results),
        "modes": modes,
    }


def _metrics_from_matrix(matrix: np.ndarray) -> dict:
    targets: list[int] = []
    predictions: list[int] = []
    for target in range(4):
        for prediction in range(4):
            count = int(matrix[target, prediction])
            targets.extend([target] * count)
            predictions.extend([prediction] * count)
    return classification_metrics(targets, predictions)


def write_outputs(summary: dict, results: list[dict], report_dir: str | Path) -> None:
    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (destination / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "subject", "session", "accuracy_pct", "macro_f1", "best_epoch"))
        for row in results:
            writer.writerow(
                (
                    row["mode"], row["subject"], row["session"],
                    f"{row['accuracy_pct']:.8f}", f"{row['macro_f1']:.8f}", row["best_epoch"],
                )
            )
    with (destination / "subjects.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "subject", "session_1", "session_2", "session_3", "mean_accuracy_pct", "mean_macro_f1"))
        for mode in MODES:
            for row in summary["modes"][mode]["per_subject"]:
                sessions = row["sessions"]
                writer.writerow(
                    (
                        mode, row["subject"], sessions.get("1", ""), sessions.get("2", ""),
                        sessions.get("3", ""), f"{row['accuracy_pct']:.8f}",
                        f"{row['macro_f1']:.8f}",
                    )
                )
    lines = [
        "# BERN SEED-IV reproduction report",
        "",
        "> Paper-constrained reimplementation; this is not a claim of line-by-line author-code reproduction.",
        "",
        "| Mode | Completed | Subject mean ± SD (%) | Paper (%) | Absolute gap (pp) |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        value = summary["modes"][mode]
        statistic = value["paper_protocol_subject_accuracy_pct"]
        gap = value["absolute_gap_to_paper_pct_points"]
        gap_text = f"{gap:.2f}" if gap is not None else "n/a"
        lines.append(
            f"| {mode.capitalize()} | {value['completed_runs']}/45 | "
            f"{statistic['mean']:.2f} ± {statistic['std']:.2f} | "
            f"{value['paper_accuracy_pct']:.2f} | {gap_text} |"
        )
    lines.extend(
        [
            "",
            "The primary aggregate first averages each subject's three sessions, then "
            "reports mean and sample standard deviation across 15 subjects. Run-level, "
            "session-level, pooled-class and per-subject values are in `summary.json`.",
            "",
        ]
    )
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="BERN/outputs")
    parser.add_argument("--report-dir", default="BERN/reports")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require all 135 formal runs (45 for each of three modes).",
    )
    args = parser.parse_args(argv)
    results = load_results(args.output_dir, strict=args.strict)
    summary = build_summary(results)
    write_outputs(summary, results, args.report_dir)
    print(
        f"Validated {len(results)} run(s); wrote report files to {args.report_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
