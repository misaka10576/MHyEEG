"""Validate and compare original and vector-token BERN Fusion experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .summarize import PAPER_ACCURACY_PCT, validate_run


IDENTITIES = {(subject, session) for subject in range(1, 16) for session in range(1, 4)}
ARCHITECTURE = {
    "fusion_variant": "vector_token_v2",
    "fusion_num_tokens": 8,
    "fusion_token_dim": 32,
    "fusion_num_heads": 4,
}
FIXED_UNIMODAL_ACCURACY_PCT = {"eye": 47.69199652413826, "eeg": 36.10104334579605}


def _load_fusion(root: Path, vector: bool) -> dict[tuple[int, int], dict]:
    rows: dict[tuple[int, int], dict] = {}
    for path in sorted((root / "fusion").glob("subject_*/session_*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        validate_run(path, metrics)
        identity = (int(metrics["subject"]), int(metrics["session"]))
        if identity in rows:
            raise ValueError(f"Duplicate Fusion result {identity} below {root}")
        if vector:
            for key, expected in ARCHITECTURE.items():
                if metrics.get(key) != expected:
                    raise ValueError(f"{path}: expected {key}={expected!r}")
            diagnostics = metrics.get("attention_diagnostics")
            if not isinstance(diagnostics, list) or len(diagnostics) != 2:
                raise ValueError(f"{path}: expected two attention diagnostic layers")
            for index, layer in enumerate(diagnostics, start=1):
                if layer.get("layer") != index:
                    raise ValueError(f"{path}: invalid diagnostic layer numbering")
                for field in (
                    "normalized_entropy",
                    "relative_uniform_deviation",
                    "attention_variance",
                ):
                    if not math.isfinite(float(layer[field])):
                        raise ValueError(f"{path}: non-finite {field}")
            if not (path.parent / "test_evaluation.json").is_file():
                raise ValueError(f"{path}: missing atomic test_evaluation.json")
        rows[identity] = metrics
    return rows


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def build_comparison(original: dict, vector: dict) -> tuple[dict, list[dict]]:
    missing_original = sorted(IDENTITIES - set(original))
    missing_vector = sorted(IDENTITIES - set(vector))
    extra = sorted((set(original) | set(vector)) - IDENTITIES)
    if missing_original or missing_vector or extra:
        raise ValueError(
            "Expected exactly 45 paired Fusion runs; "
            f"missing_original={missing_original[:5]}, "
            f"missing_vector={missing_vector[:5]}, extra={extra[:5]}"
        )
    paired: list[dict] = []
    for subject, session in sorted(IDENTITIES):
        old, new = original[(subject, session)], vector[(subject, session)]
        paired.append(
            {
                "subject": subject,
                "session": session,
                "original_accuracy_pct": old["accuracy_pct"],
                "vector_accuracy_pct": new["accuracy_pct"],
                "accuracy_delta_pct_points": new["accuracy_pct"] - old["accuracy_pct"],
                "original_macro_f1": old["macro_f1"],
                "vector_macro_f1": new["macro_f1"],
                "macro_f1_delta": new["macro_f1"] - old["macro_f1"],
            }
        )
    tolerance = 1e-12
    win_tie_loss = {
        "wins": sum(row["accuracy_delta_pct_points"] > tolerance for row in paired),
        "ties": sum(abs(row["accuracy_delta_pct_points"]) <= tolerance for row in paired),
        "losses": sum(row["accuracy_delta_pct_points"] < -tolerance for row in paired),
    }
    per_subject = []
    for subject in range(1, 16):
        rows = [row for row in paired if row["subject"] == subject]
        per_subject.append(
            {
                "subject": subject,
                "sessions": {
                    str(row["session"]): {
                        "original_accuracy_pct": row["original_accuracy_pct"],
                        "vector_accuracy_pct": row["vector_accuracy_pct"],
                        "delta_pct_points": row["accuracy_delta_pct_points"],
                    }
                    for row in rows
                },
                "original_accuracy_pct": float(np.mean([row["original_accuracy_pct"] for row in rows])),
                "vector_accuracy_pct": float(np.mean([row["vector_accuracy_pct"] for row in rows])),
                "accuracy_delta_pct_points": float(np.mean([row["accuracy_delta_pct_points"] for row in rows])),
                "original_macro_f1": float(np.mean([row["original_macro_f1"] for row in rows])),
                "vector_macro_f1": float(np.mean([row["vector_macro_f1"] for row in rows])),
            }
        )
    per_session = []
    for session in range(1, 4):
        rows = [row for row in paired if row["session"] == session]
        per_session.append(
            {
                "session": session,
                "original_accuracy_pct": _stats([row["original_accuracy_pct"] for row in rows]),
                "vector_accuracy_pct": _stats([row["vector_accuracy_pct"] for row in rows]),
                "accuracy_delta_pct_points": _stats([row["accuracy_delta_pct_points"] for row in rows]),
                "original_macro_f1": _stats([row["original_macro_f1"] for row in rows]),
                "vector_macro_f1": _stats([row["vector_macro_f1"] for row in rows]),
            }
        )
    diagnostic_summary = []
    for layer_index in range(2):
        layer_entropies = [
            row["attention_diagnostics"][layer_index]["normalized_entropy"]
            for row in vector.values()
        ]
        diagnostic_summary.append(
            {
                "layer": layer_index + 1,
                "runs_above_smoke_entropy_threshold": sum(
                    value > 0.9995 for value in layer_entropies
                ),
                **{
                    field: _stats(
                        [
                            row["attention_diagnostics"][layer_index][field]
                            for row in vector.values()
                        ]
                    )
                    for field in (
                        "normalized_entropy",
                        "relative_uniform_deviation",
                        "attention_variance",
                    )
                },
            }
        )
    old_subject = [row["original_accuracy_pct"] for row in per_subject]
    new_subject = [row["vector_accuracy_pct"] for row in per_subject]
    summary = {
        "fusion_variant": ARCHITECTURE,
        "paired_runs": len(paired),
        "fixed_unimodal_accuracy_pct": FIXED_UNIMODAL_ACCURACY_PCT,
        "original_fusion": {
            "run_accuracy_pct": _stats([row["original_accuracy_pct"] for row in paired]),
            "run_macro_f1": _stats([row["original_macro_f1"] for row in paired]),
            "subject_accuracy_pct": _stats(old_subject),
        },
        "vector_fusion": {
            "run_accuracy_pct": _stats([row["vector_accuracy_pct"] for row in paired]),
            "run_macro_f1": _stats([row["vector_macro_f1"] for row in paired]),
            "subject_accuracy_pct": _stats(new_subject),
            "paper_accuracy_pct": PAPER_ACCURACY_PCT["fusion"],
            "absolute_gap_to_paper_pct_points": abs(
                float(np.mean(new_subject)) - PAPER_ACCURACY_PCT["fusion"]
            ),
        },
        "paired_accuracy_delta_pct_points": _stats(
            [row["accuracy_delta_pct_points"] for row in paired]
        ),
        "win_tie_loss": win_tie_loss,
        "per_subject": per_subject,
        "per_session": per_session,
        "attention_diagnostics": diagnostic_summary,
    }
    return summary, paired


def write_outputs(summary: dict, paired: list[dict], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (destination / "paired_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    vector = summary["vector_fusion"]
    original = summary["original_fusion"]
    delta = summary["paired_accuracy_delta_pct_points"]
    wtl = summary["win_tie_loss"]
    eye = summary["fixed_unimodal_accuracy_pct"]["eye"]
    eeg = summary["fixed_unimodal_accuracy_pct"]["eeg"]
    diagnostics = summary["attention_diagnostics"]
    lines = [
        "# BERN vector-token Fusion comparison",
        "",
        "| Result | Original Fusion | Vector-token Fusion |",
        "|---|---:|---:|",
        f"| Subject accuracy (%) | {original['subject_accuracy_pct']['mean']:.2f} | {vector['subject_accuracy_pct']['mean']:.2f} |",
        f"| Run accuracy (%) | {original['run_accuracy_pct']['mean']:.2f} | {vector['run_accuracy_pct']['mean']:.2f} |",
        f"| Run Macro-F1 | {original['run_macro_f1']['mean']:.4f} | {vector['run_macro_f1']['mean']:.4f} |",
        "",
        f"Across 45 paired runs, the mean accuracy delta is {delta['mean']:+.2f} "
        f"percentage points (win/tie/loss: {wtl['wins']}/{wtl['ties']}/{wtl['losses']}).",
        f"The vector-token subject mean is {vector['absolute_gap_to_paper_pct_points']:.2f} "
        f"points from the paper's {vector['paper_accuracy_pct']:.2f}%.",
        "",
        f"The fixed original unimodal comparisons are Eye {eye:.2f}% and EEG "
        f"{eeg:.2f}%. Mean normalized attention entropy is "
        f"{diagnostics[0]['normalized_entropy']['mean']:.4f} in layer 1 and "
        f"{diagnostics[1]['normalized_entropy']['mean']:.4f} in layer 2. "
        f"{diagnostics[0]['runs_above_smoke_entropy_threshold']} and "
        f"{diagnostics[1]['runs_above_smoke_entropy_threshold']} formal runs, "
        "respectively, exceed the smoke threshold; the pre-registered stop rule "
        "was applied only to the S1/S1 smoke run.",
        "",
        "Per-subject, per-session, paired-run, and attention diagnostic values are "
        "available in `comparison.json` and `paired_runs.csv`.",
        "",
    ]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-output-dir", default="BERN/outputs")
    parser.add_argument(
        "--vector-output-dir", default="BERN/outputs/vector_attention"
    )
    parser.add_argument("--report-dir", default="BERN/reports/vector_attention")
    args = parser.parse_args(argv)
    original = _load_fusion(Path(args.original_output_dir), vector=False)
    vector = _load_fusion(Path(args.vector_output_dir), vector=True)
    summary, paired = build_comparison(original, vector)
    write_outputs(summary, paired, Path(args.report_dir))
    print(
        f"Validated and compared {len(paired)} paired Fusion runs; "
        f"wrote {args.report_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
