"""Command-line runner for all 135 BERN SEED-IV experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from .data import load_run
from .training import TrainingConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="BERN/configs/default.json")
    parser.add_argument("--mode", choices=("eye", "eeg", "fusion", "all"), default="fusion")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--session", type=int, default=1)
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Run all 15 subjects x 3 sessions for the selected mode(s).",
    )
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--cache-dir")
    parser.add_argument("--eeg-feature-kind", choices=("de_LDS", "de_movingAve"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fusion-num-tokens", type=int)
    parser.add_argument("--fusion-token-dim", type=int)
    parser.add_argument("--fusion-num-heads", type=int)
    parser.add_argument("--fusion-variant", choices=("vector_token_v2",))
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use at most 2 epochs and 2 batches/epoch, stored below outputs/smoke.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume partial stages and skip runs with a complete metrics.json.",
    )
    return parser


def _load_values(args: argparse.Namespace) -> dict:
    path = Path(args.config)
    values = json.loads(path.read_text(encoding="utf-8"))
    cli_names = {
        "dataset_root", "output_dir", "cache_dir", "eeg_feature_kind", "seed",
        "batch_size", "learning_rate", "weight_decay", "max_epochs", "patience",
        "num_workers", "device", "pretrained", "deterministic",
        "fusion_num_tokens", "fusion_token_dim", "fusion_num_heads", "fusion_variant",
    }
    for name in cli_names:
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    if args.smoke_test:
        values["max_epochs"] = min(int(values["max_epochs"]), 2)
        values["patience"] = min(int(values["patience"]), 1)
        values["max_batches_per_epoch"] = 2
    values["smoke_test"] = args.smoke_test
    return values


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    values = _load_values(args)
    modes = ("eye", "eeg", "fusion") if args.mode == "all" else (args.mode,)
    pairs = (
        [(subject, session) for subject in range(1, 16) for session in range(1, 4)]
        if args.all_runs
        else [(args.subject, args.session)]
    )
    if args.subject not in range(1, 16) or args.session not in range(1, 4):
        raise SystemExit("--subject must be 1..15 and --session must be 1..3")
    for mode in modes:
        for subject, session in pairs:
            raw = load_run(
                values["dataset_root"],
                subject,
                session,
                values["eeg_feature_kind"],
            )
            config = TrainingConfig(
                mode=mode,
                subject=subject,
                session=session,
                **values,
            )
            metrics = run_experiment(raw, config, resume=args.resume)
            if mode == "fusion" and args.smoke_test:
                entropies = [
                    layer["normalized_entropy"]
                    for layer in metrics["attention_diagnostics"]
                ]
                if max(entropies) > 0.9995:
                    raise SystemExit(
                        "Fusion smoke gate failed: normalized attention entropy "
                        f"{entropies} exceeds 0.9995; formal training was not started."
                    )
            print(
                f"Completed {mode} subject={subject} session={session}: "
                f"accuracy={metrics['accuracy_pct']:.2f}% "
                f"macro-F1={metrics['macro_f1']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
