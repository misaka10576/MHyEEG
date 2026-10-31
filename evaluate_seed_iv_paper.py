"""Regenerate paper metrics from completed SEED-IV checkpoints.

This evaluator deliberately does not train or select checkpoints. It reloads the
checkpoint selected on the validation subject and evaluates it once on the held-
out test subject, using the same training-subject-only standardization.
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from data.seed_iv import (
    SUBJECT_IDS,
    SeedIVDataset,
    get_subject_split,
    load_seed_iv_features,
    standardize_from_training_subjects,
)
from main_seed_iv import (
    MODEL_NAMES,
    build_model,
    build_parser as build_training_parser,
    classification_metrics,
    save_json_atomic,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate complete paper metrics from SEED-IV checkpoints."
    )
    parser.add_argument(
        "--root",
        default="checkpoints/seed_iv_paper",
        help="Directory containing <Model>_seed<seed>/loso_summary.json.",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gpu_num", type=int, default=0)
    return parser


def _training_args(summary):
    args = build_training_parser().parse_args([])
    for key, value in summary.get("config", {}).items():
        setattr(args, key, value)
    args.model = summary["model"]
    return args


def _checkpoint_path(path_value, model_directory):
    path = Path(path_value)
    if path.is_file():
        return path
    candidate = model_directory / path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Checkpoint not found: {path_value}")


def _predict(model, data_loader, device):
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            eeg = inputs["eeg"].to(device, non_blocking=True)
            eye = inputs["eye"].to(device, non_blocking=True)
            logits = model(eeg=eeg, eye=eye)
            targets.append(labels.numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def _subject_level_summary(folds):
    result = {}
    for metric in ("accuracy_pct", "macro_f1"):
        values = np.asarray([fold[metric] for fold in folds], dtype=np.float64)
        result[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return result


def _validate_coverage(folds):
    if len(folds) != len(SUBJECT_IDS):
        raise ValueError(
            f"Expected {len(SUBJECT_IDS)} folds, found {len(folds)}."
        )
    fold_ids = [fold["fold"] for fold in folds]
    test_subjects = [fold["test_subject"] for fold in folds]
    if sorted(fold_ids) != list(range(len(SUBJECT_IDS))):
        raise ValueError(f"Fold coverage is invalid: {fold_ids}")
    if sorted(test_subjects) != list(SUBJECT_IDS):
        raise ValueError(f"Test-subject coverage is invalid: {test_subjects}")
    for fold in folds:
        split = get_subject_split(fold["fold"])
        if set(fold["train_subjects"]) != set(split["train"]):
            raise ValueError(f"Fold {fold['fold']} training subjects changed.")
        if fold["validation_subject"] != split["validation"][0]:
            raise ValueError(f"Fold {fold['fold']} validation subject changed.")
        if fold["test_subject"] != split["test"][0]:
            raise ValueError(f"Fold {fold['fold']} test subject changed.")


def evaluate_model(
    model_directory,
    raw_arrays,
    batch_size,
    num_workers,
    device,
):
    summary_path = model_directory / "loso_summary.json"
    with summary_path.open("r", encoding="utf-8") as input_file:
        summary = json.load(input_file)
    if not summary.get("complete"):
        raise ValueError(f"Formal run is incomplete: {summary_path}")
    _validate_coverage(summary["folds"])

    training_args = _training_args(summary)
    enriched_folds = []
    pooled_targets = []
    pooled_predictions = []
    for fold_result in sorted(summary["folds"], key=lambda item: item["fold"]):
        fold = fold_result["fold"]
        split = get_subject_split(fold)
        arrays = standardize_from_training_subjects(raw_arrays, split["train"])
        test_dataset = SeedIVDataset(arrays, split["test"])
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

        model = build_model(training_args).to(device)
        checkpoint_path = _checkpoint_path(
            fold_result["checkpoint_path"],
            model_directory,
        )
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        targets, predictions = _predict(model, test_loader, device)
        metrics = classification_metrics(targets, predictions)
        if not np.isfinite(metrics["accuracy_pct"]) or not np.isfinite(
            metrics["macro_f1"]
        ):
            raise ValueError(f"Non-finite metric detected in fold {fold}.")

        enriched = {
            **fold_result,
            "test_accuracy": metrics["accuracy_pct"],
            "test_f1": metrics["macro_f1"],
            **metrics,
        }
        save_json_atomic(model_directory / f"fold_{fold:02d}" / "metrics.json", enriched)
        enriched_folds.append(enriched)
        pooled_targets.append(targets)
        pooled_predictions.append(predictions)
        del arrays, test_dataset, test_loader, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pooled_metrics = classification_metrics(
        np.concatenate(pooled_targets),
        np.concatenate(pooled_predictions),
    )
    summary["folds"] = enriched_folds
    summary["paper_aggregate"] = {
        "subject_level": _subject_level_summary(enriched_folds),
        "pooled_test_samples": pooled_metrics,
    }
    summary["paper_metrics_complete"] = True
    save_json_atomic(summary_path, summary)
    return {
        "model": summary["model"],
        "summary_path": str(summary_path),
        "subject_level": summary["paper_aggregate"]["subject_level"],
        "pooled_test_samples": pooled_metrics,
    }


def main():
    args = build_parser().parse_args()
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    use_cuda = bool(args.cuda and torch.cuda.is_available())
    device = torch.device(f"cuda:{args.gpu_num}" if use_cuda else "cpu")
    root = Path(args.root)
    model_names = args.models or list(MODEL_NAMES)
    model_directories = [
        root / f"{model_name}_seed{args.seed}"
        for model_name in model_names
        if (root / f"{model_name}_seed{args.seed}" / "loso_summary.json").is_file()
    ]
    if not model_directories:
        raise FileNotFoundError(f"No completed model summaries found under {root}.")

    first_summary = json.loads(
        (model_directories[0] / "loso_summary.json").read_text(encoding="utf-8")
    )
    training_args = _training_args(first_summary)
    raw_arrays = load_seed_iv_features(
        training_args.dataset_root,
        training_args.eeg_feature_kind,
    )
    results = [
        evaluate_model(
            model_directory,
            raw_arrays,
            args.batch_size,
            args.num_workers,
            device,
        )
        for model_directory in model_directories
    ]
    payload = {
        "seed": args.seed,
        "models": results,
        "complete_models": len(results),
    }
    save_json_atomic(root / "paper_results.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
