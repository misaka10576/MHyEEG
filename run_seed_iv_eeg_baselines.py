"""Run leakage-safe SEED-IV EEG baselines in the MHyEEG framework."""

import argparse
import copy
import json
import os
import random
from multiprocessing import cpu_count
from pathlib import Path

import joblib
import numpy as np
import torch
import wandb
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC

from data.seed_iv import (
    CLASS_NAMES,
    SUBJECT_IDS,
    build_seed_iv_loaders,
    get_subject_split,
    load_seed_iv_features,
    standardize_from_training_subjects,
)
from models.seed_iv_baselines import SeedIVBaseline
from models.seed_iv_eeg_baselines import SeedIVDGCNN
from training import Trainer


MODEL_NAMES = ("linear_svm", "mlp", "dgcnn")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EEG-only baselines on leakage-safe SEED-IV folds."
    )
    parser.add_argument("--config", type=str)
    parser.add_argument("--dataset_root", type=str, default="SEED_IV")
    parser.add_argument("--eeg_feature_kind", type=str, default="de_LDS")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--all_folds",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--graph_channels", type=int, default=32)
    parser.add_argument("--chebyshev_order", type=int, default=2)
    parser.add_argument("--classifier_dim", type=int, default=64)
    parser.add_argument("--svm_c_values", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--svm_max_iter", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_mom", type=float, default=0.85)
    parser.add_argument("--max_mom", type=float, default=0.95)
    parser.add_argument("--div_factor", type=float, default=10.0)
    parser.add_argument("--final_div_factor", type=float, default=100.0)
    parser.add_argument("--pct_start", type=float, default=0.1)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--amp_dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cudnn_benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output_dir", type=str, default="checkpoints/seed_iv_baselines")
    parser.add_argument("--wandb_project", type=str, default="MHyEEG-SEED-IV-Baselines")
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    parser.add_argument("--wandb_log_interval", type=int, default=20)
    parser.add_argument(
        "--smoke_test",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--smoke_epochs", type=int, default=2)
    parser.add_argument("--smoke_train_samples", type=int, default=2048)
    parser.add_argument("--smoke_eval_samples", type=int, default=512)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    config_args, _ = parser.parse_known_args(argv)
    if config_args.config:
        with open(config_args.config, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    if args.num_workers == "max":
        args.num_workers = cpu_count()
    else:
        args.num_workers = int(args.num_workers)
    if args.num_workers < 0:
        parser.error("num_workers must be non-negative or 'max'.")
    if not args.svm_c_values or any(value <= 0 for value in args.svm_c_values):
        parser.error("svm_c_values must contain positive values.")
    if args.smoke_test and args.all_folds:
        parser.error("--smoke_test and --all_folds cannot be used together.")
    return args


def set_reproducibility(args: argparse.Namespace) -> None:
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.use_deterministic_algorithms(args.deterministic, warn_only=True)


def build_neural_model(name: str, args: argparse.Namespace) -> torch.nn.Module:
    if name == "mlp":
        return SeedIVBaseline(
            mode="eeg",
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout_rate,
            num_classes=len(CLASS_NAMES),
        )
    if name == "dgcnn":
        return SeedIVDGCNN(
            graph_channels=args.graph_channels,
            chebyshev_order=args.chebyshev_order,
            classifier_dim=args.classifier_dim,
            dropout=args.dropout_rate,
            num_classes=len(CLASS_NAMES),
        )
    raise ValueError(f"{name!r} is not a neural baseline.")


def classification_metrics(targets, predictions) -> dict:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    labels = np.arange(len(CLASS_NAMES))
    matrix = confusion_matrix(targets, predictions, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_totals != 0,
    )
    precision, recall, class_f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    return {
        "accuracy_pct": float(accuracy_score(targets, predictions) * 100.0),
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, class_name in enumerate(CLASS_NAMES)
        },
    }


def save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def _stratified_indices(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or limit >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    selected = []
    class_count = len(CLASS_NAMES)
    base_count, remainder = divmod(limit, class_count)
    for class_index in range(class_count):
        candidates = np.flatnonzero(labels == class_index)
        count = min(len(candidates), base_count + int(class_index < remainder))
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def _limited_loader(
    loader: torch.utils.data.DataLoader,
    limit: int,
    seed: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    dataset = loader.dataset
    labels = dataset.arrays.labels[dataset.indices]
    relative_indices = _stratified_indices(labels, limit, seed)
    subset = torch.utils.data.Subset(dataset, relative_indices.tolist())
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=False,
    )


def _predict_neural(
    trainer: Trainer,
    data_loader: torch.utils.data.DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    trainer.net.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            if trainer.use_cuda:
                device = f"cuda:{trainer.gpu_num}"
                inputs = trainer._move_to_device(inputs, device)
            logits = trainer._forward_inputs(inputs)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def _fold_metadata(fold: int) -> dict:
    split = get_subject_split(fold)
    return {
        "fold": fold,
        "train_subjects": list(split["train"]),
        "validation_subject": split["validation"][0],
        "test_subject": split["test"][0],
    }


def run_neural_fold(
    name: str,
    args: argparse.Namespace,
    fold: int,
    fold_directory: Path,
) -> dict:
    set_reproducibility(args)
    use_cuda = bool(args.cuda and torch.cuda.is_available())
    if args.cuda and not use_cuda:
        print("CUDA requested but unavailable; using CPU.")

    train_loader, validation_loader, test_loader, class_weights = build_seed_iv_loaders(
        dataset_root=args.dataset_root,
        fold=fold,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        eeg_feature_kind=args.eeg_feature_kind,
        seed=args.seed,
    )
    epochs = args.epochs
    patience = args.patience
    if args.smoke_test:
        epochs = args.smoke_epochs
        patience = min(patience, epochs)
        train_loader = _limited_loader(
            train_loader,
            args.smoke_train_samples,
            args.seed,
            shuffle=True,
        )
        validation_loader = _limited_loader(
            validation_loader,
            args.smoke_eval_samples,
            args.seed,
            shuffle=False,
        )
        test_loader = _limited_loader(
            test_loader,
            args.smoke_eval_samples,
            args.seed,
            shuffle=False,
        )

    network = build_neural_model(name, args)
    parameter_count = sum(
        parameter.numel()
        for parameter in network.parameters()
        if parameter.requires_grad
    )
    fold_directory.mkdir(parents=True, exist_ok=True)
    run_name = (
        f"{name}-seed{args.seed}-fold{fold:02d}"
        + ("-smoke" if args.smoke_test else "")
    )
    run = wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        name=run_name,
        config=vars(args),
    )
    # Disabled W&B runs receive a random "dummy-*" name unless it is reset.
    # Trainer uses this name for checkpoints, so keep local artifacts stable.
    run.name = run_name
    try:
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=args.max_lr / args.div_factor,
            weight_decay=args.weight_decay,
            eps=1e-7,
        )
        trainer = Trainer(
            network,
            optimizer,
            epochs=epochs,
            use_cuda=use_cuda,
            gpu_num=args.gpu_num,
            checkpoint_folder=str(fold_directory),
            max_lr=args.max_lr,
            min_mom=args.min_mom,
            max_mom=args.max_mom,
            num_classes=len(CLASS_NAMES),
            sample_weights=class_weights,
            es_mode="max",
            patience=patience,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            allow_tf32=args.allow_tf32,
            wandb_log_interval=args.wandb_log_interval,
            label_smoothing=args.label_smoothing,
            max_grad_norm=args.max_grad_norm,
        )
        training_result = trainer.train(
            train_loader,
            validation_loader,
            max_lr=args.max_lr,
            div_factor=args.div_factor,
            final_div_factor=args.final_div_factor,
            pct_start=args.pct_start,
        )
        evaluation = trainer.evaluate(
            test_loader,
            checkpoint_path=training_result["checkpoint_path"],
            split="test",
        )
        targets, predictions = _predict_neural(trainer, test_loader)
    finally:
        wandb.finish()

    result = {
        "model": name,
        **_fold_metadata(fold),
        "smoke_test": bool(args.smoke_test),
        "parameter_count": parameter_count,
        "checkpoint_path": training_result["checkpoint_path"],
        "best_validation_macro_f1": float(training_result["best_val_f1"]),
        "best_validation_accuracy_pct": float(training_result["best_val_acc"]),
        "test_loss": float(evaluation["loss"]),
        **classification_metrics(targets, predictions),
    }
    save_json_atomic(fold_directory / "metrics.json", result)
    return result


def _split_numpy_arrays(args: argparse.Namespace, fold: int):
    split = get_subject_split(fold)
    arrays = load_seed_iv_features(
        args.dataset_root,
        eeg_feature_kind=args.eeg_feature_kind,
    )
    arrays = standardize_from_training_subjects(arrays, split["train"])

    def select(subjects):
        mask = np.isin(arrays.subjects, subjects)
        return arrays.eeg[mask].reshape(mask.sum(), -1), arrays.labels[mask]

    return select(split["train"]), select(split["validation"]), select(split["test"])


def run_svm_fold(
    args: argparse.Namespace,
    fold: int,
    fold_directory: Path,
) -> dict:
    set_reproducibility(args)
    (train_x, train_y), (validation_x, validation_y), (test_x, test_y) = (
        _split_numpy_arrays(args, fold)
    )
    if args.smoke_test:
        train_indices = _stratified_indices(
            train_y,
            args.smoke_train_samples,
            args.seed,
        )
        validation_indices = _stratified_indices(
            validation_y,
            args.smoke_eval_samples,
            args.seed,
        )
        test_indices = _stratified_indices(
            test_y,
            args.smoke_eval_samples,
            args.seed,
        )
        train_x, train_y = train_x[train_indices], train_y[train_indices]
        validation_x, validation_y = (
            validation_x[validation_indices],
            validation_y[validation_indices],
        )
        test_x, test_y = test_x[test_indices], test_y[test_indices]

    candidates = []
    best_model = None
    best_key = None
    for c_value in sorted(args.svm_c_values):
        model = LinearSVC(
            C=c_value,
            class_weight="balanced",
            dual="auto",
            max_iter=args.svm_max_iter,
            random_state=args.seed,
        )
        model.fit(train_x, train_y)
        validation_predictions = model.predict(validation_x)
        validation_f1 = f1_score(
            validation_y,
            validation_predictions,
            average="macro",
            zero_division=0,
        )
        validation_accuracy = accuracy_score(
            validation_y,
            validation_predictions,
        )
        candidates.append(
            {
                "C": float(c_value),
                "validation_macro_f1": float(validation_f1),
                "validation_accuracy_pct": float(validation_accuracy * 100.0),
            }
        )
        selection_key = (validation_f1, -c_value)
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_model = model

    fold_directory.mkdir(parents=True, exist_ok=True)
    model_path = fold_directory / "model.joblib"
    joblib.dump(best_model, model_path)
    test_predictions = best_model.predict(test_x)
    result = {
        "model": "linear_svm",
        **_fold_metadata(fold),
        "smoke_test": bool(args.smoke_test),
        "parameter_count": None,
        "checkpoint_path": str(model_path),
        "selected_C": float(-best_key[1]),
        "validation_candidates": candidates,
        "best_validation_macro_f1": float(best_key[0]),
        **classification_metrics(test_y, test_predictions),
    }
    save_json_atomic(fold_directory / "metrics.json", result)
    return result


def summarize_results(
    name: str,
    args: argparse.Namespace,
    fold_results: list[dict],
) -> dict:
    def aggregate(key):
        values = np.asarray([result[key] for result in fold_results], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }

    completed_test_subjects = sorted(
        result["test_subject"] for result in fold_results
    )
    return {
        "model": name,
        "seed": args.seed,
        "smoke_test": bool(args.smoke_test),
        "completed_folds": len(fold_results),
        "expected_folds": len(SUBJECT_IDS),
        "complete": (
            not args.smoke_test
            and completed_test_subjects == list(SUBJECT_IDS)
        ),
        "config": vars(args),
        "aggregate": {
            "accuracy_pct": aggregate("accuracy_pct"),
            "macro_f1": aggregate("macro_f1"),
        },
        "folds": fold_results,
    }


def run_model(name: str, args: argparse.Namespace) -> dict:
    folds = range(len(SUBJECT_IDS)) if args.all_folds else (args.fold,)
    model_directory = Path(args.output_dir) / name / f"seed_{args.seed}"
    if args.smoke_test:
        model_directory = model_directory / "smoke"
    fold_results = []
    for fold in folds:
        print(f"\n========== {name} fold {fold:02d} ==========")
        fold_directory = model_directory / f"fold_{fold:02d}"
        if name == "linear_svm":
            result = run_svm_fold(args, fold, fold_directory)
        else:
            result = run_neural_fold(name, args, fold, fold_directory)
        fold_results.append(result)
        summary = summarize_results(name, args, fold_results)
        save_json_atomic(model_directory / "loso_summary.json", summary)
        print(
            f"{name} fold {fold:02d}: "
            f"Accuracy={result['accuracy_pct']:.2f}%, "
            f"Macro-F1={result['macro_f1']:.4f}"
        )
    return summary


def main(args: argparse.Namespace) -> dict:
    if not 0 <= args.fold < len(SUBJECT_IDS):
        raise ValueError(f"fold must be between 0 and {len(SUBJECT_IDS) - 1}.")
    results = {}
    for name in args.models:
        model_args = copy.deepcopy(args)
        results[name] = run_model(name, model_args)
    return results


if __name__ == "__main__":
    main(parse_args())
