"""Train MAET on leakage-safe, subject-independent SEED-IV folds."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from data.seed_iv import (
    CLASS_NAMES,
    SUBJECT_IDS,
    SeedIVDataset,
    build_seed_iv_loaders,
    get_subject_split,
)
from models.seed_iv_maet import SeedIVMAET


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SEED-IV adaptation of MAET."
    )
    parser.add_argument("--config", type=str)
    parser.add_argument("--dataset_root", type=str, default="SEED_IV")
    parser.add_argument("--eeg_feature_kind", type=str, default="de_LDS")
    parser.add_argument("--mode", choices=SeedIVMAET.valid_modes, default="concat")
    parser.add_argument(
        "--domain_generalization",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--all_folds",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--attn_drop_rate", type=float, default=0.0)
    parser.add_argument("--drop_path_rate", type=float, default=0.0)
    parser.add_argument("--mixffn_start_layer_index", type=int, default=2)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", default=4)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--emotion_label_smoothing", type=float, default=0.05)
    parser.add_argument("--domain_loss_weight", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--pct_start", type=float, default=0.1)
    parser.add_argument("--div_factor", type=float, default=10.0)
    parser.add_argument("--final_div_factor", type=float, default=100.0)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--output_dir", type=str, default="checkpoints/seed_iv_maet")
    parser.add_argument(
        "--smoke_test",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--smoke_epochs", type=int, default=2)
    parser.add_argument("--smoke_train_samples", type=int, default=512)
    parser.add_argument("--smoke_eval_samples", type=int, default=256)
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    config_args, _ = parser.parse_known_args(argv)
    if config_args.config:
        with open(config_args.config, "r", encoding="utf-8") as config_file:
            parser.set_defaults(**(yaml.safe_load(config_file) or {}))
    args = parser.parse_args(argv)
    args.num_workers = (
        cpu_count() if args.num_workers == "max" else int(args.num_workers)
    )
    if args.num_workers < 0:
        parser.error("num_workers must be non-negative or 'max'.")
    if args.smoke_test and args.all_folds:
        parser.error("--smoke_test and --all_folds cannot be used together.")
    if args.epochs <= 0 or args.patience <= 0:
        parser.error("epochs and patience must be positive.")
    if args.train_batch_size <= 0 or args.test_batch_size <= 0:
        parser.error("batch sizes must be positive.")
    if not 0.0 <= args.emotion_label_smoothing < 1.0:
        parser.error("emotion_label_smoothing must be in [0, 1).")
    if args.domain_loss_weight < 0:
        parser.error("domain_loss_weight must be non-negative.")
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
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(args.deterministic, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)


def gradient_reversal_alpha(epoch: int, epochs: int) -> float:
    """Return the original MAET epoch-wise gradient reversal coefficient."""
    if epochs <= 0 or not 0 <= epoch < epochs:
        raise ValueError("epoch must be in [0, epochs).")
    return float(2.0 / (1.0 + math.exp(-10.0 * epoch / epochs)) - 1.0)


def domain_label_smoothing(epoch: int, epochs: int, num_domains: int) -> float:
    """Return the original MAET epoch-wise domain label smoothing value."""
    if epochs <= 0 or not 0 <= epoch < epochs:
        raise ValueError("epoch must be in [0, epochs).")
    if num_domains <= 1:
        raise ValueError("num_domains must be greater than one.")
    return float((num_domains - 1) / num_domains * epoch / epochs)


def build_domain_mapping(fold: int) -> dict[int, int]:
    """Map exactly the 13 training subjects in a fold to labels 0 through 12."""
    split = get_subject_split(fold)
    return {
        subject: domain_index
        for domain_index, subject in enumerate(split["train"])
    }


def _dataset_subject(dataset, index: int) -> int:
    if isinstance(dataset, torch.utils.data.Subset):
        return _dataset_subject(dataset.dataset, int(dataset.indices[index]))
    if isinstance(dataset, SeedIVDataset):
        sample_index = dataset.indices[index]
        return int(dataset.arrays.subjects[sample_index])
    raise TypeError(
        "Domain labels require SeedIVDataset or a torch Subset of it."
    )


class DomainLabeledDataset(torch.utils.data.Dataset):
    """Add a training-only, fold-local subject domain label."""

    def __init__(self, dataset, domain_mapping: dict[int, int]) -> None:
        self.dataset = dataset
        self.domain_mapping = dict(domain_mapping)
        observed_subjects = {
            _dataset_subject(dataset, index) for index in range(len(dataset))
        }
        unknown = observed_subjects.difference(self.domain_mapping)
        if unknown:
            raise ValueError(
                f"Dataset contains subjects absent from domain mapping: {sorted(unknown)}."
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        inputs, emotion_label = self.dataset[index]
        subject = _dataset_subject(self.dataset, index)
        return inputs, emotion_label, self.domain_mapping[subject]


def _stratified_indices(labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or limit >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    selected = []
    per_class, remainder = divmod(limit, len(CLASS_NAMES))
    for class_index in range(len(CLASS_NAMES)):
        candidates = np.flatnonzero(labels == class_index)
        count = min(
            len(candidates),
            per_class + int(class_index < remainder),
        )
        selected.extend(
            rng.choice(candidates, size=count, replace=False).tolist()
        )
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def _dataset_labels(dataset) -> np.ndarray:
    if isinstance(dataset, torch.utils.data.Subset):
        parent_labels = _dataset_labels(dataset.dataset)
        return parent_labels[np.asarray(dataset.indices, dtype=np.int64)]
    if isinstance(dataset, SeedIVDataset):
        return dataset.arrays.labels[dataset.indices]
    raise TypeError("Smoke sampling requires a SeedIVDataset or Subset.")


def limited_loader(
    loader: torch.utils.data.DataLoader,
    limit: int,
    seed: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    labels = _dataset_labels(loader.dataset)
    indices = _stratified_indices(labels, limit, seed)
    subset = torch.utils.data.Subset(loader.dataset, indices.tolist())
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=False,
    )


def add_domain_labels(
    loader: torch.utils.data.DataLoader,
    mapping: dict[int, int],
    seed: int,
) -> torch.utils.data.DataLoader:
    dataset = DomainLabeledDataset(loader.dataset, mapping)
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=loader.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0 if loader.num_workers == 0 else loader.num_workers,
        pin_memory=loader.pin_memory,
        persistent_workers=loader.num_workers > 0,
    )


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


class MAETTrainer:
    """Small dedicated trainer for emotion and optional adversarial domain loss."""

    def __init__(
        self,
        model: SeedIVMAET,
        optimizer: torch.optim.Optimizer,
        class_weights,
        epochs: int,
        device: torch.device,
        domain_loss_weight: float = 1.0,
        emotion_label_smoothing: float = 0.05,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.domain_loss_weight = float(domain_loss_weight)
        self.max_grad_norm = max_grad_norm
        self.num_domains = 13
        weights = torch.as_tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
        self.emotion_criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=emotion_label_smoothing,
        )
        self.evaluation_criterion = nn.CrossEntropyLoss()

    def _inputs_to_device(self, inputs: dict[str, torch.Tensor]):
        return {
            name: tensor.to(self.device, non_blocking=True)
            for name, tensor in inputs.items()
        }

    def compute_training_loss(
        self,
        batch,
        epoch: int,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, torch.Tensor]:
        alpha = gradient_reversal_alpha(epoch, self.epochs)
        if self.model.domain_generalization:
            inputs, emotion_labels, domain_labels = batch
            domain_labels = domain_labels.to(self.device, non_blocking=True)
            if (
                torch.any(domain_labels < 0)
                or torch.any(domain_labels >= self.num_domains)
            ):
                raise ValueError(
                    f"Domain labels must be in [0, {self.num_domains - 1}]."
                )
        else:
            inputs, emotion_labels = batch
            domain_labels = None
        inputs = self._inputs_to_device(inputs)
        emotion_labels = emotion_labels.to(self.device, non_blocking=True)

        outputs = self.model(
            inputs.get("eeg"),
            inputs.get("eye"),
            alpha=alpha,
        )
        if self.model.domain_generalization:
            emotion_logits, domain_logits = outputs
        else:
            emotion_logits = outputs
            domain_logits = None
        emotion_loss = self.emotion_criterion(emotion_logits, emotion_labels)
        domain_loss = torch.zeros((), device=self.device)
        smoothing = 0.0
        if domain_logits is not None:
            smoothing = domain_label_smoothing(
                epoch,
                self.epochs,
                self.num_domains,
            )
            domain_loss = F.cross_entropy(
                domain_logits,
                domain_labels,
                label_smoothing=smoothing,
            )
        total_loss = emotion_loss + self.domain_loss_weight * domain_loss
        details = {
            "loss": float(total_loss.detach()),
            "emotion_loss": float(emotion_loss.detach()),
            "domain_loss": float(domain_loss.detach()),
            "alpha": alpha,
            "domain_label_smoothing": smoothing,
        }
        return total_loss, details, emotion_logits, emotion_labels

    def train_epoch(self, train_loader, epoch: int, scheduler=None) -> dict:
        self.model.train()
        totals = {
            "loss": 0.0,
            "emotion_loss": 0.0,
            "domain_loss": 0.0,
        }
        sample_count = 0
        predictions = []
        targets = []
        last_details = {}
        for batch in tqdm(
            train_loader,
            desc=f"Train {epoch + 1:03d}/{self.epochs:03d}",
            unit="batch",
            leave=False,
        ):
            self.optimizer.zero_grad(set_to_none=True)
            loss, details, logits, labels = self.compute_training_loss(batch, epoch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite MAET training loss.")
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
            self.optimizer.step()
            if scheduler is not None:
                scheduler.step()

            batch_size = labels.shape[0]
            sample_count += batch_size
            for key in totals:
                totals[key] += details[key] * batch_size
            predictions.append(logits.detach().argmax(dim=1).cpu())
            targets.append(labels.detach().cpu())
            last_details = details
        predictions = torch.cat(predictions).numpy()
        targets = torch.cat(targets).numpy()
        return {
            **{key: value / sample_count for key, value in totals.items()},
            "accuracy_pct": float(accuracy_score(targets, predictions) * 100.0),
            "macro_f1": float(
                f1_score(targets, predictions, average="macro", zero_division=0)
            ),
            "alpha": last_details["alpha"],
            "domain_label_smoothing": last_details["domain_label_smoothing"],
        }

    def evaluate(self, data_loader) -> tuple[dict, np.ndarray, np.ndarray]:
        self.model.eval()
        running_loss = 0.0
        sample_count = 0
        predictions = []
        targets = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = self._inputs_to_device(inputs)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(
                    inputs.get("eeg"),
                    inputs.get("eye"),
                    alpha=0.0,
                )
                emotion_logits = outputs[0] if isinstance(outputs, tuple) else outputs
                loss = self.evaluation_criterion(emotion_logits, labels)
                if not torch.isfinite(loss) or not torch.isfinite(
                    emotion_logits
                ).all():
                    raise FloatingPointError("Non-finite MAET evaluation result.")
                batch_size = labels.shape[0]
                running_loss += float(loss) * batch_size
                sample_count += batch_size
                predictions.append(emotion_logits.argmax(dim=1).cpu())
                targets.append(labels.cpu())
        predictions_array = torch.cat(predictions).numpy()
        targets_array = torch.cat(targets).numpy()
        metrics = classification_metrics(targets_array, predictions_array)
        metrics["loss"] = running_loss / sample_count
        return metrics, targets_array, predictions_array

    def fit(
        self,
        train_loader,
        validation_loader,
        checkpoint_path: Path,
        patience: int,
        max_lr: float,
        div_factor: float,
        final_div_factor: float,
        pct_start: float,
    ) -> dict:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            epochs=self.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
            anneal_strategy="linear",
        )
        best_f1 = float("-inf")
        best_accuracy = 0.0
        stale_epochs = 0
        history = []
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.epochs):
            training_metrics = self.train_epoch(train_loader, epoch, scheduler)
            validation_metrics, _, _ = self.evaluate(validation_loader)
            epoch_metrics = {
                "epoch": epoch + 1,
                "train": training_metrics,
                "validation": validation_metrics,
                "lr": float(scheduler.get_last_lr()[0]),
            }
            history.append(epoch_metrics)
            print(
                f"Epoch {epoch + 1:03d}: "
                f"train loss={training_metrics['loss']:.4f}, "
                f"val loss={validation_metrics['loss']:.4f}, "
                f"val accuracy={validation_metrics['accuracy_pct']:.2f}%, "
                f"val macro-F1={validation_metrics['macro_f1']:.4f}"
            )
            if validation_metrics["macro_f1"] > best_f1:
                best_f1 = validation_metrics["macro_f1"]
                best_accuracy = validation_metrics["accuracy_pct"]
                stale_epochs = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"Early stopping after epoch {epoch + 1}.")
                    break
        return {
            "checkpoint_path": str(checkpoint_path),
            "best_validation_macro_f1": float(best_f1),
            "best_validation_accuracy_pct": float(best_accuracy),
            "epochs_completed": len(history),
            "history": history,
        }


def _fold_metadata(fold: int) -> dict:
    split = get_subject_split(fold)
    return {
        "fold": fold,
        "train_subjects": list(split["train"]),
        "validation_subject": split["validation"][0],
        "test_subject": split["test"][0],
    }


def build_model(args: argparse.Namespace) -> SeedIVMAET:
    return SeedIVMAET(
        mode=args.mode,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        mixffn_start_layer_index=args.mixffn_start_layer_index,
        domain_generalization=args.domain_generalization,
        num_domains=13,
    )


def run_fold(
    args: argparse.Namespace,
    fold: int,
    fold_directory: Path,
) -> dict:
    set_reproducibility(args)
    use_cuda = bool(args.cuda and torch.cuda.is_available())
    if args.cuda and not use_cuda:
        print("CUDA requested but unavailable; using CPU.")
    device = torch.device(f"cuda:{args.gpu_num}" if use_cuda else "cpu")

    train_loader, validation_loader, test_loader, class_weights = (
        build_seed_iv_loaders(
            dataset_root=args.dataset_root,
            fold=fold,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=use_cuda,
            eeg_feature_kind=args.eeg_feature_kind,
            seed=args.seed,
        )
    )
    epochs = args.epochs
    patience = args.patience
    if args.smoke_test:
        epochs = args.smoke_epochs
        patience = min(patience, epochs)
        train_loader = limited_loader(
            train_loader,
            args.smoke_train_samples,
            args.seed,
            shuffle=True,
        )
        validation_loader = limited_loader(
            validation_loader,
            args.smoke_eval_samples,
            args.seed,
            shuffle=False,
        )
        test_loader = limited_loader(
            test_loader,
            args.smoke_eval_samples,
            args.seed,
            shuffle=False,
        )
    domain_mapping = build_domain_mapping(fold)
    if args.domain_generalization:
        train_loader = add_domain_labels(
            train_loader,
            domain_mapping,
            args.seed,
        )

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr / args.div_factor,
        weight_decay=args.weight_decay,
    )
    trainer = MAETTrainer(
        model=model,
        optimizer=optimizer,
        class_weights=class_weights,
        epochs=epochs,
        device=device,
        domain_loss_weight=args.domain_loss_weight,
        emotion_label_smoothing=args.emotion_label_smoothing,
        max_grad_norm=args.max_grad_norm,
    )
    fold_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = fold_directory / "best.pt"
    training_result = trainer.fit(
        train_loader,
        validation_loader,
        checkpoint_path=checkpoint_path,
        patience=patience,
        max_lr=args.max_lr,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
        pct_start=args.pct_start,
    )
    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    trainer.model.load_state_dict(state_dict)
    test_metrics, _, _ = trainer.evaluate(test_loader)

    result = {
        "model": "SeedIVMAET",
        "mode": args.mode,
        "domain_generalization": bool(args.domain_generalization),
        **_fold_metadata(fold),
        "domain_mapping": (
            {str(subject): label for subject, label in domain_mapping.items()}
            if args.domain_generalization
            else None
        ),
        "smoke_test": bool(args.smoke_test),
        "parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        **{
            key: value
            for key, value in training_result.items()
            if key != "history"
        },
        "test_loss": float(test_metrics.pop("loss")),
        **test_metrics,
    }
    save_json_atomic(fold_directory / "metrics.json", result)
    save_json_atomic(
        fold_directory / "training_history.json",
        {"epochs": training_result["history"]},
    )
    return result


def summarize_results(
    args: argparse.Namespace,
    fold_results: list[dict],
) -> dict:
    def aggregate(key):
        values = np.asarray(
            [result[key] for result in fold_results],
            dtype=np.float64,
        )
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }

    completed_test_subjects = sorted(
        result["test_subject"] for result in fold_results
    )
    return {
        "model": "SeedIVMAET",
        "mode": args.mode,
        "domain_generalization": bool(args.domain_generalization),
        "seed": args.seed,
        "smoke_test": bool(args.smoke_test),
        "completed_folds": len(fold_results),
        "expected_folds": len(SUBJECT_IDS),
        "completed_test_subjects": completed_test_subjects,
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


def experiment_directory(args: argparse.Namespace) -> Path:
    variant = "dg" if args.domain_generalization else "vanilla"
    directory = (
        Path(args.output_dir)
        / args.mode
        / variant
        / f"seed_{args.seed}"
    )
    return directory / "smoke" if args.smoke_test else directory


def main(args: argparse.Namespace) -> dict:
    if not 0 <= args.fold < len(SUBJECT_IDS):
        raise ValueError(f"fold must be between 0 and {len(SUBJECT_IDS) - 1}.")
    folds = range(len(SUBJECT_IDS)) if args.all_folds else (args.fold,)
    output_directory = experiment_directory(args)
    fold_results = []
    for fold in folds:
        variant = "DG" if args.domain_generalization else "vanilla"
        print(f"\n========== MAET-{args.mode} {variant} fold {fold:02d} ==========")
        result = run_fold(
            copy.deepcopy(args),
            fold,
            output_directory / f"fold_{fold:02d}",
        )
        fold_results.append(result)
        summary = summarize_results(args, fold_results)
        save_json_atomic(output_directory / "loso_summary.json", summary)
        print(
            f"fold {fold:02d}: Accuracy={result['accuracy_pct']:.2f}%, "
            f"Macro-F1={result['macro_f1']:.4f}"
        )
    return summary


if __name__ == "__main__":
    main(parse_args())
