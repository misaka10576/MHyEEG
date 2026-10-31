"""Two-stage training protocol for one BERN subject/session run."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn

from .data import (
    CLASS_NAMES,
    FeatureStandardizer,
    RunArrays,
    make_loader,
    make_trial_split,
)
from .model import BERN


@dataclass(frozen=True)
class TrainingConfig:
    mode: str
    subject: int
    session: int
    dataset_root: str
    output_dir: str
    cache_dir: str
    eeg_feature_kind: str = "de_LDS"
    seed: int = 2024
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 500
    patience: int = 50
    num_workers: int = 0
    device: str = "cuda"
    pretrained: bool = True
    deterministic: bool = True
    smoke_test: bool = False
    max_batches_per_epoch: int | None = None
    fusion_num_tokens: int = 8
    fusion_token_dim: int = 32
    fusion_num_heads: int = 4
    fusion_variant: str = "vector_token_v2"


def set_reproducibility(seed: int, deterministic: bool = True) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # MultiheadAttention otherwise selects the memory-efficient CUDA SDP
        # kernel, which PyTorch reports as non-deterministic on this hardware.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. BERN does not silently fall back "
            "to CPU; expose the GPU or explicitly pass --device cpu."
        )
    return torch.device(requested)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _model_factory(config: TrainingConfig, use_pretrained: bool) -> BERN:
    return BERN(
        mode=config.mode,
        pretrained=use_pretrained,
        cache_dir=config.cache_dir,
        fusion_num_tokens=config.fusion_num_tokens,
        fusion_token_dim=config.fusion_token_dim,
        fusion_num_heads=config.fusion_num_heads,
        fusion_variant=config.fusion_variant,
    )


def _fusion_architecture(config: TrainingConfig) -> dict[str, str | int] | None:
    if config.mode != "fusion":
        return None
    return {
        "fusion_variant": config.fusion_variant,
        "fusion_num_tokens": config.fusion_num_tokens,
        "fusion_token_dim": config.fusion_token_dim,
        "fusion_num_heads": config.fusion_num_heads,
    }


def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch_index, (eeg, eye, labels, _, _) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            eeg = eeg.to(device, non_blocking=True)
            eye = eye.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(eeg, eye)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {('train' if training else 'eval')} loss.")
            if training:
                loss.backward()
                optimizer.step()
            count = int(labels.shape[0])
            total_loss += float(loss.detach()) * count
            total_count += count
    if not total_count:
        raise RuntimeError("No samples were processed in an epoch.")
    return total_loss / total_count


def _predict(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    collect_attention: bool = False,
) -> tuple[
    list[int],
    list[int],
    list[int],
    list[int],
    list[list[float]],
    list[dict[str, float]] | None,
]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    trials: list[int] = []
    windows: list[int] = []
    logits_output: list[list[float]] = []
    attention_batches: list[list[torch.Tensor]] = []
    with torch.inference_mode():
        for eeg, eye, labels, batch_trials, batch_windows in loader:
            model_output = model(
                eeg.to(device, non_blocking=True),
                eye.to(device, non_blocking=True),
                **({"return_attention": True} if collect_attention else {}),
            )
            if collect_attention:
                logits, batch_attention = model_output
                attention_batches.append(
                    [weights.detach().float().cpu() for weights in batch_attention]
                )
            else:
                logits = model_output
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite test logits.")
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            trials.extend(batch_trials.tolist())
            windows.extend(batch_windows.tolist())
            logits_output.extend(logits.cpu().tolist())
    diagnostics = (
        _attention_diagnostics(attention_batches) if collect_attention else None
    )
    return targets, predictions, trials, windows, logits_output, diagnostics


def _attention_diagnostics(
    batches: list[list[torch.Tensor]],
) -> list[dict[str, float]]:
    """Summarize attention already produced by the sole held-out test pass."""
    if not batches:
        raise RuntimeError("No attention weights were collected.")
    layer_count = len(batches[0])
    if layer_count != 2 or any(len(batch) != layer_count for batch in batches):
        raise RuntimeError("Expected exactly two attention layers in every batch.")
    output: list[dict[str, float]] = []
    for layer_index in range(layer_count):
        weights = torch.cat(
            [batch[layer_index] for batch in batches], dim=0
        ).to(torch.float64)
        if weights.ndim != 4:
            raise RuntimeError(
                f"Expected attention [B,H,Q,K], received {tuple(weights.shape)}"
            )
        key_count = weights.shape[-1]
        entropy = -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()).sum(-1)
        normalized_entropy = entropy / math.log(key_count) if key_count > 1 else entropy * 0
        uniform = 1.0 / key_count
        relative_uniform_deviation = (weights - uniform).abs().mean() / uniform
        attention_variance = weights.var(dim=-1, unbiased=False).mean()
        values = {
            "layer": layer_index + 1,
            "normalized_entropy": float(normalized_entropy.mean()),
            "relative_uniform_deviation": float(relative_uniform_deviation),
            "attention_variance": float(attention_variance),
            "sample_count": int(weights.shape[0]),
        }
        if not all(
            math.isfinite(value)
            for key, value in values.items()
            if key not in {"layer", "sample_count"}
        ):
            raise FloatingPointError("Non-finite attention diagnostics.")
        output.append(values)
    return output


def classification_metrics(targets: Iterable[int], predictions: Iterable[int]) -> dict:
    targets_array = np.asarray(list(targets), dtype=np.int64)
    predictions_array = np.asarray(list(predictions), dtype=np.int64)
    matrix = np.zeros((4, 4), dtype=np.int64)
    np.add.at(matrix, (targets_array, predictions_array), 1)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = np.divide(
        true_positive, predicted, out=np.zeros(4), where=predicted != 0
    )
    recall = np.divide(
        true_positive, support, out=np.zeros(4), where=support != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(4),
        where=(precision + recall) != 0,
    )
    total = int(matrix.sum())
    return {
        "accuracy_pct": float(100.0 * true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def select_best_epoch(
    config: TrainingConfig,
    train_arrays: RunArrays,
    validation_arrays: RunArrays,
    scaler: FeatureStandardizer,
    run_dir: Path,
    resume: bool,
    model_builder: Callable[[bool], nn.Module] | None = None,
) -> tuple[int, list[dict[str, float]]]:
    """Select an epoch from training trials only; this function has no test input."""
    device = resolve_device(config.device)
    model_builder = model_builder or (lambda pretrained: _model_factory(config, pretrained))
    last_path = run_dir / "selection_last.pt"
    best_path = run_dir / "selection_best.pt"
    complete_path = run_dir / "selection.json"
    if resume and complete_path.is_file() and best_path.is_file():
        value = json.loads(complete_path.read_text(encoding="utf-8"))
        return int(value["best_epoch"]), value["history"]

    set_reproducibility(config.seed, config.deterministic)
    resuming = resume and last_path.is_file()
    model = model_builder(config.pretrained and not resuming).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    start_epoch, best_epoch, bad_epochs = 1, 0, 0
    best_loss = math.inf
    history: list[dict[str, float]] = []
    if resuming:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_loss = float(checkpoint["best_loss"])
        bad_epochs = int(checkpoint["bad_epochs"])
        history = checkpoint["history"]

    train_loader = make_loader(
        train_arrays, config.batch_size, True, config.seed, config.num_workers
    )
    validation_loader = make_loader(
        validation_arrays, config.batch_size, False, config.seed, config.num_workers
    )
    for epoch in range(start_epoch, config.max_epochs + 1):
        train_loss = _run_epoch(
            model, train_loader, device, criterion, optimizer, config.max_batches_per_epoch
        )
        validation_loss = _run_epoch(
            model, validation_loader, device, criterion, None, config.max_batches_per_epoch
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        improved = validation_loss < best_loss
        if improved:
            best_loss, best_epoch, bad_epochs = validation_loss, epoch, 0
            _atomic_torch_save(
                best_path,
                {
                    "stage": "selection",
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "model": model.state_dict(),
                    "scaler": scaler.state_dict(),
                    "fusion_architecture": _fusion_architecture(config),
                },
            )
        else:
            bad_epochs += 1
        _atomic_torch_save(
            last_path,
            {
                "stage": "selection",
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "fusion_architecture": _fusion_architecture(config),
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "bad_epochs": bad_epochs,
                "history": history,
            },
        )
        print(
            f"[{config.mode} S{config.subject:02d}/session{config.session}] "
            f"selection {epoch}/{config.max_epochs}: train={train_loss:.6f} "
            f"validation={validation_loss:.6f} best_epoch={best_epoch}",
            flush=True,
        )
        if bad_epochs >= config.patience:
            break
    if best_epoch < 1:
        raise RuntimeError("Selection did not produce a best epoch.")
    _atomic_json(
        complete_path,
        {
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "epochs_ran": len(history),
            "stopped_early": len(history) < config.max_epochs,
            "history": history,
            "test_metrics_observed": False,
        },
    )
    # The best checkpoint is the durable selection artifact. The larger
    # optimizer-bearing "last" checkpoint is needed only while interrupted.
    last_path.unlink(missing_ok=True)
    return best_epoch, history


def train_final(
    config: TrainingConfig,
    train_arrays: RunArrays,
    scaler: FeatureStandardizer,
    epochs: int,
    run_dir: Path,
    resume: bool,
    model_builder: Callable[[bool], nn.Module] | None = None,
) -> nn.Module:
    """Reinitialize, then train on all trials 1..16 for the selected epoch count."""
    if epochs < 1:
        raise ValueError("Final training needs at least one epoch.")
    device = resolve_device(config.device)
    model_builder = model_builder or (lambda pretrained: _model_factory(config, pretrained))
    last_path = run_dir / "final_last.pt"
    final_path = run_dir / "final_checkpoint.pt"
    set_reproducibility(config.seed, config.deterministic)
    resuming = resume and last_path.is_file() and not final_path.is_file()
    restoring_final = resume and final_path.is_file()
    model = model_builder(
        config.pretrained and not (resuming or restoring_final)
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    start_epoch = 1
    history: list[dict[str, float]] = []
    if resuming:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if int(checkpoint["target_epochs"]) != epochs:
            raise RuntimeError("Selected epoch count changed since the final-stage checkpoint.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = checkpoint["history"]
    elif resume and final_path.is_file():
        checkpoint = torch.load(final_path, map_location=device, weights_only=False)
        if int(checkpoint["epochs"]) != epochs:
            raise RuntimeError("Final checkpoint does not match selected epoch count.")
        model.load_state_dict(checkpoint["model"])
        return model

    train_loader = make_loader(
        train_arrays, config.batch_size, True, config.seed, config.num_workers
    )
    for epoch in range(start_epoch, epochs + 1):
        train_loss = _run_epoch(
            model, train_loader, device, criterion, optimizer, config.max_batches_per_epoch
        )
        history.append({"epoch": epoch, "train_loss": train_loss})
        _atomic_torch_save(
            last_path,
            {
                "stage": "final",
                "epoch": epoch,
                "target_epochs": epochs,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "fusion_architecture": _fusion_architecture(config),
                "history": history,
            },
        )
        print(
            f"[{config.mode} S{config.subject:02d}/session{config.session}] "
            f"final {epoch}/{epochs}: train={train_loss:.6f}",
            flush=True,
        )
    _atomic_torch_save(
        final_path,
        {
            "stage": "final",
            "epochs": epochs,
            "model": model.state_dict(),
            "scaler": scaler.state_dict(),
            "fusion_architecture": _fusion_architecture(config),
            "history": history,
            "training_trials": list(range(1, 17)),
            "test_metrics_observed_during_training": False,
        },
    )
    last_path.unlink(missing_ok=True)
    return model


def run_experiment(
    raw: RunArrays,
    config: TrainingConfig,
    resume: bool = False,
    model_builder: Callable[[bool], nn.Module] | None = None,
) -> dict:
    """Run selection, reinitialization, final training, and exactly one test pass."""
    split = make_trial_split(config.session)
    base = Path(config.output_dir)
    if config.smoke_test:
        base = base / "smoke"
    run_dir = base / config.mode / f"subject_{config.subject:02d}" / f"session_{config.session}"
    metrics_path = run_dir / "metrics.json"
    if resume and metrics_path.is_file():
        print(f"Skipping completed run: {run_dir}", flush=True)
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics_path.is_file():
        raise FileExistsError(
            f"{run_dir} is already complete. Pass --resume to skip it or choose "
            "a different output directory."
        )
    if run_dir.is_dir() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError(
            f"{run_dir} contains a partial run. Pass --resume or choose a "
            "different output directory."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **asdict(config),
        "split": split.as_dict(),
        "protocol": "subject_within_session_first16_train_last8_test",
    }
    config_path = run_dir / "config.json"
    if resume and config_path.is_file():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        if saved_config != run_config:
            raise RuntimeError(
                f"Resume configuration differs from {config_path}; use a new "
                "output directory for changed settings."
            )
    else:
        _atomic_json(config_path, run_config)

    selection_raw = raw.subset(split.selection_train)
    validation_raw = raw.subset(split.validation)
    selection_scaler = FeatureStandardizer.fit(selection_raw)
    selection_train = selection_scaler.transform(selection_raw)
    validation = selection_scaler.transform(validation_raw)
    best_epoch, selection_history = select_best_epoch(
        config,
        selection_train,
        validation,
        selection_scaler,
        run_dir,
        resume,
        model_builder,
    )

    # Refit statistics and reinitialize the network using all 16 training trials.
    final_raw = raw.subset(split.final_train)
    final_scaler = FeatureStandardizer.fit(final_raw)
    final_train_arrays = final_scaler.transform(final_raw)
    model = train_final(
        config,
        final_train_arrays,
        final_scaler,
        best_epoch,
        run_dir,
        resume,
        model_builder,
    )

    # The held-out trials are first transformed/evaluated only after training ends.
    test_raw = raw.subset(split.test)
    test_arrays = final_scaler.transform(test_raw)
    test_loader = make_loader(
        test_arrays, config.batch_size, False, config.seed, config.num_workers
    )
    predictions_path = run_dir / "predictions.json"
    evaluation_path = run_dir / "test_evaluation.json"
    diagnostics_path = run_dir / "attention_diagnostics.json"
    if evaluation_path.is_file():
        # This single atomic artifact records everything produced by the one
        # permitted held-out pass. The public files can be reconstructed from it.
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        prediction_rows = evaluation["predictions"]
        attention_diagnostics = evaluation.get("attention_diagnostics")
        if not predictions_path.is_file():
            _atomic_json(predictions_path, prediction_rows)
        if attention_diagnostics is not None and not diagnostics_path.is_file():
            _atomic_json(diagnostics_path, attention_diagnostics)
    elif predictions_path.is_file():
        # Backward-compatible recovery for non-fusion runs created before the
        # combined test artifact was introduced.
        prediction_rows = json.loads(predictions_path.read_text(encoding="utf-8"))
        if (
            len(prediction_rows) != len(test_arrays)
            or {int(row["trial"]) for row in prediction_rows} != set(split.test)
        ):
            raise RuntimeError(f"Incomplete or incompatible predictions in {predictions_path}")
        targets = [int(row["target"]) for row in prediction_rows]
        predictions = [int(row["prediction"]) for row in prediction_rows]
        attention_diagnostics = (
            json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if config.mode == "fusion" and diagnostics_path.is_file()
            else None
        )
        if config.mode == "fusion" and attention_diagnostics is None:
            raise RuntimeError(
                "Fusion predictions exist without attention diagnostics; refusing "
                "a second held-out test pass."
            )
    else:
        targets, predictions, trials, windows, logits, attention_diagnostics = _predict(
            model,
            test_loader,
            resolve_device(config.device),
            collect_attention=config.mode == "fusion",
        )
        prediction_rows = [
            {
                "trial": int(trial),
                "window_in_trial": int(window),
                "target": int(target),
                "prediction": int(prediction),
                "logits": [float(value) for value in sample_logits],
            }
            for trial, window, target, prediction, sample_logits in zip(
                trials, windows, targets, predictions, logits
            )
        ]
        _atomic_json(
            evaluation_path,
            {
                "test_evaluations": 1,
                "predictions": prediction_rows,
                "attention_diagnostics": attention_diagnostics,
            },
        )
        if attention_diagnostics is not None:
            _atomic_json(diagnostics_path, attention_diagnostics)
        _atomic_json(predictions_path, prediction_rows)
    targets = [int(row["target"]) for row in prediction_rows]
    predictions = [int(row["prediction"]) for row in prediction_rows]
    metrics = {
        "mode": config.mode,
        "subject": config.subject,
        "session": config.session,
        "eeg_feature_kind": config.eeg_feature_kind,
        "seed": config.seed,
        "best_epoch": best_epoch,
        "selection_epochs_ran": len(selection_history),
        "selection_train_trials": list(split.selection_train),
        "validation_trials": list(split.validation),
        "final_train_trials": list(split.final_train),
        "test_trials": list(split.test),
        "selection_scaler_fit_trials": list(split.selection_train),
        "final_scaler_fit_trials": list(split.final_train),
        "test_evaluations": 1,
        "sample_count": len(targets),
        **(_fusion_architecture(config) or {}),
        **(
            {"attention_diagnostics": attention_diagnostics}
            if attention_diagnostics is not None
            else {}
        ),
        **classification_metrics(targets, predictions),
    }
    _atomic_json(metrics_path, metrics)
    return metrics
