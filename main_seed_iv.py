"""Train SEED-IV EEG/eye baselines with subject-independent folds."""

import argparse
import copy
import gc
import json
import os
import random
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

from data.seed_iv import SUBJECT_IDS, build_seed_iv_loaders, get_subject_split
from models.seed_iv_baselines import SeedIVBaseline
from models.seed_iv_fuzzy_attention import SeedIVFuzzyAttention
from training import Trainer


BASELINE_MODES = {
    "SeedIVEEG": "eeg",
    "SeedIVEye": "eye",
    "SeedIVConcat": "concat",
}
MODEL_NAMES = tuple(BASELINE_MODES) + (
    "SeedIVAttention",
    "SeedIVIT2FuzzyAttention",
)


def parse_num_workers(value):
    if value == "max":
        return value
    workers = int(value)
    if workers < 0:
        raise argparse.ArgumentTypeError(
            "num_workers must be non-negative or 'max'."
        )
    return workers


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--dataset_root", type=str, default="SEED_IV")
    parser.add_argument("--eeg_feature_kind", type=str, default="de_LDS")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", choices=MODEL_NAMES, default="SeedIVConcat")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--attention_dim", type=int, default=64)
    parser.add_argument("--attention_heads", type=int, default=4)
    parser.add_argument("--attention_layers", type=int, default=1)
    parser.add_argument("--attention_ffn_dim", type=int, default=128)
    parser.add_argument("--attention_dropout", type=float, default=0.3)
    parser.add_argument("--fuzzy_rules", type=int, default=8)
    parser.add_argument("--uncertainty_penalty", type=float, default=1.0)
    parser.add_argument("--num_workers", type=parse_num_workers, default=4)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_mom", type=float, default=0.85)
    parser.add_argument("--max_mom", type=float, default=0.95)
    parser.add_argument("--div_factor", type=float, default=10)
    parser.add_argument("--final_div_factor", type=float, default=100)
    parser.add_argument("--pct_start", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_folder", type=str, default="checkpoints/seed_iv")
    parser.add_argument(
        "--all_folds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run all 15 LOSO folds and aggregate held-out-subject metrics.",
    )
    parser.add_argument(
        "--loso_output_dir",
        type=str,
        default="checkpoints/seed_iv_loso",
        help="Base directory for all-fold checkpoints and summary JSON.",
    )
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--amp_dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--fp32_finetune_epochs", type=int, default=0)
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
    parser.add_argument(
        "--wandb_watch",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--wandb_log_interval", type=int, default=20)
    parser.add_argument("--wandb_project", type=str, default="MHyEEG-SEED-IV")
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb_name", type=str)
    return parser


def parse_args():
    parser = build_parser()
    config_args, _ = parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        parser.set_defaults(**config)
    return parser.parse_args()


def set_reproducibility(args):
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


def run_fold(args):
    set_reproducibility(args)
    num_workers = cpu_count() if args.num_workers == "max" else args.num_workers
    use_cuda = bool(args.cuda and torch.cuda.is_available())
    if args.cuda and not use_cuda:
        print("CUDA requested but unavailable; falling back to CPU.")

    os.makedirs(args.checkpoint_folder, exist_ok=True)
    train_loader, validation_loader, test_loader, class_weights = build_seed_iv_loaders(
        dataset_root=args.dataset_root,
        fold=args.fold,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.test_batch_size,
        num_workers=num_workers,
        pin_memory=use_cuda,
        eeg_feature_kind=args.eeg_feature_kind,
        seed=args.seed,
    )

    inputs, labels = next(iter(train_loader))
    print("EEG shape:", inputs["eeg"].shape)
    print("Eye shape:", inputs["eye"].shape)
    print("Label shape:", labels.shape)

    if args.model in BASELINE_MODES:
        network = SeedIVBaseline(
            mode=BASELINE_MODES[args.model],
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            dropout=args.dropout_rate,
            num_classes=4,
        )
    else:
        network = SeedIVFuzzyAttention(
            dimension=args.attention_dim,
            num_heads=args.attention_heads,
            num_layers=args.attention_layers,
            ffn_dimension=args.attention_ffn_dim,
            dropout=args.attention_dropout,
            num_classes=4,
            use_fuzzy=args.model == "SeedIVIT2FuzzyAttention",
            fuzzy_rules=args.fuzzy_rules,
            uncertainty_penalty=args.uncertainty_penalty,
        )
    parameter_count = sum(
        parameter.numel()
        for parameter in network.parameters()
        if parameter.requires_grad
    )
    print("Number of parameters:", parameter_count)
    print()

    wandb.init(
        project=args.wandb_project,
        mode=args.wandb_mode,
        name=args.wandb_name,
        config=vars(args),
    )
    try:
        if args.wandb_watch:
            wandb.watch(network)

        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=args.max_lr / args.div_factor,
            weight_decay=args.weight_decay,
            eps=1e-7,
        )
        trainer = Trainer(
            network,
            optimizer,
            epochs=args.epochs,
            use_cuda=use_cuda,
            gpu_num=args.gpu_num,
            checkpoint_folder=args.checkpoint_folder,
            max_lr=args.max_lr,
            min_mom=args.min_mom,
            max_mom=args.max_mom,
            l1_reg=False,
            num_classes=4,
            sample_weights=class_weights,
            es_mode="max",
            patience=args.patience,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            fp32_finetune_epochs=args.fp32_finetune_epochs,
            allow_tf32=args.allow_tf32,
            wandb_log_interval=args.wandb_log_interval,
            label_smoothing=args.label_smoothing,
            max_grad_norm=args.max_grad_norm,
        )
        training_result = trainer.train(
            train_loader,
            validation_loader,
            div_factor=args.div_factor,
            final_div_factor=args.final_div_factor,
            pct_start=args.pct_start,
            max_lr=args.max_lr,
        )
        test_result = trainer.evaluate(
            test_loader,
            checkpoint_path=training_result["checkpoint_path"],
            split="test",
        )
    finally:
        wandb.finish()

    split = get_subject_split(args.fold)
    return {
        "fold": args.fold,
        "train_subjects": list(split["train"]),
        "validation_subject": split["validation"][0],
        "test_subject": split["test"][0],
        "checkpoint_path": training_result["checkpoint_path"],
        "best_val_f1": training_result["best_val_f1"],
        "best_val_accuracy": training_result["best_val_acc"],
        "val_loss_at_best_f1": training_result["best_val_loss"],
        "minimum_val_loss": training_result["minimum_val_loss"],
        "test_loss": test_result["loss"],
        "test_accuracy": test_result["accuracy"],
        "test_f1": test_result["f1"],
    }


def summarize_loso_results(fold_results):
    """Return subject-level LOSO means and sample standard deviations."""
    if not fold_results:
        raise ValueError("fold_results must contain at least one completed fold.")

    aggregate = {}
    for metric in ("test_loss", "test_accuracy", "test_f1"):
        values = np.asarray(
            [result[metric] for result in fold_results],
            dtype=np.float64,
        )
        aggregate[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return aggregate


def save_loso_summary(output_directory, args, fold_results):
    """Persist partial or complete LOSO results after every successful fold."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "seed": args.seed,
        "completed_folds": len(fold_results),
        "expected_folds": len(SUBJECT_IDS),
        "complete": len(fold_results) == len(SUBJECT_IDS),
        "config": vars(args),
        "folds": fold_results,
        "aggregate": summarize_loso_results(fold_results),
    }
    summary_path = output_directory / "loso_summary.json"
    temporary_path = output_directory / "loso_summary.json.tmp"
    with temporary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(payload, summary_file, ensure_ascii=False, indent=2)
    temporary_path.replace(summary_path)
    return summary_path, payload


def run_all_folds(args):
    output_directory = (
        Path(args.loso_output_dir) / f"{args.model}_seed{args.seed}"
    )
    fold_results = []
    summary_path = None

    for fold in range(len(SUBJECT_IDS)):
        print(f"\n========== LOSO fold {fold:02d}/{len(SUBJECT_IDS) - 1:02d} ==========")
        fold_args = copy.deepcopy(args)
        fold_args.all_folds = False
        fold_args.fold = fold
        fold_args.checkpoint_folder = str(output_directory / f"fold_{fold:02d}")
        base_name = args.wandb_name or f"{args.model}-seed{args.seed}"
        fold_args.wandb_name = f"{base_name}-fold{fold:02d}"

        fold_results.append(run_fold(fold_args))
        summary_path, _ = save_loso_summary(
            output_directory,
            args,
            fold_results,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _, payload = save_loso_summary(output_directory, args, fold_results)
    accuracy = payload["aggregate"]["test_accuracy"]
    f1 = payload["aggregate"]["test_f1"]
    print("\n========== LOSO summary ==========")
    print(
        f"Accuracy: {accuracy['mean']:.4f} ± {accuracy['std']:.4f}"
    )
    print(f"Macro-F1: {f1['mean']:.4f} ± {f1['std']:.4f}")
    print(f"Results: {summary_path}")
    return payload


def main(args):
    if args.all_folds:
        return run_all_folds(args)
    return run_fold(args)


if __name__ == "__main__":
    main(parse_args())
