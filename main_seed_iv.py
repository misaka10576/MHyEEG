"""Train SEED-IV EEG/eye baselines with subject-independent folds."""

import argparse
import os
import random
from multiprocessing import cpu_count

import numpy as np
import torch
import wandb
import yaml

from data.seed_iv import build_seed_iv_loaders
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
    parser.add_argument("--attention_dropout", type=float, default=0.2)
    parser.add_argument("--fuzzy_rules", type=int, default=8)
    parser.add_argument("--uncertainty_penalty", type=float, default=1.0)
    parser.add_argument("--num_workers", type=parse_num_workers, default=4)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_mom", type=float, default=0.85)
    parser.add_argument("--max_mom", type=float, default=0.95)
    parser.add_argument("--div_factor", type=float, default=10)
    parser.add_argument("--final_div_factor", type=float, default=100)
    parser.add_argument("--pct_start", type=float, default=0.3)
    parser.add_argument("--checkpoint_folder", type=str, default="checkpoints/seed_iv")
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--amp_dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--fp32_finetune_epochs", type=int, default=10)
    parser.add_argument(
        "--allow_tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--cudnn_benchmark",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = args.cudnn_benchmark


def main(args):
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
    )
    training_result = trainer.train(
        train_loader,
        validation_loader,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
        pct_start=args.pct_start,
        max_lr=args.max_lr,
    )
    trainer.evaluate(
        test_loader,
        checkpoint_path=training_result["checkpoint_path"],
        split="test",
    )
    wandb.finish()


if __name__ == "__main__":
    main(parse_args())
