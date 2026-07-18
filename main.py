import argparse
import os
import yaml
import numpy as np
from data.dataloader import MyDataLoader
import torch
from training import Trainer
import wandb
import random
from multiprocessing import cpu_count
from models.hyperfusenet import HyperFuseNet
from models.hyperfusenetv2 import HyperFuseNetv2
from models.hypernet import PHemoNet, HyperNetv2
from models.h2 import H2
from models.cross_attention_h2 import CrossAttentionH2
from models.baselines import ConvNet

def parse_num_workers(value):
    if value == 'max':
        return value
    workers = int(value)
    if workers < 0:
        raise argparse.ArgumentTypeError("num_workers must be non-negative or 'max'")
    return workers

def main(args, n_workers):

    # 设置类别数量
    num_classes = 3
    lr = args.max_lr / 10
    # lr = args.lr
    train_loader, eval_loader, sample_weights = MyDataLoader(train_file=args.train_file_path, 
                                                             test_file=args.test_file_path, 
                                                             train_batch_size=args.train_batch_size, 
                                                             test_batch_size=args.test_batch_size,
                                                             num_workers=n_workers,
                                                             pin_memory=args.cuda and torch.cuda.is_available())

    eye, gsr, eeg, ecg = next(iter(train_loader))[0]
    print("Eye shape: ", eye.shape)
    print("GSR shape: ", gsr.shape)
    print("EEG shape: ", eeg.shape)
    print("ECG shape: ", ecg.shape)       
    
    if args.model == 'HyperFuseNet': # ICASSPW 2023 中提出的模型
        net = HyperFuseNet(n=args.n, dropout_rate=args.dropout_rate)
    elif args.model == 'HyperFuseNetv2': # 轻量化版本：移除部分层，用于消融实验
        net = HyperFuseNetv2(n=args.n, dropout_rate=args.dropout_rate)
    elif args.model == 'PHemoNet': # RTSI 2024 中提出的模型，与 HyperFuseNetv2 相同，但编码器使用 PHM 层（又称 HyperNet）
        net = PHemoNet(n=args.n, dropout_rate=args.dropout_rate, 
                       n_eye=args.n_eye, n_gsr=args.n_gsr, n_eeg=args.n_eeg, n_ecg=args.n_ecg)
    elif args.model == 'HyperNetv2': # 编码器使用 PHM 层，但每个编码器仅移除一层（与 HyperNet 不同）
        net = HyperNetv2(n=args.n, dropout_rate=args.dropout_rate, 
                       n_eye=args.n_eye, n_gsr=args.n_gsr, n_eeg=args.n_eeg, n_ecg=args.n_ecg)
    elif args.model == 'H2': # MLSP 2024 中提出的模型（又称 ConvHyperNet）
        net = H2(n=args.n, dropout_rate=args.dropout_rate, 
                           n_eye=args.n_eye, n_gsr=args.n_gsr, n_eeg=args.n_eeg, n_ecg=args.n_ecg)
    elif args.model == 'CrossAttentionH2':
        net = CrossAttentionH2(
            n=args.n,
            dropout_rate=args.dropout_rate,
            n_eye=args.n_eye,
            n_gsr=args.n_gsr,
            n_eeg=args.n_eeg,
            n_ecg=args.n_ecg,
            attention_dim=args.attention_dim,
            attention_heads=args.attention_heads,
            attention_layers=args.attention_layers,
            attention_dropout=args.attention_dropout,
            num_classes=num_classes,
        )
    elif args.model == 'ConvNet': # 与 H2 相同，但编码器使用卷积，用于消融实验
        net = ConvNet(dropout_rate=args.dropout_rate)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    wandb.init(project="MHyEEG")
    wandb.config.update(args, allow_val_change=True)
    if args.wandb_watch:
        wandb.watch(net)
    
    # 统计神经网络参数数量
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Number of parameters:', params)
    print()
    
    # 初始化优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=args.weight_decay, eps=1e-7)
    
    # 训练/评估模型
    trainer = Trainer(net, optimizer, epochs=args.epochs,
                      use_cuda=args.cuda, gpu_num=args.gpu_num,
                      checkpoint_folder=args.checkpoint_folder,
                      max_lr=args.max_lr, min_mom=args.min_mom,
                      max_mom=args.max_mom, l1_reg=args.l1_reg,
                      num_classes=num_classes,
                      sample_weights=sample_weights,
                      es_mode=args.es_mode,
                      patience=args.patience,
                      amp=args.amp,
                      amp_dtype=args.amp_dtype,
                      fp32_finetune_epochs=args.fp32_finetune_epochs,
                      allow_tf32=args.allow_tf32,
                      wandb_log_interval=args.wandb_log_interval)
    
    trainer.train(train_loader, eval_loader, 
                  div_factor=args.div_factor, 
                  final_div_factor=args.final_div_factor, 
                  pct_start=args.pct_start, 
                  max_lr=args.max_lr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_file_path', type=str, default='hci-tagging-database/torch_datasets/train_augmented_data_Arsl.pt', help='Path to training .pt file')
    parser.add_argument('--test_file_path', type=str, default='hci-tagging-database/torch_datasets/test_data_Arsl.pt', help='Path to test .pt file')
    parser.add_argument('--checkpoint_folder', type=str, default='checkpoints')
    parser.add_argument('--model', type=str, default='H2', help='Model to use (HyperFuseNet, PHemoNet, H2, CrossAttentionH2)')
    parser.add_argument('--num_workers', type=parse_num_workers, default=1, help="Number of workers, 'max' for maximum number")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument('--n', type=int, default=4, help="n parameter for PHM layers")
    parser.add_argument('--n_eye', type=int, default=4, help="n parameter for PHM layers")	
    parser.add_argument('--n_gsr', type=int, default=1, help="n parameter for PHM layers")
    parser.add_argument('--n_eeg', type=int, default=10, help="n parameter for PHM layers")
    parser.add_argument('--n_ecg', type=int, default=3, help="n parameter for PHM layers")
    parser.add_argument('--attention_dim', type=int, default=256)
    parser.add_argument('--attention_heads', type=int, default=8)
    parser.add_argument('--attention_layers', type=int, default=2)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--l1_reg', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--min_mom', type=float, default=0.7403, help="Minimum momentum for OneCycleLR")
    parser.add_argument('--max_mom', type=float, default=0.7985, help="Maximum momentum for OneCycleLR")
    parser.add_argument('--max_lr', type=float, default=0.00000796, help="Maximum learning rate for OneCycleLR")
    parser.add_argument('--div_factor', type=int, default=10, help="div factor for OneCycleLR")
    parser.add_argument('--final_div_factor', type=int, default=10, help="final div factor for OneCycleLR")
    parser.add_argument('--pct_start', type=float, default=0.425, help="pct_start for OneCycleLR")
    # parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--es_mode', type=str, default='max', help="mode for EarlyStopping, 'max' or 'min'")
    parser.add_argument('--patience', type=int, default=10, help="patience for EarlyStopping, 20 for HyperFuseNet and 10 for the others")
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False, help='Enable automatic mixed precision')
    parser.add_argument('--amp_dtype', choices=['bfloat16', 'float16'], default='bfloat16')
    parser.add_argument('--fp32_finetune_epochs', type=int, default=0, help='Use full FP32 for the final N epochs')
    parser.add_argument('--allow_tf32', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--deterministic', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--cudnn_benchmark', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--wandb_watch', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--wandb_log_interval', type=int, default=1, help='Log step learning rate every N batches')
    parser.add_argument('--config', type=str, help='Path to YAML config file')

    # 首先解析已知参数以获取配置文件路径
    config_args, _ = parser.parse_known_args()

    # 加载 YAML 配置并覆盖默认值
    if config_args.config:
        with open(config_args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            parser.set_defaults(**config_dict)  # 使用配置覆盖解析器的默认值
    args = parser.parse_args()

    seed = args.seed
    n_workers = args.num_workers

    if n_workers == 'max':
        n_workers = cpu_count()  # 获取系统中的 CPU 数量
    
    # 设置随机种子
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = args.deterministic
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.use_deterministic_algorithms(args.deterministic, warn_only=True)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)

    main(args, n_workers)
