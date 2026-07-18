## 基于 EEG 与外周生理信号的超复数多模态情感识别 🎭

本项目是以下论文的官方 PyTorch 实现：

1. *Hypercomplex Multimodal Emotion Recognition from EEG and Peripheral Physiological Signals*，ICASSPW 2023。[[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10193329)] [[ArXiv 预印本](https://arxiv.org/abs/2310.07648)]
2. *Hierarchical Hypercomplex Network for Multimodal Emotion Recognition*，MLSP 2024。[[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10734815)] [[ArXiv 预印本](https://arxiv.org/abs/2409.09194)]
3. *PHemoNet: A Multimodal Network for Physiological Signals*，RTSI 2024。[[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10761462)] [[ArXiv 预印本](https://arxiv.org/abs/2410.00010)]

作者：

Eleonora Lopez、Eleonora Chiarantano、[Eleonora Grassucci](https://sites.google.com/uniroma1.it/eleonoragrassucci/home-page)、[Aurelio Uncini](https://www.uncini.com/) 和 [Danilo Comminiello](https://danilocomminiello.site.uniroma1.it/)，来自 [ISPAMM 实验室](https://sites.google.com/uniroma1.it/ispamm/) 🏘️

### 📰 最新动态

- [2025.05.15] 发布预训练权重 💣
- [2025.05.14] 根据 MLSP 和 RTSI 论文更新 H2 与 PHemoNet 模型代码 👩🏻‍💻
- [2024.07] 扩展论文被 MLSP 2024 和 RTSI 2024 接收
- [2023.11.11] 发布 HyperFuseNet 代码 👩🏼‍💻
- [2023.04.14] 论文被 ICASSP Workshop 2023 接收 🎉

### 📚 论文与模型

| 模型 | 论文 | 唤醒度 F1 | 唤醒度准确率 | 效价 F1 | 效价准确率 | 主要特点 | 权重 |
|---|---|---:|---:|---:|---:|---|---|
| 🥇 **H2** | MLSP 2024 [[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10734815)] [[ArXiv](https://arxiv.org/abs/2409.09194)] | **0.557** | **56.91** | **0.685** | **67.87** | 使用模态专属域 PHC 编码器的层次化模型，取得**最佳性能** | [唤醒度](https://drive.google.com/file/d/1xvC5mVaoHG2UINJv-jJ8_pR1R5f2z1oG/view?usp=sharing) - [效价](https://drive.google.com/file/d/1tBTmbxswkNTa9e_7_1RPnRQZSD-Kr9vf/view?usp=sharing) |
| 🥈 **PHemoNet** | RTSI 2024 [[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10761462)] [[ArXiv](https://arxiv.org/abs/2410.00010)] | 0.401 | 42.54 | 0.505 | 50.77 | 使用模态专属域 PHM 编码器和改进的超复数融合模块 | [唤醒度](https://drive.google.com/file/d/1d8tF93EtHXOmC0IOa_ID0gn9JYxAHxhF/view?usp=sharing) - [效价](https://drive.google.com/file/d/1b-KUJ_mhJhSG8AAHBeqE_39e8z67nwiJ/view?usp=sharing) |
| 🥉 **HyperFuseNet** | ICASSPW 2023 [[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10193329)] [[ArXiv](https://arxiv.org/abs/2310.07648)] | 0.397 | 41.56 | 0.436 | 44.30 | 引入超复数融合模块 | [唤醒度](https://drive.google.com/file/d/1VrOiBj2t_xwn-MIUPxSYVz_5gUZYhS6F/view?usp=sharing) - [效价](https://drive.google.com/file/d/1XgqthdUTKYrWy7Vh10MceJVt94KN2DbU/view?usp=sharing) |

### 使用方法

#### 安装依赖

```bash
pip install -r requirements.txt
```

#### 数据预处理

1. 从[官方网站](https://mahnob-db.eu/hci-tagging/)下载 MAHNOB-HCI 数据集。
2. 执行预处理：

   ```bash
   python data/preprocessing.py
   ```

   程序会为每位受试者创建一个目录，将预处理后的 CSV 文件保存到 `args.save_path`。

3. 创建经过数据增强和划分的 PyTorch 数据文件：

   ```bash
   python data/create_dataset.py
   ```

   - 该步骤会对预处理数据进行划分和增强。
   - 通过 `label_kind` 选择标签类型：`Arsl` 表示唤醒度，`Vlnc` 表示效价。
   - 最终生成训练所需的 `.pt` 文件。

#### 模型训练

不同模型和任务对应的配置文件如下：

- `configs/h2.yml`：H2
- `configs/cross_attention_h2.yml`：采用跨模态交叉注意力融合的 H2
- `configs/phemonet.yml`：PHemoNet
- `configs/hyperfusenet_arousal.yml`：用于唤醒度任务的 HyperFuseNet
- `configs/hyperfusenet_valence.yml`：用于效价任务的 HyperFuseNet

训练命令：

```bash
python main.py \
  --train_file_path /path/to/arsl_or_vlnc_train.pt \
  --test_file_path /path/to/arsl_or_vlnc_test.pt \
  --config configs/config.yml
```

运行 HyperFuseNet 论文使用的超参数搜索：

```bash
python sweep.py
```

实验将通过 [Weights & Biases](https://wandb.ai/) 进行记录。

### SEED-IV 双模态研究

SEED-IV 使用官方对齐的 4 秒特征窗口：

- EEG 平滑 DE 特征：`[62, 5]`，对应 62 个通道和 5 个频带。
- Eye 特征：`[31]`。
- 标签：neutral、sad、fear、happy。

将数据集放在仓库根目录的 `SEED_IV/`，目录中至少需要：

```text
SEED_IV/
├── eeg_feature_smooth/
│   ├── 1/
│   ├── 2/
│   └── 3/
└── eye_feature_smooth/
    ├── 1/
    ├── 2/
    └── 3/
```

运行 EEG + Eye 拼接基线：

```bash
conda activate mhyeeg
python main_seed_iv.py --config configs/seed_iv_concat.yml
```

可通过命令行运行同一数据划分下的单模态基线：

```bash
python main_seed_iv.py \
  --config configs/seed_iv_concat.yml \
  --model SeedIVEEG \
  --checkpoint_folder checkpoints/seed_iv_eeg_fold0

python main_seed_iv.py \
  --config configs/seed_iv_concat.yml \
  --model SeedIVEye \
  --checkpoint_folder checkpoints/seed_iv_eye_fold0
```

项目采用 5 折受试者独立划分。每折包含 9 名训练受试者、3 名验证受试者和
3 名测试受试者；特征标准化仅使用训练受试者，避免窗口级和受试者级泄漏。
使用 `--fold 0` 至 `--fold 4` 切换折。训练结束后会自动载入验证集
Macro-F1 最优的 checkpoint，并报告未见测试受试者上的 Accuracy 和 Macro-F1。

普通多模态注意力消融：

```bash
python main_seed_iv.py --config configs/seed_iv_attention.yml
```

区间二型模糊可靠性引导注意力：

```bash
python main_seed_iv.py \
  --config configs/seed_iv_it2_fuzzy_attention.yml
```

注意力模型将 EEG 表示为 5 个频带 token，将眼动表示为瞳孔、离散度、注视、
扫视和事件统计 5 个语义 token。完整模型通过区间二型 Gaussian 模糊规则
估计每个 token 的可靠性及 FOU 不确定性，并将可靠性作为 attention logits
的先验。普通注意力配置关闭模糊模块，用于隔离 IT2 模糊引导的贡献。

### 跨模态交叉注意力 H2

`CrossAttentionH2` 保留 H2 的四路编码器和 PHM 分类头，将原有拼接融合替换为模态级交叉注意力。每个查询模态只能关注另外三个模态，不会读取自身特征。

从仓库根目录在 AutoDL 上运行：

```bash
python main.py \
  --config configs/cross_attention_h2.yml \
  --train_file_path /path/to/train_augmented_data_Arsl.pt \
  --test_file_path /path/to/test_data_Arsl.pt
```

模型输入形状：

| 模态 | 输入形状 |
|---|---|
| Eye | `[B, 4, 600]` |
| GSR | `[B, 1, 1280]` |
| EEG | `[B, 10, 1280]` |
| ECG | `[B, 3, 1280]` |

可在 `configs/cross_attention_h2.yml` 中调整以下参数：

| 参数 | 含义 |
|---|---|
| `attention_dim` | 模态特征投影后的公共维度 |
| `attention_heads` | 多头注意力的头数 |
| `attention_layers` | 交叉注意力层数 |
| `attention_dropout` | 注意力分支的 Dropout |

H2 与 CrossAttentionH2 默认使用两阶段精度训练：

- Epoch 1～40：BF16 AMP + TF32，用于提高 RTX 5090 等新显卡的吞吐量。
- Epoch 41～50：自动切换回完整 FP32，用于低学习率阶段的稳定微调。

相关配置：

```yaml
amp: true
amp_dtype: "bfloat16"
fp32_finetune_epochs: 10
allow_tf32: true
cudnn_benchmark: true
deterministic: false
```

如需恢复原始全 FP32、确定性训练，可在命令行覆盖：

```bash
python main.py \
  --config configs/cross_attention_h2.yml \
  --no-amp \
  --no-allow_tf32 \
  --no-cudnn_benchmark \
  --deterministic \
  --fp32_finetune_epochs 0 \
  --train_file_path /path/to/train.pt \
  --test_file_path /path/to/test.pt
```

获取注意力诊断信息：

```python
logits, info = model(eye, gsr, eeg, ecg, return_attention=True)
cross_attention = info["cross_attention"]
pooling_weights = info["pooling_weights"]
```

- `cross_attention` 的形状为 `[layer, batch, query_modality, head, other_modality]`。
- `pooling_weights` 的形状为 `[batch, modality]`。
- 模态顺序固定为 Eye、GSR、EEG、ECG。

运行不依赖数据集的单元测试：

```bash
python -m unittest tests.test_cross_attention
```

### 引用

如果本项目对你的研究有帮助，请引用以下论文 🫶

#### H2

```bibtex
@inproceedings{lopez2024hierarchical,
  title={Hierarchical hypercomplex network for multimodal emotion recognition},
  author={Lopez, Eleonora and Uncini, Aurelio and Comminiello, Danilo},
  booktitle={2024 IEEE 34th International Workshop on Machine Learning for Signal Processing (MLSP)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```

#### PHemoNet

```bibtex
@inproceedings{lopez2024phemonet,
  title={PHemoNet: A Multimodal Network for Physiological Signals},
  author={Lopez, Eleonora and Uncini, Aurelio and Comminiello, Danilo},
  booktitle={2024 IEEE 8th Forum on Research and Technologies for Society and Industry Innovation (RTSI)},
  pages={260--264},
  year={2024},
  organization={IEEE}
}
```

#### HyperFuseNet

```bibtex
@inproceedings{lopez2023hypercomplex,
  title={Hypercomplex Multimodal Emotion Recognition from EEG and Peripheral Physiological Signals},
  author={Lopez, Eleonora and Chiarantano, Eleonora and Grassucci, Eleonora and Comminiello, Danilo},
  booktitle={2023 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

### 更多超复数模型

- *Multi-view hypercomplex learning for breast cancer screening*，TMI 审稿中，2022。[[论文](https://arxiv.org/abs/2204.05798)] [[GitHub](https://github.com/ispamm/PHBreast/)]
- *PHNNs: Lightweight neural networks via parameterized hypercomplex convolutions*，IEEE Transactions on Neural Networks and Learning Systems，2022。[[论文](https://ieeexplore.ieee.org/document/9983846)] [[GitHub](https://github.com/elegan23/hypernets)]
- *Hypercomplex Image-to-Image Translation*，IJCNN，2022。[[论文](https://ieeexplore.ieee.org/document/9892119)] [[GitHub](https://github.com/ispamm/HI2I)]
