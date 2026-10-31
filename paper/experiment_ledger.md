# SEED-IV小论文正式实验台账

最后更新：2026-07-19

## 固定协议

- 数据：SEED-IV `de_LDS` EEG `[62,5]` + Eye `[31]`
- 标签：neutral、sad、fear、happy
- 划分：15折，每折13训练被试、1验证被试、1测试被试
- 标准化：仅使用当前折训练被试
- checkpoint：验证被试Macro-F1最优
- 主汇总：15名测试被试指标的均值±样本标准差
- 随机种子：0

## 已完成

| 模型 | Accuracy/% | Macro-F1 | 15折完整 | 结果文件 |
|---|---:|---:|---|---|
| EEG MLP | 38.72 ± 9.20 | 0.3275 ± 0.1282 | 是 | `checkpoints/seed_iv_baselines/mlp/seed_0/loso_summary.json` |
| EEG DGCNN | 40.53 ± 11.64 | 0.3432 ± 0.1439 | 是 | `checkpoints/seed_iv_baselines/dgcnn/seed_0/loso_summary.json` |
| MAET-Concat | 52.54 ± 11.14 | 0.4753 ± 0.1242 | 是 | `checkpoints/seed_iv_maet/concat/vanilla/seed_0/loso_summary.json` |
| Eye MLP | 65.80 ± 6.24 | 0.6515 ± 0.0685 | 是 | `checkpoints/seed_iv_paper/SeedIVEye_seed0/loso_summary.json` |
| Concat MLP | 53.74 ± 8.70 | 0.4975 ± 0.1040 | 是 | `checkpoints/seed_iv_paper/SeedIVConcat_seed0/loso_summary.json` |
| Attention | 51.72 ± 9.21 | 0.4800 ± 0.1168 | 是 | `checkpoints/seed_iv_paper/SeedIVAttention_seed0/loso_summary.json` |
| IT2 Fuzzy Attention | 50.78 ± 9.84 | 0.4690 ± 0.1165 | 是 | `checkpoints/seed_iv_paper/SeedIVIT2FuzzyAttention_seed0/loso_summary.json` |

Eye MLP完整测试共37,575个窗口。其池化逐类结果为：

| 类别 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.7258 | 0.7378 | 0.7317 | 10,170 |
| sad | 0.5937 | 0.5205 | 0.5547 | 10,245 |
| fear | 0.6656 | 0.7314 | 0.6970 | 9,225 |
| happy | 0.6335 | 0.6481 | 0.6408 | 7,935 |

注意：池化Macro-F1为0.6560；论文主表使用逐被试Macro-F1均值0.6515，二者
统计口径不同，不得混写。

## 正在运行

无。

## 等待运行

无。

## Go / No-Go结论

- IT2相对普通Attention的Macro-F1均值差为−0.0110，95% bootstrap CI为
  [−0.0478, 0.0260]，Holm校正后`p=1.0`，没有改善证据。
- Eye MLP显著优于Concat、Attention和MAET-Concat，Macro-F1配对差的Holm
  校正`p`分别为0.0021、0.0070和0.0070。
- 停止扩展或事后调参；论文主线转为严格跨被试条件下的多模态负迁移实证与
  可靠性融合失败分析。

## 结果使用规则

- 只有`complete=true`且测试被试为1至15各一次的结果才进入主表。
- 不使用部分折均值、smoke结果或旧协议结果填补空项。
- `accuracy_pct`和`macro_f1`主表采用逐被试均值；逐类指标和混淆矩阵采用
  37,575个测试窗口的池化统计，并明确统计口径。
