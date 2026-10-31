# SEED-IV正式结果统计

主结果为15名测试被试的均值±样本标准差。

| 模型 | Accuracy/% | Macro-F1 |
|---|---:|---:|
| EEG MLP | 38.72 ± 9.20 | 0.3275 ± 0.1282 |
| DGCNN | 40.53 ± 11.64 | 0.3432 ± 0.1439 |
| MAET-Concat | 52.54 ± 11.14 | 0.4753 ± 0.1242 |
| Eye MLP | 65.80 ± 6.24 | 0.6515 ± 0.0685 |
| Concat MLP | 53.74 ± 8.70 | 0.4975 ± 0.1040 |
| Attention | 51.72 ± 9.21 | 0.4800 ± 0.1168 |
| IT2 Fuzzy | 50.78 ± 9.84 | 0.4690 ± 0.1165 |

## 逐被试配对比较

| 指标 | 比较 | A−B均值差 | 95% bootstrap CI | Wilcoxon p | Holm校正p |
|---|---|---:|---:|---:|---:|
| accuracy_pct | IT2 Fuzzy − Eye MLP | -15.0259 | [-21.1657, -8.2607] | 0.001526 | 0.007629 |
| accuracy_pct | IT2 Fuzzy − Concat MLP | -2.9568 | [-9.7968, 4.0267] | 0.3591 | 0.9084 |
| accuracy_pct | IT2 Fuzzy − Attention | -0.9395 | [-4.1251, 2.1505] | 0.7197 | 0.9084 |
| accuracy_pct | IT2 Fuzzy − MAET-Concat | -1.7645 | [-6.8451, 3.6434] | 0.3028 | 0.9084 |
| accuracy_pct | Eye MLP − Concat MLP | 12.0692 | [7.1697, 17.3174] | 0.0001831 | 0.001282 |
| accuracy_pct | Eye MLP − Attention | 14.0865 | [8.2981, 19.5609] | 0.00116 | 0.006958 |
| accuracy_pct | Eye MLP − MAET-Concat | 13.2615 | [6.5308, 19.1805] | 0.004272 | 0.01709 |
| macro_f1 | IT2 Fuzzy − Eye MLP | -0.1824 | [-0.2537, -0.1046] | 0.00116 | 0.006958 |
| macro_f1 | IT2 Fuzzy − Concat MLP | -0.0285 | [-0.1080, 0.0489] | 0.5614 | 1 |
| macro_f1 | IT2 Fuzzy − Attention | -0.0110 | [-0.0478, 0.0260] | 0.6387 | 1 |
| macro_f1 | IT2 Fuzzy − MAET-Concat | -0.0063 | [-0.0607, 0.0539] | 0.3591 | 1 |
| macro_f1 | Eye MLP − Concat MLP | 0.1539 | [0.0925, 0.2175] | 0.0003052 | 0.002136 |
| macro_f1 | Eye MLP − Attention | 0.1715 | [0.1017, 0.2376] | 0.00116 | 0.006958 |
| macro_f1 | Eye MLP − MAET-Concat | 0.1762 | [0.1021, 0.2426] | 0.001526 | 0.006958 |
