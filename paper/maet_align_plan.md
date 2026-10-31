# MAET-Align 实施方案（v1.0，2026-08-14）

> 总控台：御坂10176号。本文档自包含，执行线程无需其他上下文。
> 目标：在 MAET 底座上做「对比对齐 + 模态缺失训练 + 熵门控」的组合微创新，修复严格跨被试下的多模态负迁移，支持单模态推理。

---

## 0. 背景与目标（先读）

- **数据**：SEED-IV（15 被试 × 3 会话，4 类：neutral/sad/fear/happy）
  - EEG 特征：62 通道 DE（微分熵，5 频段）= 310 维；眼动特征：31 维
  - 本地数据：`/home/misaka/datasets/SEED_IV/{eeg_feature_smooth,eye_feature_smooth}/{session 1..3}/{被试}_{日期}.mat`
  - EEG 特征键 `de_LDS1..24` 形状 [62,T,5]（转置 [T,62,5]）；眼动键 `eye_1..24` 形状 [31,T]；4s 无重叠窗口
- **协议（铁律）**：15 折跨被试 LOSO，13 训练 / 1 验证 / 1 测试被试；归一化只用当前折训练被试统计（防泄漏）；用**验证被试**选 epoch（禁止测试选 epoch）；随机种子 0（主表）+ 1/2（关键组）
- **基线（严格同协议，已锁定，勿重跑）**：
  | 模型 | Acc% | Macro-F1 |
  |---|---|---|
  | Eye MLP | 65.80 ± 6.24 | 0.6515 ± 0.0685 |
  | MAET-Concat | 53.10 ± 11.39 | 0.4896 ± 0.1334 |
  | Concat MLP | 53.74 | 0.4975 |
  | Attention | 51.72 | 0.4800 |
  | EEG MLP / DGCNN | 38.72 / 40.53 | 0.3275 / 0.3432 |
  - 结果文件：`/tmp/maet_15fold_baseline/concat/vanilla/seed_0/loso_summary.json`、`/home/misaka/projects/MHyEEG（修改）/checkpoints/seed_iv_maet/concat/vanilla/seed_0/loso_summary.json`
- **问题**：所有朴素融合（concat/attention/IT2/MAET）均负迁移（< 眼动单模态 65.80%）；诊断证据：融合与眼动逐折 Pearson r≈0.019（零相关）、眼动≥67% 的折融合平均被拖 18pp、融合方差≈眼动 2 倍
- **成功标准（预注册）**：
  1. 主结果：MAET-Align 15 折 Macro-F1 配对检验显著优于 MAET-Concat 基线（Wilcoxon p<0.05）
  2. 正向融合：融合 ≥ Eye MLP 65.80%（达到则论文叙事大升级）
  3. 单模态可用：模型在 eye-only 输入下 ≥ 60%（证明灵活推理）
- **论文定位**：毕业导向，组合式微创新。「首次在 EEG+眼动严格跨被试 LOSO 下，系统组合对比对齐+模态缺失训练+熵门控修复负迁移并支持单模态推理」（窄口径）。**禁止声称**：首次对比学习融合（中文普刊 2025 已发）/ 首次改进 MAET（CMFC-MAET 2026 已发）/ 首次熵门控（AECF arXiv 2505.15417 已发，非 EEG）

---

## 1. 模型架构：MAET-Align

### 1.1 底座：MAET（现有，勿重写）
- 文件：`/home/misaka/projects/MHyEEG（修改）/models/seed_iv_maet.py`（去 timm 依赖适配版）
- 结构：MultiViewEmbedding → 双 Adaptive Transformer（eeg/eye 模态专属）→ 共享 Mixture Transformer（VLMo 血缘，mixffn_start_layer_index=2）→ Fusion（可学习标量 softmax 加权：alpha 2 维，`eeg*alpha0 + eye*alpha1`）→ head
- 已有：`head_eeg`/`head_eye` 辅助头（L307-308）、`eeg_prompt`/`eye_prompt`（prompt_len=2）、`domain_generalization` 开关（ReverseLayerF + domain_classifier，num_domains=19）
- 入口：`run_seed_iv_maet.py --config configs/seed_iv_maet.yml --all_folds --dataset_root /home/misaka/datasets/SEED_IV --cuda`
- 环境：conda `mhyeeg`（`/home/misaka/ENTER/envs/mhyeeg/bin/python`，py310、torch 2.13+cu130、scipy、timm；GPU RTX 4060 Ti 17.2GB，单折 2-4 分钟，15 折 30-40 分钟）

### 1.2 新增组件 1：EFCL 跨模态对比对齐（移植自 CSMM）
- 参照：`/home/misaka/projects/CSMM/modalFusionModels/contrastiveModel_IV.py`（类 `EFCL_seed_final`，loss 部分 L51-90）
- 机制：动量编码器（MAET 双编码器的 EMA 副本，momentum=0.995）+ 特征队列（4096）+ InfoNCE 式跨模态对比
- **正样本定义**：同一 trial（视频编号）的 EEG-眼动对互为正样本；负样本 = 队列中其余
- 公式：
  - `sim_e2f = eeg_embeds @ eye_feat_all / temp`（eye_feat_all = 动量眼动编码器输出 + 队列拼接）
  - `loss_e2f = -Σ log_softmax(sim_e2f, dim=1) * sim_targets`（sim_targets 由 trial idx 构造，正样本对均分）
  - `loss_efcl = (loss_e2f + loss_f2e) / 2`
  - 超参：`temp=0.07`、`queue_size=4096`、`momentum=0.995`、损失权重 `λ1=0.9`
- **接入位置**：双 Adaptive Transformer 输出 → 取 eeg_cls/eye_cls token（或 mean 池化）→ L2 normalize → 对比。**不做 stop-grad on encoder**（对比损失要回传梯度，与 CSMM 一致——CSMM 用动量编码器回传的是在线编码器）
- **trial idx 获取**：MHyEEG 的 loader 需补 trial 索引（SEED-IV 每个 session 24 个视频，用 trail_boundary 划分，参照 CSMM `main_modalFusion.py` L65-69 的 boundary 数组；SESSION_LABELS 已硬编码在 `data/seed_iv.py`）

### 1.3 新增组件 2：模态 Dropout（ModDrop 式缺失训练）
- 机制：训练时每个 batch 以概率 p=0.3 随机将 eeg 或 eye 整支置零（特征 mask），让模型学会单模态下工作
- 实现：训练循环内，`mask = torch.rand(B) < 0.3` 决定丢弃 eeg 或 eye（一次只丢一个模态，不双丢；测试不丢）
- 目的：①提高对模态缺失的鲁棒性（单模态推理能力）②防止模型过度依赖强模态（眼动）而忽略弱模态
- 与 AECF 的 ACM（课程掩码）区别：AECF 是熵驱动的自适应课程掩码（非 EEG），我们是固定概率随机缺失——论文里注明

### 1.4 新增组件 3：熵门控（模态可靠性加权）
- 机制：每个模态分支输出乘可靠性 `r_m = 1 - H(p_m)/log(K)`，K=4 类
- `p_m` 来自模态辅助头（`head_eeg`/`head_eye` 输出 logits 的 softmax）；`H(p_m) = -Σ p_m log p_m`
- **接入位置**：双 Adaptive Transformer 输出 →（辅助头算 r_m）→ 分支特征乘 r_m → 进 Fusion（替代/叠加原标量 alpha）
- **stop-gradient 默认开启**（r_m 不回传梯度，防止门控坍缩），另做「不开启」消融
- 撞车说明：与 CAFNet（TCP 置信度回归）、SwiP-UADF（不确定性融合）、AECF（熵门控）同属「样本级置信度」家族——**不能 claim 首次**，机制差异在论文里写清（我们是归一化预测熵，信息论可解释）

### 1.5 损失函数
`L = L_cls + 0.9 * L_efcl`
- L_cls = 现有分类损失（融合头）
- 模态 dropout 与熵门控不引入额外损失项
- 可选消融：L_efcl 权重 0.3/0.9/1.5

---

## 2. 实验计划（消融表）

| # | 实验 | 说明 | 预期 |
|---|---|---|---|
| 1 | MAET-Concat | 已有基线（53.10/0.4896），勿重跑 | 对照 |
| 2 | MAET + EFCL | 只加对比对齐 | > 基线 |
| 3 | MAET + EFCL + Dropout | 对齐+缺失训练 | > #2 |
| 4 | **MAET-Align 完整**（+熵门控） | 三组件全开 | 最高 |
| 5 | 单模态对照 | Eye MLP 65.80 / EEG MLP 38.72（已有） | 门槛 |
| 6 | 决策级融合 | 推理期三路概率平均（MAET 有 head_eeg/head_eye） | 推理期对照 |
| 7 | 熵门控消融 | ±stop-grad、±辅助头、门控 vs 无门控 | 归因 |
| 8 | 模态缺失测试 | 完整模型在 fusion / eye-only / eeg-only 三种输入下分别测试 | 单模态可用性证明 |

- 主表：seed 0 全 15 折；关键组（#2/#3/#4）加跑 seed 1、2 报告均值±std
- 统计：15 折配对 Wilcoxon（Holm 校正）+ 效应量；逐折差分布图；类别混淆矩阵（37,575 测试窗口池化，口径注明）
- 每折记录：acc / Macro-F1 / 每类 Precision-Recall / 训练历史（早停 epoch）

---

## 3. 实施步骤（执行线程按序）

1. **读代码**：`models/seed_iv_maet.py`（底座）、`data/seed_iv.py`（loader，确认能否取 trial idx）、`run_seed_iv_maet.py`（训练循环，确认 mask/损失挂点）
2. **实现 EFCL 模块**：新文件 `models/contrastive_module.py`（动量编码器副本 + 队列 + loss，参照 CSMM contrastiveModel_IV.py）
3. **实现熵门控**：改 Fusion 前分支（乘 r_m），复用 head_eeg/head_eye
4. **实现模态 dropout**：训练循环 mask
5. **配置**：新 `configs/seed_iv_maet_align.yml`（继承现有 + 新增参数：efcl_weight=0.9、dropout_prob=0.3、temp=0.07、queue_size=4096、momentum=0.995、gate=True、gate_stop_grad=True）
6. **冒烟**：单折（fold 0）跑通，确认 loss 收敛、无 NaN、显存 OK（4060Ti 17.2GB，MAET 仅 0.15M 参数+动量副本，无压力）
7. **跑 #2 单折** → 对比基线同折数字（fold 0 基线 acc 43.87）→ 有提升信号再跑全 15 折
8. **全量**：#2/#3/#4 各 15 折 × seed 0（约 40-90 分钟/组），seed 1/2 关键组
9. **统计+出图**：`paper/analyze_results.py` 扩展配对检验；逐折差图、消融条形图、混淆矩阵热图
10. **汇报**：按五要素（做了什么/到哪步/什么问题/解决什么/用了什么方法）回报总控台

---

## 4. 坑与注意事项（重要）

- **防泄漏铁律**：归一化只 fit 训练折；验证被试选 epoch；测试只评一次；禁止任何测试集统计进训练决策
- **对比正样本**：trial idx 必须与标签对齐（用 trail_boundary 算，别用 session 内顺序假设）
- **动量编码器**：`@torch.no_grad()` 更新 EMA（`param_m = m*param_m + (1-m)*param`）；动量副本 `requires_grad=False`
- **熵门控坍缩防护**：默认 stop-grad；若门控全收敛到 1（失效），检查辅助头是否收敛（熵来源），必要时给辅助头加轻量 CE 监督
- **模态 dropout 与验证**：验证/测试阶段禁用 dropout（训练才丢）
- **确定性**：`torch.backends.cudnn.deterministic=True`、FP32、固定 seed
- **显存**：动量副本+队列极小，4060Ti 无压力；若 OOM 降低 batch_size（现有 32）
- **撞车红线**（写论文时）：必引 CMFC-MAET（JMMB 2026，特征校准路线）、AECF（arXiv 2505.15417，熵门控+对比，非 EEG）、M2S（AAAI 2025，单模态反超）、CSMM（TAFFC 2025，对齐路线）、CMCRD（TAC 2026，蒸馏路线）；标题/摘要禁用「首次」「最优」字样

---

## 5. 参考材料（执行线程可读）

- 底座代码：`/home/misaka/projects/MHyEEG（修改）/models/seed_iv_maet.py`
- EFCL 参照：`/home/misaka/projects/CSMM/modalFusionModels/contrastiveModel_IV.py`（L51-90 loss，L163-189 动量/队列）
- CSMM 数据构建参照：`/home/misaka/projects/CSMM/main_modalFusion.py`（L41-107 preprocess，L65-69 trail_boundary）
- 基线结果：`/tmp/maet_15fold_baseline/concat/vanilla/seed_0/loso_summary.json`
- 统计脚本：`/home/misaka/projects/MHyEEG（修改）/paper/analyze_results.py`
- 论文文献（桌面）：`C:\Users\yvhyn\Desktop\参考文献\`（CSMM/AECF/CMFC-MAET/M2S/OGM-GE/MAAB-ER 等 PDF 全在）

## 6. 时间线（预期）

- 第 1-2 天：读码 + EFCL/门控/dropout 实现 + 冒烟
- 第 3-5 天：单折验证 + #2/#3/#4 全 15 折
- 第 6-7 天：seed 1/2 + 统计 + 出图 + 汇报
