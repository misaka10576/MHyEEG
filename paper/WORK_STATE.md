# 当前工作状态（压缩版）

更新时间：2026-07-19

## 唯一目标

完成SEED-IV严格13训练/1验证/1测试的15折实验，并形成可交导师的中文初稿。

## 已完成正式结果

- EEG MLP：38.72±9.20%，Macro-F1 0.3275±0.1282。
- EEG DGCNN：40.53±11.64%，Macro-F1 0.3432±0.1439。
- MAET-Concat：52.54±11.14%，Macro-F1 0.4753±0.1242。
- Eye MLP：65.80±6.24%，Macro-F1 0.6515±0.0685。
- Concat MLP：53.74±8.70%，Macro-F1 0.4975±0.1040。
- Attention：51.72±9.21%，Macro-F1 0.4800±0.1168。
- IT2 Fuzzy Attention：50.78±9.84%，Macro-F1 0.4690±0.1165。

所有以上结果均为15折完整、测试被试1至15各一次。

## Go / No-Go结论

- IT2没有优于Attention：Macro-F1差−0.0110，95% CI跨0，Holm校正`p=1.0`。
- Eye-only显著优于Concat、Attention和MAET-Concat。
- 不再扩模型或按测试结果调参。论文转为“跨被试多模态负迁移：统一评测与
  可靠性融合分析”。

## 论文与代码

- 中文初稿：`paper/seed_iv_it2_draft.md`
- 实验台账：`paper/experiment_ledger.md`
- 来源核查：`paper/method_provenance.md`
- 统一评估：`evaluate_seed_iv_paper.py`
- 统计检验：`paper/analyze_results.py`
- 出图：`paper/generate_figures.py`
- 单元测试：27项通过。

## 来源边界

Mean/Std/Energy + 区间二型Gaussian加权与Tang等2026年的IT2-GaussFormer
高度相关，必须引用 DOI `10.1016/j.bspc.2026.110804`。Karnik–Mendel类型
约简必须引用 DOI `10.1016/S0020-0255(01)00069-X`。不得写首创或完全自主原创。

## 后续规则

1. 只从summary读取真实结果，不从训练日志或记忆抄数字。
2. 只有`complete=true`才填论文。
3. 不再扩模型；IT2完成后执行配对Wilcoxon、出图、填初稿、同步Obsidian。
4. 训练监控只读取完成折数，不展开epoch日志。
