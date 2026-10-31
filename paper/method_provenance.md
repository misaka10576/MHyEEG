# 方法与代码来源核查

日期：2026-07-19

## 已确认事实

- `models/seed_iv_fuzzy_attention.py` 由仓库提交
  `3c489d95a65b2ac3af151cb7b81849b9a5476ced` 于2026-07-18加入。
- 原提交信息和README没有记录该模块的方法学参考文献。
- 仓库中没有发现 IT2-GaussFormer 官方代码或同名文件，当前实现也没有声明
  自己是该论文的官方复现。
- 当前实现与 Tang、Tang和Mo（2026）的IT2-GaussFormer在以下设计上高度接近：
  Mean/Std/Energy统计量、区间二型高斯隶属度、模糊区间加权和
  subject-independent评测思想。
- Karnik–Mendel类型约简不是本项目原创，应引用Karnik和Mendel的基础工作。

## 当前项目中的适配

- EEG按5个频带构造成token，而非短时医学信号的多尺度patch。
- 31维眼动特征按瞳孔、离散度、注视、扫视和事件统计构造成5个异构语义token。
- EEG与眼动分别估计模态内可靠性，再将可靠性先验加入10-token联合注意力。
- 同一可靠性同时用于注意力键偏置和模态内加权汇聚。
- 使用SEED-IV四分类和严格13训练/1验证/1测试的15折跨被试协议。

这些内容可以表述为“适配”“构建”和“实验研究”，但在没有更完整的文献与
代码比对前，不宜表述为基础IT2加权方法的原创。

## 论文允许与禁止的表述

允许：

- “参考已有区间二型模糊加权方法，构建面向EEG–眼动token的可靠性引导融合。”
- “研究区间二型可靠性先验在严格跨被试多模态评测中的表现。”
- “本文实现将模态内可靠性同时引入联合注意力和加权汇聚。”

禁止：

- “首次提出Mean/Std/Energy区间二型模糊注意力。”
- “代码完全自主原创”或“与已有方法无关。”
- 在未完成完整对照前写“显著优于”“有效缓解负迁移”或“达到先进水平”。

## 必须引用

1. Tang C, Tang Y K, Mo H. IT2-GaussFormer: An interval type-2 Gaussian fuzzy
   transformer for medical time-series classification. Biomedical Signal
   Processing and Control, 2026, 125: 110804.
   DOI: 10.1016/j.bspc.2026.110804.
2. Karnik N N, Mendel J M. Centroid of a type-2 fuzzy set. Information
   Sciences, 2001, 132(1-4): 195-220.
   DOI: 10.1016/S0020-0255(01)00069-X.
3. Herman P A, Prasad G, McGinnity T M. Designing an interval type-2 fuzzy
   logic system for handling uncertainty effects in BCI classification of
   motor-imagery EEG. IEEE Transactions on Fuzzy Systems, 2017, 25(1): 29-42.
   DOI: 10.1109/TFUZZ.2016.2637934.

