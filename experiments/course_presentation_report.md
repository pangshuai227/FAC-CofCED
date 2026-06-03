# CofCED 复现与 FaC-CofCED v1 改进实验汇报文档

日期：2026-06-03

## 1. 汇报结论概览

本项目以 COLING 2022 论文 **A Coarse-to-fine Cascaded Evidence-Distillation Neural Network for Explainable Fake News Detection** 中提出的 **CofCED** 作为 baseline。我们的目标不是复现论文中所有对比方法，而是先把 CofCED 在本地跑通，得到一个可比较的原始 baseline，然后借鉴 2025 年 FaithfulRAG 论文中的 fact-level conflict modeling 思想，做一个轻量级的冲突感知改进版本 **FaC-CofCED v1**。

最终结论如下：

1. 我们已经完成了 CofCED baseline 的本地可运行复现，并获得了完整验证集和测试集结果。
2. 我们的原始 CofCED 本地复现结果低于论文报告的 CofCED 数字，因此它更准确地说是“本地 baseline 复现”，还不是严格意义上的“论文数字完全复现”。
3. 我们实现的 FaC-CofCED v1 在完整 LIAR-RAW 数据集上，相比本地原始 CofCED baseline 有明确分类提升。
4. 提升主要体现在 veracity classification，即真假分类任务；sentence-level explanation F1 下降，因此不能说所有任务都全面提升。
5. 目前 v1 是一个轻量原型：它实现了 conflict-aware feature 的接入和学习式融合，但还没有实现完整的 LLM fact extraction、NLI stance classifier 或 sentence-transformer fact alignment。

最重要的完整数据集测试结果：

| 模型 | Test micF1 / Accuracy | Test macF1 | ROUGE-1 | ROUGE-2 | ROUGE-L | Sent macF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 CofCED 本地 baseline | 0.255795 | 0.248890 | 15.34 | 3.45 | 11.84 | 0.292483 |
| FaC-CofCED v1 | 0.280576 | 0.270895 | 15.44 | 3.47 | 11.91 | 0.266978 |
| v1 - CofCED | +0.024781 | +0.022005 | +0.10 | +0.02 | +0.07 | -0.025505 |

因此，课堂汇报中可以概括为：

> 我们没有完全复现论文 CofCED 的原始最高数字，但已经得到一个可运行、可对比的本地 CofCED baseline。在这个 baseline 上，我们加入轻量 conflict-aware 证据建模后，完整 LIAR-RAW 测试集 macro-F1 从 0.248890 提升到 0.270895，提升约 2.20 个百分点。

## 2. 背景：原始 CofCED 是什么

### 2.1 任务定义

CofCED 面向 explainable fake news detection。输入包括：

- 一条 claim；
- 多篇与 claim 相关的 raw reports；
- 每条 claim 的真实性标签；
- 用于解释的 evidence / explanation 信息。

模型需要同时完成：

1. **Report selection**：从多篇 raw reports 中筛选有价值的 reports。
2. **Sentence evidence selection**：从选中 reports 中抽取解释性证据句。
3. **Veracity classification**：预测 claim 的真实性标签。
4. **Explanation generation / extraction evaluation**：根据抽取证据与参考解释计算 ROUGE。

LIAR-RAW 是六分类任务，标签为：

```text
pants-fire, false, barely-true, half-true, mostly-true, true
```

### 2.2 CofCED 的核心思想

CofCED 的名字是 **Coarse-to-fine Cascaded Evidence-Distillation Neural Network**。它的核心思想是“由粗到细地从 raw reports 中蒸馏证据”：

1. 粗粒度：先在 report 层面判断哪些 reports 更重要。
2. 细粒度：再在 sentence 层面选择解释性证据。
3. 最后：把 claim 表征、report 表征、evidence 表征拼接后用于 veracity prediction。

原始 CofCED 的 sentence selector 主要考虑四类信号：

```text
claim relevance + richness + salience - non-redundancy
```

也就是说，它会倾向于选择：

- 和 claim 语义相关的句子；
- 信息量更丰富的句子；
- 在 report 中更显著的句子；
- 与已选证据不重复的句子。

### 2.3 CofCED 的不足

我们分析后认为，CofCED 的关键不足是：它能找“相关”和“显著”的证据，但没有显式区分证据的立场方向。

在 fake news detection 中，raw reports 里可能同时存在：

- 支持 claim 的证据；
- 反驳 claim 的证据；
- 互相冲突的报道；
- 重复转述 claim 但没有验证的信息；
- 部分正确但结论误导的信息。

因此，句子“相关”并不等于句子“可靠”或“对真假判断有帮助”。CofCED 原始结构没有显式建模：

```text
supporting evidence
vs.
refuting evidence
vs.
neutral evidence
vs.
conflicting evidence
```

这就是我们引入 FaithfulRAG 思想的切入点。

## 3. 借鉴的论文思想：FaithfulRAG 对我们的启发

我们参考的第二篇论文是 2025 年的 FaithfulRAG。它的原任务是提升 RAG 在上下文冲突场景下的忠实性。我们没有直接把 CofCED 改成生成式 RAG 模型，而是抽取其中更适合 CofCED 的思想：

> 在事实粒度上进行证据对齐、冲突识别和推理融合。

FaithfulRAG 对我们有三点启发：

1. **Fact-level alignment**：不要只做粗粒度文本相关性，而要考虑 claim 中事实与 evidence 中事实是否对齐。
2. **Conflict modeling**：当不同证据之间互相矛盾时，模型应该显式知道这种冲突。
3. **Reasoning before final answer**：最终判断前应先识别支持、反驳和冲突信息，而不是直接把所有 evidence 混在一起池化。

迁移到 CofCED 后，我们的设计目标是：

> 让 CofCED 不仅会选择相关 evidence，还能学习 evidence 对 claim 的支持、反驳和冲突倾向，从而改善真假分类。

## 4. 我们实现的 FaC-CofCED v1 是什么

### 4.1 方法名称

我们将当前改进版本称为：

```text
FaC-CofCED v1: Fact-level Conflict-aware CofCED
```

中文可以表述为：

> 事实级冲突感知的 CofCED 轻量改进版本。

### 4.2 当前 v1 的实现边界

需要明确的是，当前 v1 是轻量可运行版本，不是完整 FaithfulRAG 复刻。我们目前实现的是：

- 对每个 claim-report sentence 生成 5 维 deterministic proxy features；
- 将这些 features 接入 CofCED 的 sentence selector；
- 将 support/refute/conflict 相关表示接入 veracity classifier；
- 让模型学习如何利用这些 conflict-aware features。

当前还没有实现：

- LLM claim fact extraction；
- OpenIE fact extraction；
- NLI 模型判断 support/refute/neutral；
- sentence-transformer embedding alignment；
- 跨 report 的显式 agreement graph；
- 多 seed 统计显著性实验。

这意味着当前工作是一个“证明方向有效的轻量原型”，后续还有继续提升空间。

### 4.3 v1 的五维 FAC 特征

我们在 `Codes/helpers/reader5.py` 中为每个 claim 和候选 report sentence 生成五维特征：

```text
[alignment, support_proxy, refute_proxy, neutral_proxy, conflict_proxy]
```

含义如下：

| 特征 | 含义 |
| --- | --- |
| `alignment` | claim 和 sentence 的 token overlap 对齐程度 |
| `support_proxy` | sentence 中是否包含支持/验证类 cue，并结合 alignment 得到支持倾向 |
| `refute_proxy` | sentence 中是否包含否定/反驳类 cue，并结合 alignment 得到反驳倾向 |
| `neutral_proxy` | 当前句子缺乏明显支持/反驳信号时的中性程度 |
| `conflict_proxy` | 同时出现支持和反驳 cue，或支持/反驳信号共同存在时的冲突程度 |

这些特征是启发式的，优点是：

- 不需要额外训练 NLI 模型；
- 不需要在线调用 LLM；
- 可以稳定生成并接入完整数据集训练；
- 便于验证“冲突感知信息是否有帮助”。

缺点是：

- cue-based 特征比较粗糙；
- 不能真正理解复杂语义；
- 对讽刺、间接否定、数值矛盾等复杂事实冲突识别能力有限。

### 4.4 v1 对 CofCED 的具体改动

当前主要代码改动涉及：

- `Codes/helpers/reader5.py`
- `Codes/model/model_exp_fc5.py`
- `Codes/train_exp_fc5_LIAR_RAW2.py`
- `Codes/eval_exp_fc5.py`

主要改动如下。

#### 4.4.1 数据读取阶段增加 FAC 特征

在 `reader5.py` 中新增：

- tokenization / stopword 过滤；
- support cue 词表；
- refute cue 词表；
- claim-sentence alignment 计算；
- 五维 FAC 特征生成；
- 将 `fac_features` 加入 batch 字典。

这样每个样本在进入模型时，除了原来的 claim、report、sentence、domain 等信息，还会额外带上每个候选句子的 conflict-aware proxy features。

#### 4.4.2 Sentence selector 增加 FAC selector

原 CofCED 句子选择分数大致是：

```text
total_weights = claim_scores + doc_scores + content_scores - red_scores
```

v1 在启用 `COFCED_USE_FAC_FEATURES=1` 时，额外加入：

```text
fac_scores = Linear(fac_features)
total_weights = total_weights + fac_scores
```

这意味着模型在选择 evidence sentence 时，不再只依赖相关性、显著性、丰富度和非冗余性，还可以学习五维 FAC 特征对 evidence selection 的影响。

#### 4.4.3 Veracity classifier 增加 support/refute/conflict 表示

原始 CofCED 分类器输入可以概括为：

```text
[claim_repr; pooled_sentence_evidence_repr; pooled_report_repr]
```

v1 在选中 evidence 后，使用 FAC 特征学习三个 evidence group representation：

```text
support_repr
refute_repr
conflict_repr
```

然后分类器输入变为：

```text
[claim_repr; report_repr; evidence_repr; support_repr; refute_repr; conflict_repr]
```

这使分类器可以区分“证据支持 claim”和“证据反驳 claim”这两类信息，而不是把所有证据混成一个 evidence vector。

#### 4.4.4 增加环境变量控制，便于做 ablation

训练脚本增加了多个环境变量：

| 环境变量 | 作用 |
| --- | --- |
| `COFCED_USE_FAC_FEATURES` | 是否启用 FaC-CofCED features |
| `COFCED_FAC_VERSION` | 选择 v1/v2/v3/v4 变体 |
| `COFCED_DATA_ROOT` | 指定数据目录 |
| `COFCED_N_EPOCHS` | 指定训练轮数 |
| `COFCED_BATCH_SIZE` | 指定 batch size |
| `COFCED_REPORT_EACH_CLAIM` | 指定每条 claim 使用多少 reports |
| `COFCED_TRAIN_LIMIT` | 小样本实验时限制训练集大小 |
| `COFCED_SKIP_TEST` | 是否跳过 test evaluation |
| `COFCED_TRANSFORMERS_LOCAL_ONLY` | 是否只使用本地 transformers 缓存 |

这些改动让我们可以比较：

- 原始 CofCED：`COFCED_USE_FAC_FEATURES=0`
- FaC-CofCED v1：`COFCED_USE_FAC_FEATURES=1` 且 `COFCED_FAC_VERSION=1`
- 后续 v2/v3/v4 ablation 变体。

#### 4.4.5 修复 classification report 的标签覆盖问题

在 `eval_exp_fc5.py` 中，我们显式传入全部 6 个标签：

```text
labels = list(range(len(LABEL_IDS)))
```

这样即使某次小样本实验没有预测到某个类别，classification report 也能保持六分类口径一致，避免不同实验的 macro-F1 统计维度不一致。

## 5. 数据集与数据规模

### 5.1 数据位置

本次实验使用的数据位于：

```text
/root/FAC-CofCED/Codes/dataset/LIAR-RAW
```

用户之前下载的数据位于：

```text
/root/cofced_data_unpack
```

此前已检查过核心 `train.json`、`val.json`、`test.json` 与解压数据一致，因此当前实验差异不主要来自数据文件不一致。

数据目录大小：

| 路径 | 大小 |
| --- | ---: |
| `/root/FAC-CofCED/Codes/dataset/LIAR-RAW` | 435M |
| `/root/cofced_data_unpack` | 503M |

### 5.2 数据划分

完整 LIAR-RAW 数据划分如下：

| Split | 样本数 |
| --- | ---: |
| Train | 10065 |
| Validation | 1274 |
| Test | 1251 |

标签分布：

| Split | pants-fire | false | barely-true | half-true | mostly-true | true |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 812 | 1958 | 1611 | 2087 | 1950 | 1647 |
| Validation | 115 | 259 | 236 | 244 | 251 | 169 |
| Test | 86 | 249 | 210 | 263 | 238 | 205 |

可以看到，数据存在明显类别不均衡，尤其是 `pants-fire` 类别样本最少。这会影响 macro-F1，也解释了为什么 macro-F1 比 accuracy/micro-F1 更难提升。

## 6. 实验环境与设备

### 6.1 硬件环境

本次 full v1 实验运行在服务器上，硬件信息如下：

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 系列 |
| GPU 显存 | 32760 MiB，约 32GB |
| NVIDIA Driver | 580.105.08 |
| CUDA Driver Version | 13.0 |
| CPU | Intel Xeon Platinum 8352V @ 2.10GHz |
| CPU 核心/线程 | 2 sockets, 32 cores/socket, 2 threads/core，共 128 CPUs |
| 系统内存 | 503 GiB |
| Swap | 0B |

### 6.2 软件环境

| 项目 | 版本 |
| --- | --- |
| Python | 3.8.10 |
| PyTorch | 2.4.1+cu121 |
| PyTorch CUDA | 12.1 |
| Transformers | 4.46.3 |
| 训练设备 | CUDA |

### 6.3 运行方式

由于训练时间较长，我们使用后台/tmux 方式运行，避免 SSH 断开导致训练中断。正式 full v1 命令如下：

```bash
cd /root/FAC-CofCED/Codes
source /etc/network_turbo >/dev/null 2>&1 || true
OMP_NUM_THREADS=1 \
COFCED_DATA_ROOT=/root/FAC-CofCED/Codes/dataset/LIAR-RAW \
COFCED_USE_FAC_FEATURES=1 \
COFCED_FAC_VERSION=1 \
COFCED_TRANSFORMERS_LOCAL_ONLY=1 \
COFCED_N_EPOCHS=8 \
COFCED_BATCH_SIZE=2 \
COFCED_REPORT_EACH_CLAIM=30 \
python3 train_exp_fc5_LIAR_RAW2.py 2>&1 | tee /root/FAC-CofCED/experiments/run_logs/full_v1_$(date +%Y%m%d_%H%M%S).log
```

## 7. 原始 CofCED baseline 的复现情况

### 7.1 原始 CofCED 本地 baseline 日志

我们用于对比的原始 CofCED baseline 日志为：

```text
/root/FAC-CofCED/Codes/dataset/logs/2026-05-17_021805_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW2-all.log
```

该日志对应的是一次从 checkpoint 继续训练的 baseline 实验：

```text
resume_model_url = /root/CofCED-main/Codes/LIAR-RAW_model_epoch4_pre0.274725_rec0.274725_micf0.274725_macf0.263419_rouge0.035971.pt
```

它从显示 epoch 6 继续训练，完成 epoch 6、7、8，并在验证集选择最优 checkpoint 后进行测试。

### 7.2 原始 CofCED 训练配置

| 配置 | 值 |
| --- | --- |
| Epochs | 8 |
| 实际日志从 | epoch 6 续训 |
| Batch size | 2 |
| Learning rate | 1e-5 |
| Report each claim | 30 |
| TOP_K | 4 |
| Early stop | 3 |
| Freeze encoder | false |
| n_tags | 6 |
| 设备 | CUDA |

需要注意：这个 baseline 是本地可运行 baseline，但不完全等同于论文官方最优设置。原因包括：

- 它是从 checkpoint 续训；
- checkpoint 不包含 optimizer/scheduler 完整状态；
- batch size、report 数、TOP_K 等配置与论文描述可能不完全一致；
- 只有单 seed 结果；
- 公开代码中的训练/评估口径存在一些需要人工确认的细节。

### 7.3 原始 CofCED 复现耗时

这份正式 baseline 日志的时间为：

```text
开始：2026-05-17 02:18:05
结束：2026-05-17 04:50:51
耗时：2小时32分46秒
```

因为它是从 epoch 6 续训，所以这个耗时不能代表从零开始完整 8 轮训练的耗时。

### 7.4 原始 CofCED 复现结果

验证集最佳结果：

| 指标 | 值 |
| --- | ---: |
| Best validation macF1 | 0.263725 |
| Best validation epoch | 7 |

测试集结果：

| 指标 | 值 |
| --- | ---: |
| Test P | 0.255795 |
| Test R | 0.255795 |
| Test micF1 / Accuracy | 0.255795 |
| Test macF1 | 0.248890 |
| ROUGE-1 F | 15.34 |
| ROUGE-2 F | 3.45 |
| ROUGE-L F | 11.84 |
| Sentence evidence P | 0.271904 |
| Sentence evidence R | 0.404523 |
| Sentence evidence macF1 | 0.292483 |

测试集六分类明细：

| 类别 | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| pants-fire | 0.220779 | 0.197674 | 0.208589 | 86 |
| false | 0.268293 | 0.353414 | 0.305026 | 249 |
| barely-true | 0.245536 | 0.261905 | 0.253456 | 210 |
| half-true | 0.241379 | 0.186312 | 0.210300 | 263 |
| mostly-true | 0.260090 | 0.243697 | 0.251627 | 238 |
| true | 0.270408 | 0.258537 | 0.264339 | 205 |

### 7.5 与论文 CofCED 数字的差距

论文中 CofCED 在 LIAR-RAW veracity classification 上报告的 Macro-F1 为约 `28.93%`。我们的本地 CofCED baseline 为：

```text
24.89%
```

差距约：

```text
28.93 - 24.89 = 4.04 个百分点
```

在 explanation ROUGE 上：

| 指标 | 论文 CofCED | 本地 CofCED |
| --- | ---: | ---: |
| ROUGE-1 | 17.14 | 15.34 |
| ROUGE-2 | 3.49 | 3.45 |
| ROUGE-L | 12.96 | 11.84 |

ROUGE-2 接近论文结果，但 ROUGE-1 和 ROUGE-L 仍有差距。因此我们不能声称“完全复现了论文 CofCED 数字”，更严谨的说法是：

> 我们完成了 CofCED 本地 baseline 的可运行复现，但复现结果低于论文报告值。

## 8. FaC-CofCED v1 完整数据集实验

### 8.1 v1 日志与 checkpoint

正式 full v1 日志：

```text
/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_225554_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW2-all.log
```

tee 日志：

```text
/root/FAC-CofCED/experiments/run_logs/full_v1_20260602_225550.log
```

最佳 checkpoint：

```text
/root/FAC-CofCED/Codes/LIAR-RAW_model_epoch6_pre0.279435_rec0.279435_micf0.279435_macf0.270046_rouge0.036304.pt
```

说明：checkpoint 文件名中是 `epoch6`，日志显示 best validation epoch 是 `#7`。这大概率是内部保存文件名使用 zero-based epoch index，而日志展示使用 one-based epoch number。

### 8.2 v1 训练配置

| 配置 | 值 |
| --- | --- |
| Dataset | `/root/FAC-CofCED/Codes/dataset/LIAR-RAW` |
| Train split | 10065 |
| Validation split | 1274 |
| Test split | 1251 |
| Epochs | 8 |
| Batch size | 2 |
| Learning rate | 1e-5 |
| Report each claim | 30 |
| TOP_K | 4 |
| FAC enabled | True |
| FAC version | 1 |
| Local transformer cache | True |
| 设备 | CUDA |

与原始 baseline 对比，v1 的主要差异是启用了：

```text
COFCED_USE_FAC_FEATURES=1
COFCED_FAC_VERSION=1
```

### 8.3 v1 复现耗时

v1 full run 的时间为：

```text
开始：2026-06-02 22:55:54
结束：2026-06-03 06:14:21
耗时：7小时18分27秒
```

这次 v1 是完整 full dataset 8 轮训练并完成 test evaluation，因此这个耗时可以作为当前服务器上完整 v1 实验的参考时间。

### 8.4 v1 验证集过程

v1 在 8 个 epoch 上的验证集结果如下：

| Metric | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Epoch 6 | Epoch 7 | Epoch 8 | 原始 CofCED best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best validation macF1 so far | 0.164941 | 0.193474 | 0.229380 | 0.235826 | 0.259630 | 0.264436 | 0.270046 | 0.270046 | 0.263725 |
| Current validation macF1 | 0.164941 | 0.193474 | 0.229380 | 0.235826 | 0.259630 | 0.264436 | 0.270046 | 0.269993 | 0.263725 |
| Validation accuracy | 0.239403 | 0.257457 | 0.271586 | 0.281790 | 0.280220 | 0.268446 | 0.279435 | 0.278650 | 0.264521 |
| ROUGE-1 F | 15.49 | 14.47 | 15.20 | 15.46 | 15.44 | 15.48 | 15.58 | 15.69 | 15.52 |
| ROUGE-2 F | 3.52 | 3.39 | 3.52 | 3.62 | 3.55 | 3.62 | 3.63 | 3.65 | 3.61 |
| ROUGE-L F | 11.88 | 11.22 | 11.72 | 11.90 | 11.85 | 11.92 | 11.96 | 12.04 | 11.95 |
| Sentence F1 | 0.339333 | 0.393711 | 0.367406 | 0.362870 | 0.372049 | 0.387941 | 0.374608 | 0.379496 | not extracted |

可以看到：

- v1 从 epoch 1 到 epoch 7 验证集 macro-F1 持续上升；
- epoch 7 达到最佳 validation macro-F1 `0.270046`；
- epoch 8 当前 macro-F1 略降到 `0.269993`，所以最终选择 epoch 7 checkpoint；
- v1 最佳 validation macro-F1 高于原始 CofCED baseline 的 `0.263725`。

### 8.5 v1 测试集结果

v1 最终测试集结果：

| 指标 | 值 |
| --- | ---: |
| Test P | 0.280576 |
| Test R | 0.280576 |
| Test micF1 / Accuracy | 0.280576 |
| Test macF1 | 0.270895 |
| ROUGE-1 F | 15.44 |
| ROUGE-2 F | 3.47 |
| ROUGE-L F | 11.91 |
| Sentence evidence P | 0.247319 |
| Sentence evidence R | 0.367397 |
| Sentence evidence macF1 | 0.266978 |

测试集六分类明细：

| 类别 | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| pants-fire | 0.312500 | 0.174419 | 0.223881 | 86 |
| false | 0.302752 | 0.265060 | 0.282655 | 249 |
| barely-true | 0.270115 | 0.223810 | 0.244792 | 210 |
| half-true | 0.252874 | 0.334601 | 0.288052 | 263 |
| mostly-true | 0.301418 | 0.357143 | 0.326923 | 238 |
| true | 0.276243 | 0.243902 | 0.259067 | 205 |

相比原始 CofCED，v1 在多数类别的 F1 上有提升：

| 类别 | CofCED F1 | v1 F1 | 差值 |
| --- | ---: | ---: | ---: |
| pants-fire | 0.208589 | 0.223881 | +0.015292 |
| false | 0.305026 | 0.282655 | -0.022371 |
| barely-true | 0.253456 | 0.244792 | -0.008664 |
| half-true | 0.210300 | 0.288052 | +0.077752 |
| mostly-true | 0.251627 | 0.326923 | +0.075296 |
| true | 0.264339 | 0.259067 | -0.005272 |

最大的提升来自：

- `half-true`
- `mostly-true`
- `pants-fire`

这说明 v1 的改进可能更有利于中间真实性标签和强假标签的区分，但对 `false`、`barely-true`、`true` 的提升并不稳定。

## 9. 原始 CofCED vs FaC-CofCED v1 对比分析

### 9.1 主结果对比

| 模型 | Best val macF1 | Test micF1 | Test macF1 | ROUGE-1 | ROUGE-2 | ROUGE-L | Sent macF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始 CofCED | 0.263725 | 0.255795 | 0.248890 | 15.34 | 3.45 | 11.84 | 0.292483 |
| FaC-CofCED v1 | 0.270046 | 0.280576 | 0.270895 | 15.44 | 3.47 | 11.91 | 0.266978 |
| 差值 | +0.006321 | +0.024781 | +0.022005 | +0.10 | +0.02 | +0.07 | -0.025505 |

### 9.2 分类任务是否提升

结论：提升明确。

```text
Test micF1: 0.255795 -> 0.280576, 提升 0.024781
Test macF1: 0.248890 -> 0.270895, 提升 0.022005
```

换成百分制表达：

```text
Micro-F1 / Accuracy: 25.58% -> 28.06%，提升 2.48 个百分点
Macro-F1: 24.89% -> 27.09%，提升 2.20 个百分点
```

对 fake news 六分类任务来说，macro-F1 的提升更重要，因为数据不均衡，macro-F1 更能反映模型对少数类的处理能力。

### 9.3 ROUGE 是否提升

结论：轻微提升。

```text
ROUGE-1: 15.34 -> 15.44
ROUGE-2: 3.45 -> 3.47
ROUGE-L: 11.84 -> 11.91
```

ROUGE 提升幅度很小，因此不宜夸大。但至少说明 v1 没有明显破坏生成/抽取解释的文本重合质量。

### 9.4 Sentence-level explanation 是否提升

结论：没有提升，反而下降。

```text
Sent macF1: 0.292483 -> 0.266978
```

这说明 v1 更擅长帮助最终真假分类，但不一定更擅长选择与人工标注 explanation 完全一致的句子。

可能原因是：

- FAC 特征强调 support/refute/conflict 的判别作用；
- 对 veracity classification 有帮助的句子，不一定是数据集中人工 explanation 标注的句子；
- 分类目标和解释句选择目标存在一定张力；
- 当前 FAC 特征是启发式 proxy，可能改变了 sentence selector 的偏好。

因此课堂汇报中应谨慎表述：

> v1 是分类任务上的有效改进，但还不是解释选择任务上的全面改进。

## 10. 为什么 v1 会提升

我们认为 v1 提升的主要原因有三点。

### 10.1 原始 CofCED 缺少证据立场方向

原 CofCED 会选择相关、显著、不冗余的证据，但没有显式告诉模型一句话是在支持 claim 还是反驳 claim。对于真假分类来说，这个方向信息很关键。

例如：

```text
Claim: A politician voted for a bill.
Sentence A: The record confirms he voted for the bill.
Sentence B: The official record shows he did not vote for the bill.
```

两句话都可能和 claim 高度相关，但对真实性判断的方向完全相反。v1 的 support/refute/conflict proxy features 能给模型额外提示。

### 10.2 v1 让 evidence aggregation 更结构化

原始 CofCED 把 selected evidence 聚合成一个整体表示：

```text
h_evidence
```

v1 进一步构造：

```text
h_support, h_refute, h_conflict
```

这让最终分类器可以学习：

- 支持证据强时更可能预测真实类；
- 反驳证据强时更可能预测虚假类；
- 冲突证据强时可能对应中间标签或不确定标签。

这种结构化 evidence 表示与 LIAR-RAW 的六分类任务更匹配，因为 LIAR 标签不是简单二分类，而是从 `pants-fire` 到 `true` 的有序真实性程度。

### 10.3 FAC 特征起到了轻量正则化作用

当前数据中 raw reports 噪声较多。只依赖语义相关性，模型容易选到重复转述 claim 或高相关但低判别力的句子。

FAC 特征提供了额外归纳偏置：

- 对齐 claim 的句子更重要；
- 带有 refute cue 的句子可能对假新闻识别更重要；
- 同时有支持/反驳线索的句子可能代表冲突证据；
- neutral 句子可能降低选择优先级。

这些先验不一定完美，但它们让模型更容易从 noisy reports 中找到对分类有用的信号。

## 11. 为什么 sentence-level F1 会下降

v1 的问题也很明显：sentence-level explanation F1 降低。

我们认为原因包括：

### 11.1 分类有用证据不等于人工 explanation 标注

模型可能选择了对最终分类有帮助的句子，但这些句子不一定和数据集 reference explanation 的句子完全一致。这样分类会提升，但 sentence-level F1 会下降。

### 11.2 FAC proxy 特征改变了 selector 偏好

原始 selector 更偏向相关性、显著性和非冗余。v1 加入 FAC selector 后，模型可能更偏向带有支持/反驳 cue 的句子，从而牺牲一部分与 reference explanation 的重合。

### 11.3 启发式特征存在噪声

当前 support/refute/conflict 是 cue-based proxy，不是真正 NLI。比如 `not`、`false` 等词有时可能出现在引用、否定范围或无关上下文中，导致 stance 判断不准。

### 11.4 多任务目标存在冲突

CofCED 同时优化 report selection、sentence selection 和 veracity classification。加入 FAC 后，分类任务获益，但 sentence selection 任务可能被分类判别信号牵引，导致解释句选择指标下降。

## 12. v1、v2、v3、v4 小规模实验回顾

在跑 full v1 之前，我们做过 `SMALL180` 小规模平衡子集实验：

```text
train 180, val 60, test 60
每个类别 train 30 条，val/test 各 10 条
训练 2 epochs
```

结果如下：

| Model | Best val macF1 | Test acc/micF1 | Test macF1 | Test ROUGE-1 | Test ROUGE-2 | Test ROUGE-L | Test sent macF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CofCED | 7.32 | 15.00 | 6.60 | 19.88 | 3.20 | 14.62 | 1.75 |
| FaC-CofCED v1 | 14.24 | 23.33 | 12.37 | 19.17 | 3.21 | 14.25 | 5.37 |
| FaC-CofCED v2 | 13.65 | 13.33 | 5.60 | 19.80 | 3.39 | 14.67 | 4.35 |
| FaC-CofCED v3 | 4.48 | 20.00 | 10.29 | 19.86 | 3.21 | 14.65 | 1.83 |
| FaC-CofCED v4 | 7.50 | 13.33 | 6.57 | 19.79 | 3.19 | 14.62 | 1.98 |

小规模实验说明：

- v1 是最稳的版本；
- v3 也比原始 CofCED 好，但不如 v1；
- v2/v4 的启发式 fusion 更强，反而更容易过拟合或破坏分类；
- 因此最终选择 v1 跑完整数据集。

这也说明了一个设计判断：

> conflict-aware features 是有用的，但最好让模型学习怎么用，而不是用过强的手写规则直接支配分类结果。

## 13. 当前复现和改进工作的局限

### 13.1 原始 CofCED 不是论文数字完全复现

我们的本地 CofCED baseline 低于论文 CofCED。主要可能原因：

- 训练配置和论文设置不完全一致；
- 使用了 checkpoint 续训；
- 没有恢复 optimizer/scheduler 完整状态；
- 公开代码中的 TOP_K、REPORT_EACH_CLAIM 等设置需要进一步核对；
- 只有单 seed；
- 评估口径可能和论文表格存在细节差异。

### 13.2 v1 只跑了一次 full run

目前 full LIAR-RAW 上 v1 和原始 CofCED 都是单次结果。严格论文实验应继续补充：

- 3 到 5 个随机种子；
- 平均值和标准差；
- 显著性检验；
- 更严格的同配置从零训练 baseline。

### 13.3 当前 FAC 特征仍然比较粗糙

目前的 FAC 特征是 deterministic proxy，不是真正事实级推理。后续更强版本可以替换为：

- NLI 模型输出的 support/refute/neutral 概率；
- sentence-transformer 的 claim-sentence alignment；
- OpenIE/LLM 抽取的 atomic facts；
- 跨 report agreement/conflict graph；
- 对 LIAR 六分类标签顺序更敏感的 ordinal classifier。

### 13.4 explanation 指标下降需要进一步解决

如果后续目标不仅是提升 classification，还要提升 explainability，则需要额外设计：

- explanation-aware FAC loss；
- sentence-level rationale alignment loss；
- 保留原 selector 与 FAC selector 的双路融合；
- 对 FAC features 做更高质量 stance 标注或蒸馏；
- 分类提升和解释质量之间的权重调节。

## 14. 后续改进方向

### 14.1 用 NLI 替换 cue-based stance proxy

当前 `support_proxy/refute_proxy/neutral_proxy` 是基于关键词和 overlap 的启发式特征。更合理的做法是使用 NLI 模型：

```text
input: [claim, sentence]
output: entailment / contradiction / neutral
```

再映射为：

```text
support = entailment
refute = contradiction
neutral = neutral
```

这会更接近 FaithfulRAG 中的 fact-level conflict modeling。

### 14.2 加入 sentence-transformer alignment

当前 alignment 是 token overlap。后续可以替换为 embedding cosine similarity：

```text
alignment = cosine(emb(claim_fact), emb(sentence_fact))
```

这样可以识别词面不同但语义相近的证据。

### 14.3 做 claim fact extraction

目前我们把 claim 整体作为对齐目标。后续可以把 claim 拆成多个 atomic facts：

```text
claim -> fact1, fact2, fact3
```

然后分别与 report sentence 对齐，得到更细粒度的 support/refute/conflict。

### 14.4 显式建模跨 report 冲突

目前 v1 主要是 claim-sentence 层面的特征。后续可以建图：

```text
sentence_i --supports/conflicts--> sentence_j
```

然后用 graph attention 或 agreement pooling 建模多个 reports 之间的一致性和冲突。

### 14.5 同时优化分类和解释

为了避免 sentence-level F1 下降，可以考虑：

- 保留原始 evidence selector 的权重；
- 增加 explanation consistency loss；
- 把 FAC selector 作为辅助分支，而不是直接改变主 selector；
- 对解释句选择和分类任务分别设置不同的融合比例。

## 15. 课堂汇报建议结构

建议课堂汇报按下面顺序展开：

1. **任务背景**
   - Fake news detection 为什么需要 explainability。
   - LIAR-RAW 输入是 claim + raw reports。

2. **Baseline: CofCED**
   - Coarse-to-fine evidence distillation。
   - Report selection、sentence selection、veracity prediction。
   - 原始 selector 的四类信号。

3. **复现情况**
   - 数据规模：10065 / 1274 / 1251。
   - 设备：RTX 4080 32GB、128 CPUs、503GiB 内存。
   - 原始 CofCED 本地复现结果：macF1 0.248890。
   - 与论文 CofCED 数字仍有差距。

4. **问题分析**
   - CofCED 缺少 support/refute/conflict 显式建模。
   - 相关证据不等于可信证据。

5. **借鉴 FaithfulRAG**
   - Fact-level alignment。
   - Conflict modeling。
   - Reasoning before final decision。

6. **我们的方法 FaC-CofCED v1**
   - 五维 FAC 特征。
   - FAC selector。
   - support/refute/conflict evidence representation。
   - 环境变量控制 ablation。

7. **实验结果**
   - 小规模实验选择 v1。
   - full LIAR-RAW 上 v1 macro-F1 从 0.248890 到 0.270895。
   - ROUGE 小幅提升。
   - Sent macF1 下降。

8. **结果分析**
   - 为什么分类提升。
   - 为什么解释句 F1 下降。
   - 当前方法边界。

9. **未来工作**
   - NLI stance。
   - sentence-transformer alignment。
   - atomic fact extraction。
   - cross-report conflict graph。
   - 多 seed 实验。

## 16. 可引用的日志与文档

原始 CofCED baseline 日志：

```text
/root/FAC-CofCED/Codes/dataset/logs/2026-05-17_021805_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW2-all.log
```

FaC-CofCED v1 full run 日志：

```text
/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_225554_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW2-all.log
```

FaC-CofCED v1 tee 日志：

```text
/root/FAC-CofCED/experiments/run_logs/full_v1_20260602_225550.log
```

完整 v1 对比文档：

```text
/root/FAC-CofCED/experiments/full_v1_comparison_results.md
```

小规模 variant 对比文档：

```text
/root/FAC-CofCED/experiments/small_comparison_results.md
```

baseline 复现分析：

```text
/root/FAC-CofCED/baseline_reproduction_analysis.md
```

FaithfulRAG 与 CofCED 对齐分析：

```text
/root/FAC-CofCED/alignment_memo.md
```

## 17. 最终汇报用一句话总结

> 我们以 CofCED 为本地 baseline，在完整 LIAR-RAW 上复现得到 24.89% macro-F1；借鉴 FaithfulRAG 的事实级冲突建模思想，加入轻量 support/refute/conflict-aware evidence modeling 后，FaC-CofCED v1 将测试集 macro-F1 提升到 27.09%，说明冲突感知证据建模对真假分类有效，但当前解释句选择指标下降，后续需要用更强的 NLI/fact-level alignment 和 explanation-aware training 继续改进。
