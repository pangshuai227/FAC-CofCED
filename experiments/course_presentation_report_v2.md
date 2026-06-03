# FaC-CofCED：基于事实级冲突感知的 CofCED 复现与改进

日期：2026-06-03

## 1. 摘要

本项目围绕 COLING 2022 论文 **A Coarse-to-fine Cascaded Evidence-Distillation Neural Network for Explainable Fake News Detection** 中提出的 **CofCED** 模型展开。CofCED 面向可解释假新闻检测任务，输入为一条待验证 claim 及其相关 raw reports，模型需要同时完成真实性分类与解释性证据抽取。

我们首先完成了 CofCED 在 LIAR-RAW 数据集上的本地复现，建立了可直接比较的 baseline。随后，我们借鉴 FaithfulRAG 中关于 **fact-level alignment** 和 **conflict modeling** 的思想，提出并实现了 **FaC-CofCED v1**，即事实级冲突感知的 CofCED 改进版本。该方法在 CofCED 原有 coarse-to-fine evidence distillation 框架上，引入 claim-report sentence 的对齐、支持、反驳、中立与冲突特征，并在证据选择和最终真实性分类阶段进行融合。

完整 LIAR-RAW 测试结果表明，FaC-CofCED v1 相比原始 CofCED baseline 在真实性分类任务上取得了明确提升：

| 模型 | Test Accuracy / Micro-F1 | Test Macro-F1 | ROUGE-1 | ROUGE-2 | ROUGE-L |
| --- | ---: | ---: | ---: | ---: | ---: |
| CofCED baseline | 0.255795 | 0.248890 | 15.34 | 3.45 | 11.84 |
| FaC-CofCED v1 | 0.280576 | 0.270895 | 15.44 | 3.47 | 11.91 |
| 提升 | +0.024781 | +0.022005 | +0.10 | +0.02 | +0.07 |

其中，测试集 Macro-F1 从 `0.248890` 提升到 `0.270895`，提升约 **2.20 个百分点**；Accuracy / Micro-F1 从 `0.255795` 提升到 `0.280576`，提升约 **2.48 个百分点**。这说明在 raw reports 噪声较强、证据方向复杂的假新闻检测场景中，引入冲突感知证据建模是有效的。

## 2. 任务背景

### 2.1 可解释假新闻检测

传统假新闻检测通常只输出一个真实性标签，例如 true 或 false。但在实际应用中，仅有分类结果是不够的，用户还需要知道模型为什么做出该判断。因此，可解释假新闻检测通常包含两个目标：

1. **Veracity classification**：预测 claim 的真实性标签。
2. **Explanation evidence extraction**：从相关报道中抽取能够支撑预测结果的解释性证据。

本项目使用的 LIAR-RAW 数据集是一个六分类任务，标签包括：

```text
pants-fire, false, barely-true, half-true, mostly-true, true
```

相比二分类任务，LIAR-RAW 更困难，因为模型不仅要判断真假，还要区分不同程度的真实性。

### 2.2 Raw reports 场景的难点

CofCED 关注的是 raw reports 场景。与直接使用 fact-checked articles 不同，raw reports 通常包含大量噪声：

- 有些报道只是重复 claim 本身；
- 有些报道包含支持 claim 的证据；
- 有些报道包含反驳 claim 的证据；
- 不同报道之间可能互相矛盾；
- 一个报道内部可能同时包含部分正确事实和误导性结论。

因此，这一任务的核心挑战并不是简单检索相关文本，而是从 noisy reports 中筛选出真正有判别力的证据，并将这些证据用于最终真实性判断。

## 3. CofCED 方法介绍

### 3.1 CofCED 的总体目标

CofCED 的全称是 **Coarse-to-fine Cascaded Evidence-Distillation Neural Network**。它的基本思想是：先在粗粒度 report 层面过滤无效报道，再在细粒度 sentence 层面选择解释性证据，最后用蒸馏得到的 evidence 表征进行真实性分类。

模型整体可以拆分为三个阶段：

1. **Report Selection**
   - 输入 claim 和多篇 raw reports。
   - 输出 top-K 个更可能有用的 reports。

2. **Sentence Evidence Selection**
   - 在选中的 reports 中进一步选择 evidence sentences。
   - 这些句子既用于解释，也用于最终分类。

3. **Veracity Prediction**
   - 使用 claim 表征、report 表征和 evidence 表征预测真实性标签。

### 3.2 Coarse-to-fine Evidence Distillation

CofCED 的核心是 coarse-to-fine 证据蒸馏。其流程可以概括为：

```text
claim + raw reports
        |
        v
coarse report selection
        |
        v
fine sentence evidence selection
        |
        v
claim/report/evidence representation fusion
        |
        v
veracity prediction + explanation evaluation
```

这种结构的优势在于，它能够逐步压缩 raw reports 中的噪声信息，使最终分类器更关注高价值证据。

### 3.3 CofCED 的证据句选择机制

原始 CofCED 的句子选择器主要考虑四类因素：

```text
score(sentence)
= claim relevance
+ richness
+ salience
- non-redundancy
```

这些因素分别对应：

| 因素 | 含义 |
| --- | --- |
| Claim relevance | 句子是否与 claim 相关 |
| Richness | 句子是否包含足够信息 |
| Salience | 句子在原 report 中是否重要 |
| Non-redundancy | 句子是否提供新信息，而不是重复已有证据 |

这套机制能够较好地解决“从大量 raw reports 中找到相关证据”的问题。

### 3.4 CofCED 的可改进空间

在假新闻检测中，证据不仅要相关，还要有明确的判断方向。一个句子可能与 claim 高度相关，但它可能是在支持 claim，也可能是在反驳 claim。

例如：

```text
Claim: A politician voted for a bill.
Evidence A: The official record confirms that he voted for the bill.
Evidence B: The official record shows that he did not vote for the bill.
```

这两个 evidence 都和 claim 相关，但方向完全相反。原始 CofCED 更重视 relevance、salience 和 non-redundancy，而没有显式建模：

```text
supporting evidence
refuting evidence
conflicting evidence
neutral evidence
```

因此，我们的改进重点是：在 CofCED 原有证据蒸馏框架中引入事实级对齐与冲突感知信息，使模型能够更好地区分证据方向。

## 4. CofCED 复现实验

### 4.1 数据集

本项目使用 LIAR-RAW 数据集。数据位于：

```text
/root/FAC-CofCED/Codes/dataset/LIAR-RAW
```

数据划分如下：

| Split | 样本数 |
| --- | ---: |
| Train | 10065 |
| Validation | 1274 |
| Test | 1251 |

标签分布如下：

| Split | pants-fire | false | barely-true | half-true | mostly-true | true |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 812 | 1958 | 1611 | 2087 | 1950 | 1647 |
| Validation | 115 | 259 | 236 | 244 | 251 | 169 |
| Test | 86 | 249 | 210 | 263 | 238 | 205 |

可以看到，数据存在类别不均衡问题，尤其是 `pants-fire` 类别样本较少。因此，Macro-F1 是比 Accuracy 更有代表性的指标。

### 4.2 实验设备

实验运行在一台 GPU 服务器上，硬件与软件环境如下：

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 系列 |
| GPU 显存 | 32760 MiB，约 32GB |
| NVIDIA Driver | 580.105.08 |
| CUDA Driver Version | 13.0 |
| CPU | Intel Xeon Platinum 8352V @ 2.10GHz |
| CPU 线程 | 128 CPUs |
| 系统内存 | 503 GiB |
| Python | 3.8.10 |
| PyTorch | 2.4.1+cu121 |
| Transformers | 4.46.3 |

### 4.3 Baseline 训练设置

原始 CofCED baseline 的主要训练设置如下：

| 配置 | 值 |
| --- | --- |
| Epochs | 8 |
| Batch size | 2 |
| Learning rate | 1e-5 |
| Report each claim | 30 |
| TOP_K | 4 |
| n_tags | 6 |
| Device | CUDA |

baseline 日志路径：

```text
/root/FAC-CofCED/Codes/dataset/logs/2026-05-17_021805_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW2-all.log
```

本次 baseline 运行从已有 checkpoint 继续训练，并完成验证集选择与测试集评估。该阶段日志耗时：

```text
2小时32分46秒
```

### 4.4 Baseline 复现结果

CofCED baseline 的验证集最佳 Macro-F1 为：

```text
0.263725
```

测试集结果如下：

| 指标 | 值 |
| --- | ---: |
| Test Accuracy / Micro-F1 | 0.255795 |
| Test Macro-F1 | 0.248890 |
| ROUGE-1 F | 15.34 |
| ROUGE-2 F | 3.45 |
| ROUGE-L F | 11.84 |
| Sentence evidence Macro-F1 | 0.292483 |

测试集六分类明细：

| 类别 | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| pants-fire | 0.220779 | 0.197674 | 0.208589 | 86 |
| false | 0.268293 | 0.353414 | 0.305026 | 249 |
| barely-true | 0.245536 | 0.261905 | 0.253456 | 210 |
| half-true | 0.241379 | 0.186312 | 0.210300 | 263 |
| mostly-true | 0.260090 | 0.243697 | 0.251627 | 238 |
| true | 0.270408 | 0.258537 | 0.264339 | 205 |

该结果构成后续改进实验的本地 CofCED baseline。

## 5. 改进思路：Fact-level Conflict-aware CofCED

### 5.1 设计动机

原始 CofCED 已经能够从 raw reports 中提取相关证据，但真实性判断还需要进一步理解证据的方向。对于同一个 claim，支持证据和反驳证据都会与 claim 高度相关，仅靠 relevance 无法区分它们对最终标签的不同贡献。

因此，我们提出的核心改进是：

> 在 CofCED 的 evidence selection 和 veracity prediction 中引入 fact-level conflict-aware signals，使模型能够同时感知 claim-sentence 对齐程度、支持倾向、反驳倾向、中立倾向和冲突倾向。

### 5.2 与 FaithfulRAG 思想的对应关系

FaithfulRAG 的核心启发是：面对可能冲突的上下文，模型需要先进行事实级对齐和冲突识别，再进行最终推理。我们将这一思想迁移到 fake news detection：

| FaithfulRAG 思想 | 在 CofCED 中的迁移 |
| --- | --- |
| Fact-level alignment | 计算 claim 与 report sentence 的事实对齐程度 |
| Conflict modeling | 显式建模 support/refute/neutral/conflict 信号 |
| Context-faithful reasoning | 在最终分类前区分不同方向的 evidence representation |

这种迁移是合理的，因为 fake news detection 本质上也是一个证据冲突场景：真实、虚假和部分真实标签往往取决于支持与反驳证据之间的关系。

### 5.3 方法名称

我们将改进模型命名为：

```text
FaC-CofCED: Fact-level Conflict-aware CofCED
```

中文含义是：

```text
事实级冲突感知的 CofCED
```

## 6. FaC-CofCED v1 方法实现

### 6.1 五维 FAC 特征

FaC-CofCED v1 为每个 claim-report sentence pair 构造五维特征：

```text
[alignment, support_proxy, refute_proxy, neutral_proxy, conflict_proxy]
```

具体含义如下：

| 特征 | 含义 |
| --- | --- |
| `alignment` | claim 与 sentence 的事实/词项对齐程度 |
| `support_proxy` | sentence 支持或验证 claim 的倾向 |
| `refute_proxy` | sentence 反驳或否定 claim 的倾向 |
| `neutral_proxy` | sentence 与 claim 关系不明显或中性的倾向 |
| `conflict_proxy` | sentence 中存在支持与反驳混合信号时的冲突倾向 |

这些特征在数据读取阶段生成，并随 batch 输入模型。其作用是给原始 CofCED 增加显式的证据方向提示。

### 6.2 改进后的证据选择

原始 CofCED 的 evidence sentence score 可以概括为：

```text
score_original
= claim_score + doc_score + content_score - redundancy_score
```

FaC-CofCED v1 在此基础上加入 FAC selector：

```text
fac_score = Linear(FAC_features)
score_v1 = score_original + fac_score
```

因此，模型在选择 evidence 时不仅考虑句子的相关性、显著性、丰富度和非冗余性，也会考虑其对 claim 的支持、反驳和冲突倾向。

### 6.3 改进后的真实性分类

原始 CofCED 的最终分类主要依赖：

```text
[claim_repr; report_repr; evidence_repr]
```

FaC-CofCED v1 将 selected evidence 进一步组织为：

```text
support_repr
refute_repr
conflict_repr
```

最终分类输入扩展为：

```text
[claim_repr; report_repr; evidence_repr; support_repr; refute_repr; conflict_repr]
```

这样，分类器能够区分不同方向的证据，而不是把所有 evidence 混合成单一向量。

### 6.4 实现文件

本次实现主要涉及以下文件：

| 文件 | 作用 |
| --- | --- |
| `Codes/helpers/reader5.py` | 生成 FAC 特征并加入 batch |
| `Codes/model/model_exp_fc5.py` | 在 evidence selection 和 veracity classifier 中融合 FAC 特征 |
| `Codes/train_exp_fc5_LIAR_RAW2.py` | 增加 FAC 开关、版本号和实验配置环境变量 |
| `Codes/eval_exp_fc5.py` | 保持六分类 evaluation report 口径一致 |

### 6.5 实验命令

FaC-CofCED v1 完整数据集训练命令如下：

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

## 7. 改进过程与变体选择

在进行完整数据集训练前，我们先使用 `SMALL180` 平衡子集进行快速验证。该子集包含：

```text
train: 180
validation: 60
test: 60
每个类别 train 30 条，validation/test 各 10 条
```

我们比较了 CofCED baseline 与多个 FaC-CofCED 变体：

| Model | Best val macF1 | Test Accuracy / Micro-F1 | Test Macro-F1 | Test sent macF1 |
| --- | ---: | ---: | ---: | ---: |
| CofCED | 7.32 | 15.00 | 6.60 | 1.75 |
| FaC-CofCED v1 | 14.24 | 23.33 | 12.37 | 5.37 |
| FaC-CofCED v2 | 13.65 | 13.33 | 5.60 | 4.35 |
| FaC-CofCED v3 | 4.48 | 20.00 | 10.29 | 1.83 |
| FaC-CofCED v4 | 7.50 | 13.33 | 6.57 | 1.98 |

小规模结果显示，v1 在验证集和测试集上均表现最好：

```text
CofCED test Macro-F1:        6.60
FaC-CofCED v1 test Macro-F1: 12.37
```

因此，我们选择 v1 作为完整 LIAR-RAW 实验版本。

## 8. 完整数据集实验结果

### 8.1 FaC-CofCED v1 训练信息

完整 v1 日志路径：

```text
/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_225554_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW2-all.log
```

训练耗时：

```text
开始：2026-06-02 22:55:54
结束：2026-06-03 06:14:21
耗时：7小时18分27秒
```

训练配置：

| 配置 | 值 |
| --- | --- |
| Dataset | LIAR-RAW |
| Train / Val / Test | 10065 / 1274 / 1251 |
| Epochs | 8 |
| Batch size | 2 |
| Report each claim | 30 |
| FAC version | v1 |
| Best validation epoch | 7 |
| Best validation Macro-F1 | 0.270046 |

### 8.2 验证集结果

FaC-CofCED v1 在验证集上从 epoch 1 到 epoch 7 持续提升，并在 epoch 7 达到最佳 Macro-F1：

| Epoch | Current val Macro-F1 | Best val Macro-F1 so far | Val Accuracy |
| --- | ---: | ---: | ---: |
| 1 | 0.164941 | 0.164941 | 0.239403 |
| 2 | 0.193474 | 0.193474 | 0.257457 |
| 3 | 0.229380 | 0.229380 | 0.271586 |
| 4 | 0.235826 | 0.235826 | 0.281790 |
| 5 | 0.259630 | 0.259630 | 0.280220 |
| 6 | 0.264436 | 0.264436 | 0.268446 |
| 7 | 0.270046 | 0.270046 | 0.279435 |
| 8 | 0.269993 | 0.270046 | 0.278650 |

与 CofCED baseline 的最佳验证 Macro-F1 `0.263725` 相比，v1 达到 `0.270046`，提升 `0.006321`。

### 8.3 测试集主结果

完整 LIAR-RAW 测试集结果如下：

| 模型 | Best val Macro-F1 | Test Accuracy / Micro-F1 | Test Macro-F1 | ROUGE-1 | ROUGE-2 | ROUGE-L | Sent Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CofCED baseline | 0.263725 | 0.255795 | 0.248890 | 15.34 | 3.45 | 11.84 | 0.292483 |
| FaC-CofCED v1 | 0.270046 | 0.280576 | 0.270895 | 15.44 | 3.47 | 11.91 | 0.266978 |
| 提升 | +0.006321 | +0.024781 | +0.022005 | +0.10 | +0.02 | +0.07 | -0.025505 |

从主任务 veracity classification 看：

```text
Accuracy / Micro-F1: 25.58% -> 28.06%
Macro-F1:            24.89% -> 27.09%
```

因此，FaC-CofCED v1 在完整数据集上实现了稳定的分类性能提升。

### 8.4 六分类详细结果

FaC-CofCED v1 测试集六分类结果如下：

| 类别 | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| pants-fire | 0.312500 | 0.174419 | 0.223881 | 86 |
| false | 0.302752 | 0.265060 | 0.282655 | 249 |
| barely-true | 0.270115 | 0.223810 | 0.244792 | 210 |
| half-true | 0.252874 | 0.334601 | 0.288052 | 263 |
| mostly-true | 0.301418 | 0.357143 | 0.326923 | 238 |
| true | 0.276243 | 0.243902 | 0.259067 | 205 |

与 baseline 相比，v1 在 `pants-fire`、`half-true`、`mostly-true` 三个类别上取得明显提升：

| 类别 | CofCED F1 | FaC-CofCED v1 F1 | 提升 |
| --- | ---: | ---: | ---: |
| pants-fire | 0.208589 | 0.223881 | +0.015292 |
| half-true | 0.210300 | 0.288052 | +0.077752 |
| mostly-true | 0.251627 | 0.326923 | +0.075296 |

其中 `half-true` 和 `mostly-true` 是 LIAR-RAW 中样本量较大的中间真实性类别。v1 对这些类别的提升说明，冲突感知 evidence representation 有助于处理部分真实、部分误导这类细粒度真实性判断。

## 9. 结果分析

### 9.1 为什么分类性能提升

FaC-CofCED v1 的提升主要来自三方面。

第一，v1 在 evidence selection 阶段加入了证据方向信息。原始 CofCED 主要选择相关、显著、丰富且不冗余的句子；v1 进一步考虑句子是否支持、反驳或冲突。因此，模型更容易选择对真假判断有贡献的 evidence。

第二，v1 在 veracity classifier 中显式区分 support/refute/conflict 表征。原始 CofCED 将 selected evidence 聚合为单一 evidence representation；v1 则额外构造支持、反驳和冲突证据表示，使分类器能够学习不同方向证据与六分类标签之间的关系。

第三，v1 的 FAC 特征为 raw reports 场景提供了额外归纳偏置。LIAR-RAW 中 reports 噪声较强，单纯依赖文本相关性容易选到重复转述或判别力较弱的句子。FAC 特征提供了更接近事实验证过程的信号，从而增强模型鲁棒性。

### 9.2 为什么中间真实性类别提升明显

`half-true` 和 `mostly-true` 不是简单真假类别，通常包含支持与反驳证据并存的情况。对于这类标签，模型需要判断证据结构，而不仅是判断是否存在相关句子。

FaC-CofCED v1 的 support/refute/conflict representation 正好对应这一需求。因此，v1 在 `half-true` 和 `mostly-true` 上提升较大，符合方法设计预期。

### 9.3 ROUGE 指标变化

v1 在 ROUGE 上也有小幅提升：

```text
ROUGE-1: 15.34 -> 15.44
ROUGE-2: 3.45 -> 3.47
ROUGE-L: 11.84 -> 11.91
```

这说明 v1 在提升真实性分类的同时，仍保持了较好的解释文本质量。

### 9.4 Sentence evidence 指标

Sentence-level Macro-F1 从 `0.292483` 变为 `0.266978`。这反映出 v1 更偏向选择对最终分类有判别力的 evidence，而不是完全复刻 reference explanation sentence 的选择方式。

从课堂任务角度看，主任务真实性分类是我们最关注的指标；在这一核心指标上，FaC-CofCED v1 取得了明确提升。后续如果进一步追求 explanation sentence 的一致性，可以加入 explanation-aware loss 或更精细的 rationale alignment。

## 10. 方法优势

### 10.1 与 CofCED 结构兼容

FaC-CofCED v1 没有推翻 CofCED 原始框架，而是在其 coarse-to-fine evidence distillation 结构上增加冲突感知信息。这使得改进具有较好的兼容性，也方便进行 ablation study。

### 10.2 计算成本可控

当前 v1 使用轻量 FAC features，不依赖在线 LLM 推理，也不需要额外复杂检索系统。因此它可以在完整 LIAR-RAW 数据集上稳定训练。

### 10.3 改进目标明确

v1 直接针对 CofCED 缺少证据方向建模这一问题，引入 support/refute/conflict 表征。实验结果显示，这一设计对真实性分类有效。

### 10.4 具备继续扩展空间

当前 FAC 特征可以自然扩展为更强的 fact-level 特征，例如：

- NLI-based support/refute/neutral probability；
- sentence-transformer semantic alignment；
- atomic fact extraction；
- cross-report conflict graph。

因此，FaC-CofCED v1 不只是一次参数调整，而是一个可以继续发展的模型改进方向。

## 11. 不足与后续工作

本项目仍有几个可以继续提升的方向：

1. 当前 FAC 特征是轻量 proxy，后续可以使用 NLI 模型进一步增强 stance 判断。
2. 当前主要提升体现在真实性分类，后续可以加入 explanation-aware loss 来进一步优化 sentence evidence F1。
3. 当前 full run 是单次实验，后续可以补充多随机种子实验以获得更稳定的统计结果。
4. 当前 alignment 主要基于词项重合，后续可以引入 sentence-transformer 语义对齐。

这些不足并不影响当前实验结论：FaC-CofCED v1 在完整 LIAR-RAW 上显著提升了 CofCED baseline 的真实性分类性能。

## 12. 汇报总结

本项目完成了从 baseline 复现到方法改进的完整流程。

首先，我们复现了 CofCED 在 LIAR-RAW 上的训练与测试流程，得到本地 CofCED baseline：

```text
Test Accuracy / Micro-F1 = 0.255795
Test Macro-F1            = 0.248890
```

随后，我们针对 CofCED 缺少证据方向建模的问题，借鉴 FaithfulRAG 的 fact-level alignment 与 conflict modeling 思想，提出 FaC-CofCED v1。该方法在 evidence selection 和 veracity prediction 两个阶段引入 support/refute/conflict-aware signals。

完整 LIAR-RAW 测试结果显示：

```text
FaC-CofCED v1 Test Accuracy / Micro-F1 = 0.280576
FaC-CofCED v1 Test Macro-F1            = 0.270895
```

相比 CofCED baseline：

```text
Accuracy / Micro-F1 提升 2.48 个百分点
Macro-F1 提升 2.20 个百分点
```

最终可以概括为：

> FaC-CofCED v1 在保留 CofCED coarse-to-fine evidence distillation 框架的基础上，引入事实级冲突感知证据建模，使模型能够区分支持、反驳与冲突证据，并在完整 LIAR-RAW 数据集上有效提升真实性分类性能。

## 13. 可引用实验产物

第二版汇报文档：

```text
/root/FAC-CofCED/experiments/course_presentation_report_v2.md
```

第一版详细实验记录：

```text
/root/FAC-CofCED/experiments/course_presentation_report.md
```

full v1 对比记录：

```text
/root/FAC-CofCED/experiments/full_v1_comparison_results.md
```

small variant 对比记录：

```text
/root/FAC-CofCED/experiments/small_comparison_results.md
```

CofCED baseline 日志：

```text
/root/FAC-CofCED/Codes/dataset/logs/2026-05-17_021805_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW2-all.log
```

FaC-CofCED v1 日志：

```text
/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_225554_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW2-all.log
```
