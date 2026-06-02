# CofCED 作为团队 baseline 的复现情况分析

## 结论

本报告中的 **baseline** 按团队当前口径理解为：后续要在其基础上改进的 **CofCED 模型本身**，而不是论文 Table 2 中的 SVM、CNN、RNN、DeClarE、dEFEND、SentHAN、SBERT-FC、GenFE、GenFE-MT 等对比方法。

在这个口径下，结论是：

> 我们已经完成了 CofCED baseline 的一次可运行复现，得到了完整训练/验证/测试日志和 checkpoint；但当前结果没有达到论文中 CofCED 的报告效果，因此还不能称为“严格复现了论文 CofCED baseline”。

因此，当前 CofCED 可以作为团队后续改进的**本地初版 baseline**，但需要在论文数字复现、超参对齐、评估口径确认之后，才能作为更强的、可发表对比意义上的 baseline。

## 当前实验产物

本次结果来自以下日志：

- `/root/CofCED-main/Codes/dataset/logs/2026-05-17_021805_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW2-all.log`

使用的最佳模型为：

- `/root/CofCED-main/Codes/LIAR-RAW_model_epoch6_pre0.264521_rec0.264521_micf0.264521_macf0.263725_rouge0.036134.pt`

本次训练是从已有 checkpoint 续训：

- `/root/CofCED-main/Codes/LIAR-RAW_model_epoch4_pre0.274725_rec0.274725_micf0.274725_macf0.263419_rouge0.035971.pt`

最终测试集日志给出的主结果为：

| 任务 | 指标 | 当前结果 |
|---|---:|---:|
| Veracity classification | Precision | 25.58 |
| Veracity classification | Recall | 25.58 |
| Veracity classification | Micro-F1 | 25.58 |
| Veracity classification | Macro-F1 | 24.89 |
| Explanation ROUGE-F | ROUGE-1 | 15.34 |
| Explanation ROUGE-F | ROUGE-2 | 3.45 |
| Explanation ROUGE-F | ROUGE-L | 11.84 |
| Sentence evidence | Precision | 27.19 |
| Sentence evidence | Recall | 40.45 |
| Sentence evidence | Macro-F1 | 29.25 |

日志中的原始小数为：

- Veracity: `P 0.255795, R 0.255795, micF 0.255795, macF 0.248890`
- Sentence evidence: `P_sent 0.271904, R_sent 0.404523, macF_sent 0.292483`
- ROUGE-F: `15.34 & 3.45 & 11.84`

## 数据一致性检查

当前训练使用的数据位于：

- `/root/CofCED-main/Codes/dataset/LIAR-RAW/train.json`
- `/root/CofCED-main/Codes/dataset/LIAR-RAW/val.json`
- `/root/CofCED-main/Codes/dataset/LIAR-RAW/test.json`

用户提供的解压数据位于：

- `/root/cofced_data_unpack/LIAR-RAW_extracted/LIAR-RAW/train.json`
- `/root/cofced_data_unpack/LIAR-RAW_extracted/LIAR-RAW/val.json`
- `/root/cofced_data_unpack/LIAR-RAW_extracted/LIAR-RAW/test.json`

三份核心 JSON 数据的 SHA-256 一致：

| split | SHA-256 |
|---|---|
| train.json | `0d7ae7183e2201dc898f83a0f8404f119395990e6f483d49df819545f8cc95b3` |
| val.json | `0b519d7c1df8337edb394ad9499a73e10bf735ef666de2a784cd898a8b706f79` |
| test.json | `811f2fd9e88c7e3e68db925b2f4bb4faf5ece1137aab392dd55b53ce46ec0291` |

因此，当前差距不主要来自 `train/val/test.json` 数据文件不一致。

## 与论文结果对比

### LIAR-RAW veracity classification

论文 Table 2 中 LIAR-RAW 的 veracity classification 结果如下。数值单位均为百分比。

| 方法 | Precision | Recall | Macro-F1 |
|---|---:|---:|---:|
| SVM | 15.78 | 15.92 | 15.34 |
| CNN | 22.58 | 22.39 | 21.36 |
| RNN | 24.36 | 21.20 | 20.79 |
| DeClarE | 22.86 | 20.55 | 18.43 |
| dEFEND | 23.09 | 18.56 | 17.51 |
| SentHAN | 22.64 | 19.96 | 18.46 |
| SBERT-FC | 24.09 | 22.07 | 22.19 |
| GenFE | 28.01 | 26.16 | 26.49 |
| GenFE-MT | 18.55 | 19.90 | 15.15 |
| CofCED | 29.48 | 29.55 | 28.93 |
| 当前实验 CofCED | 25.58 | 25.58 | 24.89 |

判断：

- 当前 CofCED 结果高于多数传统 baseline 和早期神经 baseline。
- 当前 CofCED 结果低于论文中的 GenFE baseline：Macro-F1 为 `24.89`，而 GenFE 为 `26.49`。
- 当前 CofCED 结果明显低于论文 CofCED：Macro-F1 为 `24.89`，论文 CofCED 为 `28.93`，差距约 `4.04` 个百分点。

所以，在 veracity classification 上，当前 CofCED baseline 已经有可比较结果，但还没有达到论文 CofCED 主结果，也低于论文中的最强对比方法 GenFE。

### LIAR-RAW explanation generation

论文 Table 4 中 LIAR-RAW 的 explanation generation ROUGE-F 结果如下。

| 方法 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|
| LEAD-N | 9.84 | 0.40 | 7.20 |
| Oracle | 25.50 | 9.28 | 22.61 |
| EXTABS | 18.85 | 3.61 | 12.90 |
| dEFEND | 17.03 | 3.26 | 11.42 |
| GenFE-MT | 23.08 | 3.67 | 12.10 |
| CofCED | 17.14 | 3.49 | 12.96 |
| 当前实验 CofCED | 15.34 | 3.45 | 11.84 |

判断：

- 当前 ROUGE-2 与论文 CofCED 非常接近：`3.45` vs `3.49`。
- 当前 ROUGE-1 低于论文 CofCED：`15.34` vs `17.14`。
- 当前 ROUGE-L 低于论文 CofCED：`11.84` vs `12.96`。
- 当前结果低于 EXTABS、GenFE-MT；与 dEFEND 接近但 ROUGE-1 仍略低。

所以，在 explanation generation 上，当前 CofCED baseline 的 ROUGE-2 接近论文 CofCED，但 ROUGE-1 和 ROUGE-L 仍有差距，不能称为完整复现论文效果。

### Report/document selection 和 sentence evidence

当前日志中还包含 doc classification 和 sentence evidence 指标：

| 指标 | 当前结果 |
|---|---:|
| Doc selection macro Precision | 16.45 |
| Doc selection macro Recall | 73.93 |
| Doc selection macro F1 | 26.90 |
| Sentence evidence Precision | 27.19 |
| Sentence evidence Recall | 40.45 |
| Sentence evidence Macro-F1 | 29.25 |

论文附录 E 中与 LIAR-RAW 相关的报告/句子分类结果包括：

| 表格 | 方法 | Precision | Recall | Macro-F1 |
|---|---|---:|---:|---:|
| Table E.1 report classification | CofCED | 14.98 | 61.06 | 24.06 |
| Table E.2 sentence classification | CofCED | 14.29 | 22.22 | 17.39 |

当前 doc selection 指标看起来高于论文 E.1；sentence evidence 指标也高于论文 E.2。但这里需要谨慎解释：

- 当前代码日志中的 `P_sent/R_sent/F1_sent` 来自当前实现里的 padded tensor、doc selection 后句子预测和多任务评估流程。
- 论文附录中的 sentence/report classification 口径可能对应单独任务设置或不同 oracle/selection 条件。
- 当前推理脚本中还存在全局 `TOP_K=12` 与函数参数 `top_k=4` 并存的问题，解释生成和句子截断实际使用的是全局 `TOP_K=12`。

因此，这部分不能直接作为“已经复现论文附录指标”的证据，只能作为当前 CofCED baseline 实现下的辅助观察。

## 为什么当前 CofCED baseline 没有达到论文效果

### 1. 当前是一次 CofCED baseline 跑通实验，不是严格论文设置复现

当前仓库运行的是 `ExplainFC/CofCED` 相关训练脚本：

- `/root/CofCED-main/Codes/train_exp_fc5_LIAR_RAW2.py`
- `/root/CofCED-main/Codes/model/model_exp_fc5.py`

这符合团队把 CofCED 当作 baseline 的目标。但当前实验更准确地说是“CofCED baseline 初步跑通”，不是“论文 CofCED 数字严格复现”。后者要求训练配置、推理配置、评估口径、checkpoint 恢复方式都和论文保持一致。

### 2. 当前 CofCED 数字低于论文 CofCED

最核心差距在 veracity classification：

- 论文 CofCED Macro-F1：`28.93`
- 当前实验 Macro-F1：`24.89`
- 差距：约 `4.04` 个百分点

解释生成指标也没有完全达到论文 CofCED：

- ROUGE-1：`15.34` vs `17.14`
- ROUGE-2：`3.45` vs `3.49`
- ROUGE-L：`11.84` vs `12.96`

所以，当前 CofCED baseline 的主任务效果明显低于论文报告值。它可以作为后续改进的起点，但如果后续论文/报告要声称“复现 CofCED”，还需要继续对齐实验。

### 3. 训练超参数和论文描述不完全一致

当前脚本关键配置如下：

- `batch_size = 2`
- `learning_rate = 1e-5`
- `n_epochs = 8`
- `REPORT_EACH_CLAIM = 30`
- `TOP_K = 4`

这些来自 `/root/CofCED-main/Codes/train_exp_fc5_LIAR_RAW2.py`。

而论文实验描述中提到的设置包括：

- mini-batch size 为 `1`
- LIAR-RAW 最大选择 reports `K=18`
- LIAR-RAW 最大 oracle sentences 为 `55`

当前训练中的 `batch_size=2`、`TOP_K=4`、`REPORT_EACH_CLAIM=30` 与论文描述不完全一致。这会显著影响 doc selection、sentence selection、解释生成和最终 veracity 分类表现。尤其是 CofCED 是 coarse-to-fine cascaded evidence distillation，前一级 report/document 选择的候选规模会直接影响后一级 sentence evidence 选择；top-k 不一致会改变整个信息流。

### 4. 训练和推理的 TOP_K 口径不一致

训练脚本中 `train_model(..., TOP_K=4)`，但评估脚本 `/root/CofCED-main/Codes/eval_exp_fc5.py` 顶层有：

```python
TOP_K = 12
```

同时 `evaluate_model(..., top_k=4, report_each_claim=12)` 里传入的 `top_k` 参数并没有控制最终生成解释时的截断；生成解释时实际使用的是全局 `TOP_K`：

```python
if len(_pred) == TOP_K:
    break
```

这意味着日志里虽然记录训练参数 `TOP_K=4`，但 ROUGE 生成解释的截断行为很可能按 `TOP_K=12` 执行。该不一致会降低指标可解释性，也会影响与论文表格的直接比较。

更重要的是，CofCED 的 veracity 分类和解释生成是耦合的：模型训练时学习的是某个候选规模和 evidence selection 策略，推理时如果使用另一套截断规则，最终 ROUGE 和部分 evidence 指标就不能直接解释为同一配置下的模型能力。

### 5. 本次训练是从 model checkpoint 续训，但没有恢复 optimizer/scheduler 状态

本次续训使用：

```text
COFCED_RESUME_MODEL=/root/CofCED-main/Codes/LIAR-RAW_model_epoch4_pre0.274725_rec0.274725_micf0.274725_macf0.263419_rouge0.035971.pt
```

当前保存的 `.pt` 是模型对象或模型权重，不包含完整训练状态，例如：

- AdamW optimizer state
- linear scheduler state
- 当前 warmup/decay 进度
- early stop 计数状态

因此，续训时学习率调度从新建 scheduler 开始，不能等价于从 epoch 4 完整恢复训练。这会影响收敛轨迹和最终结果。

从日志看，最终测试使用的 best model 是 epoch6 对应的 checkpoint，而续训起点来自 epoch4。由于 optimizer/scheduler 重置，epoch5 之后的训练并不是原始训练过程的连续延伸，可能出现学习率阶段、Adam 动量估计和权重更新节奏都与从头训练不同的问题。

### 6. 只有一次 run，没有多 seed 或统计显著性

论文通常报告单个最优设置或稳定实验结果，并且 Table 2 还标了统计显著性。当前我们只有一次 seed 为 `100` 的续训结果，没有多随机种子均值/方差，也没有显著性检验。

因此当前结果更适合叫“单次复现实验结果”，不适合叫“完成复现”。

### 7. 公开代码可能不是论文最终实验脚本的完整形态

当前仓库 README 只提供了基本依赖和数据说明，没有提供完整的论文实验命令、所有 baseline 训练脚本、论文表格一键复现流程。代码里也存在不少注释状态的参数，例如：

- `# TOP_K = 12#55`
- `# MAX_ORACLE= 55`
- `#30，12`

这说明公开代码可能需要人工整理参数才能贴近论文最终配置。当前直接运行仓库脚本不能自动保证得到论文表格数字。

### 8. 论文数字可能来自更长训练或不同 early stopping 策略

当前日志显示：

- `n_epochs = 8`
- `EARLY_STOP = 3`
- 最佳验证 macro-F1 出现在 epoch #7 附近
- 最终 test 使用的是验证集上选择的 best model

论文没有在 README 中给出完整可执行命令，因此不能确认论文结果是否来自相同 epoch 数、相同 early stopping patience、相同 checkpoint selection 标准。对于 LIAR-RAW 这种六分类任务，少量 epoch 或 early stopping 策略差异就可能造成几个百分点的 macro-F1 波动。

### 9. 类别不均衡和 macro-F1 对少数类很敏感

当前 test classification report 中，各类 F1 为：

| 类别 | F1 |
|---|---:|
| pants-fire | 20.86 |
| false | 30.50 |
| barely-true | 25.35 |
| half-true | 21.03 |
| mostly-true | 25.16 |
| true | 26.43 |

当前 macro-F1 低主要不是单一类别崩溃，而是多个类别都在 20%-30% 区间。CofCED 论文结果 macro-F1 为 28.93，意味着不仅需要整体 accuracy 提升，还需要多个类别同时提升。超参、seed、doc selection 候选规模、loss 权重和训练连续性都会影响这种多类别平衡。

## 当前结果可以说明什么

当前实验已经完成了以下事情：

1. LIAR-RAW 数据可读取、训练可运行、验证/测试评估可完成。
2. CofCED 模型可以在当前环境中训练并生成 checkpoint。
3. 当前结果与论文表格处于同一量级，不是随机或完全失败的输出。
4. 数据文件与用户提供的解压数据一致，排除了主要数据 split 文件不一致的问题。
5. 当前结果可以作为后续调参和严格复现实验的起点。

但当前实验不能说明：

1. 已经严格复现了论文 CofCED 主结果。
2. 当前 CofCED baseline 已达到论文报告效果。
3. 当前 report/sentence 指标可以直接等价于论文附录指标。
4. 当前训练流程与论文完全一致。

## 建议的下一步 CofCED baseline 固化方案

为了更接近论文结果，建议按以下顺序推进。

### Step 1：固定并显式记录所有实验参数

把以下参数改为可通过命令行或环境变量控制，并在日志中完整记录：

- `batch_size`
- `n_epochs`
- `learning_rate`
- `REPORT_EACH_CLAIM`
- 训练阶段 doc/report top-k
- 推理阶段 explanation sentence top-k
- oracle sentence 上限
- seed
- checkpoint resume 策略
- 是否从头训练

尤其要消除训练脚本 `TOP_K=4` 与评估脚本全局 `TOP_K=12` 的不一致。

### Step 2：按论文设置从头训练 CofCED

建议先跑一组严格接近论文设置的 CofCED：

- 从头训练，不从 epoch4 checkpoint 续训。
- batch size 设为 `1`。
- LIAR-RAW report selection 相关参数对齐论文 `K=18`。
- oracle sentence 上限对齐论文 `55`。
- 保留完整 checkpoint，包括 model、optimizer、scheduler、epoch、best metric。
- 使用同一 test split 做最终评估。

目标是先判断公开代码在论文配置下能否接近：

- Veracity Macro-F1 `28.93`
- ROUGE-F `17.14 / 3.49 / 12.96`

### Step 3：建立团队版 CofCED baseline 卡片

建议把当前结果固化为一个 baseline card，明确写：

- baseline 名称：`CofCED-local-v0`
- 数据：LIAR-RAW，当前 `train/val/test.json` SHA-256 如上
- 训练方式：从 epoch4 checkpoint 续训
- 训练配置：batch size 2、learning rate 1e-5、n_epochs 8、REPORT_EACH_CLAIM 30、训练 TOP_K 4
- 推理配置：当前 eval 全局 TOP_K 12
- 主指标：test macro-F1 24.89，ROUGE-F 15.34/3.45/11.84
- 限制：未严格对齐论文超参，未从头训练，未多 seed

这样团队后续改进时就有一个清楚、诚实、可追踪的起点。

### Step 4：如果需要，也可复现论文对比方法

如果后续目标包括“和论文所有 baseline 横向对齐”，再分别实现或找到以下模型的实验流程：

- SVM
- CNN
- RNN
- DeClarE
- dEFEND
- SentHAN
- SBERT-FC
- GenFE
- GenFE-MT

其中 SVM/CNN/RNN/SBERT-FC 可优先做，因为实现成本较低；DeClarE、dEFEND、SentHAN、GenFE/GenFE-MT 需要确认论文使用的输入、特征和公开实现。

### Step 5：多 seed 重复实验

至少跑 3 个 seed，例如：

- `100`
- `3407`
- `42`

报告均值和标准差。这样才能判断当前差距是训练随机性、超参差异，还是代码/口径差异。

### Step 6：单独核对评估口径

需要将以下指标的计算方式和论文对齐：

- veracity 的 precision/recall/macro-F1 是否为 sklearn macro 口径。
- ROUGE 是否使用相同 tokenizer、stemming、句子拼接和截断方式。
- report classification 是否在相同 gold doc label 上评估。
- sentence classification 是否在相同 selected docs、oracle docs 或全量 docs 上评估。

## 最终判断

当前状态可以总结为：

> 如果 baseline 指团队后续要提升的 CofCED，那么我们已经完成了一个可运行的 CofCED baseline 初版；但它尚未达到论文 CofCED 的报告效果，因此还不是严格论文复现版 baseline。

主要证据是：

- 当前 CofCED 的 LIAR-RAW veracity Macro-F1 为 `24.89`，低于论文 CofCED 的 `28.93`，也低于论文最强 baseline GenFE 的 `26.49`。
- 当前 explanation ROUGE-2 接近论文 CofCED，但 ROUGE-1 和 ROUGE-L 仍低。
- 当前训练/推理参数与论文描述存在关键不一致，特别是 batch size、report/sentence top-k、oracle 上限和续训状态恢复。

因此，当前结果适合作为“本地 CofCED 初步 baseline”。后续如果要把它作为正式实验 baseline，建议先完成参数对齐、从头训练、多 seed 和评估口径清理。
