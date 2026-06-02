# Full LIAR-RAW CofCED vs FaC-CofCED v1 comparison

Date: 2026-06-02

This note tracks the full-data experiment requested for comparing FaC-CofCED
v1 against the original CofCED implementation on `LIAR-RAW`. The full v1 run
has completed, and this document records the final held-out test comparison.

## Purpose

The small balanced experiment suggested that FaC-CofCED v1 was the best current
variant:

| Model | Small180 best val macF1 | Small180 test micF1 | Small180 test macF1 | Small180 sent macF1 |
| --- | ---: | ---: | ---: | ---: |
| CofCED | 7.32 | 15.00 | 6.60 | 1.75 |
| FaC-CofCED v1 | 14.24 | 23.33 | 12.37 | 5.37 |

The goal of this full run is to check whether that positive trend survives on
the complete `LIAR-RAW` split, rather than only on a small balanced subset.

## Common Setup

Both runs use the local `LIAR-RAW` data and the same main training settings:

| Setting | Value |
| --- | --- |
| Dataset | `Codes/dataset/LIAR-RAW` |
| Train split size | 10065 examples |
| Epochs | 8 |
| Batch size | 2 |
| `REPORT_EACH_CLAIM` | 30 |
| Local transformer cache | `COFCED_TRANSFORMERS_LOCAL_ONLY=1` |
| Evaluation target | best validation macro-F1, then held-out test metrics |

## Original CofCED Full Result

Authoritative log:

```text
/root/FAC-CofCED/Codes/dataset/logs/2026-05-17_021805_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW2-all.log
```

Extracted metrics:

| Metric | Value |
| --- | ---: |
| Best validation macF1 | 0.263725 |
| Best validation epoch | 7 |
| Test P | 0.255795 |
| Test R | 0.255795 |
| Test micF1 | 0.255795 |
| Test macF1 | 0.248890 |
| Test ROUGE-1 F | 15.34 |
| Test ROUGE-2 F | 3.45 |
| Test ROUGE-L F | 11.84 |
| Test sentence macF1 | 0.292483 |

This is the baseline that the full FaC-CofCED v1 run must be compared against.

## FaC-CofCED v1 Full Run

Command:

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

Run state:

| Item | Value |
| --- | --- |
| tmux session | `cofced_full_v1` |
| Formal metrics log | `/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_225554_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW2-all.log` |
| Tee run log | `/root/FAC-CofCED/experiments/run_logs/full_v1_20260602_225550.log` |
| FAC enabled | `True` |
| FAC version | `1` |
| Full train split loaded | 10065 examples |
| Final validation epoch completed | 8 |
| Best validation macF1 | 0.270046 |
| Best validation epoch | 7 |
| Best model checkpoint | `/root/FAC-CofCED/Codes/LIAR-RAW_model_epoch6_pre0.279435_rec0.279435_micf0.279435_macf0.270046_rouge0.036304.pt` |
| Final status | completed |

FaC-CofCED v1 means:

- learned FAC selector;
- learned support/refute/conflict representation gates;
- learned FAC classifier branch;
- no deterministic stance-pooling or LIAR-label prior from later variants.

Note: the checkpoint filename contains `epoch6`, while the logger reports the
best validation result as epoch `#7`. This appears to be an internal zero-based
checkpoint naming convention versus one-based log display.

## Final Result Table

| Model | Best val macF1 | Test micF1 | Test macF1 | ROUGE-1 | ROUGE-2 | ROUGE-L | Sent macF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CofCED | 0.263725 | 0.255795 | 0.248890 | 15.34 | 3.45 | 11.84 | 0.292483 |
| FaC-CofCED v1 | 0.270046 | 0.280576 | 0.270895 | 15.44 | 3.47 | 11.91 | 0.266978 |
| Delta | +0.006321 | +0.024781 | +0.022005 | +0.10 | +0.02 | +0.07 | -0.025505 |

## Validation Progress

Across the 8 full-data epochs, the validation blocks reported:

| Metric | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Epoch 5 | Epoch 6 | Epoch 7 | Epoch 8 | Original CofCED best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Best validation macF1 so far | 0.164941 | 0.193474 | 0.229380 | 0.235826 | 0.259630 | 0.264436 | 0.270046 | 0.270046 | 0.263725 |
| Current validation macF1 | 0.164941 | 0.193474 | 0.229380 | 0.235826 | 0.259630 | 0.264436 | 0.270046 | 0.269993 | 0.263725 |
| Validation claim accuracy | 0.239403 | 0.257457 | 0.271586 | 0.281790 | 0.280220 | 0.268446 | 0.279435 | 0.278650 | 0.264521 |
| Validation ROUGE-1 F | 15.49 | 14.47 | 15.20 | 15.46 | 15.44 | 15.48 | 15.58 | 15.69 | 15.52 |
| Validation ROUGE-2 F | 3.52 | 3.39 | 3.52 | 3.62 | 3.55 | 3.62 | 3.63 | 3.65 | 3.61 |
| Validation ROUGE-L F | 11.88 | 11.22 | 11.72 | 11.90 | 11.85 | 11.92 | 11.96 | 12.04 | 11.95 |
| Validation sent F1 | 0.339333 | 0.393711 | 0.367406 | 0.362870 | 0.372049 | 0.387941 | 0.374608 | 0.379496 | not extracted |

The main validation-selection signal is positive. v1 starts below the original
CofCED full-run best validation macro-F1, then steadily improves through epoch
7. Epoch 7 reaches 0.270046 validation macro-F1, above the original CofCED
full-run best validation macro-F1 of 0.263725 by 0.006321. Epoch 8 has slightly
lower current validation macro-F1 at 0.269993, so the selected checkpoint
remains epoch 7. ROUGE continues to increase through epoch 8, reaching 15.69 /
3.65 / 12.04 on the validation set.

## Final Interpretation

The full-data v1 run does improve over the original CofCED baseline on the main
claim-veracity classification metrics:

- validation macro-F1 improves by 0.006321;
- held-out test micro-F1 / accuracy improves by 0.024781;
- held-out test macro-F1 improves by 0.022005;
- held-out test ROUGE-F improves slightly: +0.10 / +0.02 / +0.07.

The improvement is therefore real for classification, but it is not uniform
across every objective. Sentence-level explanation macro-F1 drops from
0.292483 to 0.266978. This means v1 should be described as an improvement for
veracity classification and a slight ROUGE improvement, not as a full
multi-task win.

The likely reason for the classification gain is that v1 gives CofCED explicit
conflict-aware proxy features and lets the model learn how strongly to use
them. Compared with the original CofCED evidence aggregation path, v1 adds a
learned FAC selector, support/refute/conflict gates, and an auxiliary FAC
classifier branch. These extra signals plausibly regularize which evidence
sentences matter for the final truthfulness decision, especially when different
retrieved reports point in different directions.

The likely reason sentence-level F1 falls is that the new FAC path is optimized
for veracity-discriminative evidence rather than exact explanation sentence
alignment. A sentence can help classify the claim through support/refute/conflict
signals without matching the dataset's annotated explanation sentence. In other
words, v1 appears to make the classifier better, but it may shift attention
toward decision-useful evidence rather than annotation-aligned rationale
selection.

## Reproduction Commands

```bash
ps -ef | grep 'train_exp_fc5_LIAR_RAW2' | grep -v grep || true
grep -n "Finish .* epoch\|maximum of f1 value\|results on the test set\|ROUGE-F(1/2/l)\|P_sent:\|claim classification result" \
  /root/FAC-CofCED/Codes/dataset/logs/2026-06-02_225554_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW2-all.log | tail -80
grep -n "maximum of f1 value\|results on the test set\|ROUGE-F(1/2/l)\|P_sent:\|claim classification result" \
  /root/FAC-CofCED/Codes/dataset/logs/2026-05-17_021805_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW2-all.log | tail -80
```
