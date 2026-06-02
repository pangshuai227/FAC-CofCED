# Small CofCED vs FaC-CofCED comparison

Date: 2026-06-02

This note records small but more credible comparisons than the initial smoke
run. The purpose is to check whether the current lightweight FaC-CofCED idea
shows a positive trend before running the full 12-hour LIAR-RAW experiment.

## Setup

Common settings:

- Dataset source: `Codes/dataset/LIAR-RAW`
- Balanced subset generator: `scripts/make_smoke_dataset.py`
- Local transformer cache: `COFCED_TRANSFORMERS_LOCAL_ONLY=1`
- Batch size: `1`
- Reports per claim: `8`
- GPU: RTX 4080-class card reported by `nvidia-smi`

The lightweight FaC-CofCED version uses deterministic proxy features:

```text
[alignment, support_proxy, refute_proxy, neutral_proxy, conflict_proxy]
```

These are not yet NLI or sentence-transformer features. Treat this as a
prototype integration result.

## Run 1: SMALL

Subset:

```text
train 60, val 24, test 24
balanced over 6 LIAR-RAW classes
1 epoch
```

Commands:

```bash
OMP_NUM_THREADS=1 \
COFCED_DATA_ROOT=/root/FAC-CofCED/Codes/dataset/LIAR-RAW_SMALL \
COFCED_USE_FAC_FEATURES=0 \
COFCED_TRANSFORMERS_LOCAL_ONLY=1 \
COFCED_N_EPOCHS=1 \
COFCED_BATCH_SIZE=1 \
COFCED_REPORT_EACH_CLAIM=8 \
COFCED_TRAIN_LIMIT=60 \
python3 train_exp_fc5_LIAR_RAW2.py
```

```bash
OMP_NUM_THREADS=1 \
COFCED_DATA_ROOT=/root/FAC-CofCED/Codes/dataset/LIAR-RAW_SMALL \
COFCED_USE_FAC_FEATURES=1 \
COFCED_TRANSFORMERS_LOCAL_ONLY=1 \
COFCED_N_EPOCHS=1 \
COFCED_BATCH_SIZE=1 \
COFCED_REPORT_EACH_CLAIM=8 \
COFCED_TRAIN_LIMIT=60 \
python3 train_exp_fc5_LIAR_RAW2.py
```

Logs:

- CofCED: `/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_204856_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW_SMALL2-all.log`
- FaC-CofCED: `/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_205147_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW_SMALL2-all.log`

Result:

| Model | Val macF1 | Test macF1 | Test ROUGE-1 | Test ROUGE-2 | Test ROUGE-L | Test sent macF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CofCED | 4.76 | 4.76 | 18.92 | 3.25 | 14.61 | 1.49 |
| FaC-CofCED | 4.76 | 4.76 | 18.92 | 3.25 | 14.61 | 1.55 |

Interpretation:

This run was still too small. Both models mostly collapsed to the same class
behavior, so it is not useful for judging improvement.

## Run 2: SMALL180

Subset:

```text
train 180, val 60, test 60
balanced over 6 LIAR-RAW classes
2 epochs
```

Distribution:

```text
train: 30 examples per class
val:   10 examples per class
test:  10 examples per class
```

Commands:

```bash
OMP_NUM_THREADS=1 \
COFCED_DATA_ROOT=/root/FAC-CofCED/Codes/dataset/LIAR-RAW_SMALL180 \
COFCED_USE_FAC_FEATURES=0 \
COFCED_TRANSFORMERS_LOCAL_ONLY=1 \
COFCED_N_EPOCHS=2 \
COFCED_BATCH_SIZE=1 \
COFCED_REPORT_EACH_CLAIM=8 \
COFCED_TRAIN_LIMIT=180 \
python3 train_exp_fc5_LIAR_RAW2.py
```

```bash
OMP_NUM_THREADS=1 \
COFCED_DATA_ROOT=/root/FAC-CofCED/Codes/dataset/LIAR-RAW_SMALL180 \
COFCED_USE_FAC_FEATURES=1 \
COFCED_TRANSFORMERS_LOCAL_ONLY=1 \
COFCED_N_EPOCHS=2 \
COFCED_BATCH_SIZE=1 \
COFCED_REPORT_EACH_CLAIM=8 \
COFCED_TRAIN_LIMIT=180 \
python3 train_exp_fc5_LIAR_RAW2.py
```

Logs:

- CofCED: `/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_205626_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW_SMALL1802-all.log`
- FaC-CofCED: `/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_210027_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW_SMALL1802-all.log`

Result:

| Model | Best val macF1 | Test acc/micF1 | Test macF1 | Test ROUGE-1 | Test ROUGE-2 | Test ROUGE-L | Test sent macF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CofCED | 7.32 | 15.00 | 6.60 | 19.88 | 3.20 | 14.62 | 1.75 |
| FaC-CofCED | 14.24 | 23.33 | 12.37 | 19.17 | 3.21 | 14.25 | 5.37 |

Delta:

| Metric | FaC-CofCED - CofCED |
| --- | ---: |
| Best val macF1 | +6.92 |
| Test acc/micF1 | +8.33 |
| Test macF1 | +5.77 |
| Test ROUGE-1 | -0.71 |
| Test ROUGE-2 | +0.01 |
| Test ROUGE-L | -0.37 |
| Test sent macF1 | +3.62 |

## Preliminary conclusion

On the more credible small balanced run (`SMALL180`), the current lightweight
FaC-CofCED prototype improves veracity classification over CofCED:

```text
test macro-F1: 6.60 -> 12.37
test micF1:    15.00 -> 23.33
```

It also improves sentence evidence macro-F1:

```text
1.75 -> 5.37
```

ROUGE is mixed:

- ROUGE-2 is effectively unchanged/slightly higher.
- ROUGE-1 and ROUGE-L are slightly lower.

This supports the hypothesis that conflict-aware evidence features help the
classification side first, while explanation text overlap needs better feature
quality or decoding/selection calibration.

## Caveats

This is still not a final result:

- one seed only;
- small balanced subset, not full LIAR-RAW;
- only 2 epochs;
- heuristic FAC features, not NLI/embedding features;
- model still predicts only a subset of labels in this small setting.

The result is good enough to justify the next step: replace proxy features with
offline NLI + sentence-transformer features and rerun a larger subset.
