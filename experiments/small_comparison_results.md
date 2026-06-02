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

For FaC-CofCED variants, add:

```bash
COFCED_USE_FAC_FEATURES=1
COFCED_FAC_VERSION=<1|2|3|4>
```

Variant definitions:

- v1: learned FAC selector, learned support/refute/conflict gates, learned FAC classifier.
- v2: deterministic FAC selector boost, deterministic support/refute/conflict pooling, FAC summary classifier plus heuristic LIAR-label prior.
- v3: centered and weaker deterministic selector boost, base CofCED residual, weaker FAC summary residual, weaker label prior.
- v4: v1 learned selector/gates as the main path, plus small FAC summary and label-prior residuals.

Logs:

- CofCED: `/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_205626_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW_SMALL1802-all.log`
- FaC-CofCED v1: `/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_210027_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW_SMALL1802-all.log`
- FaC-CofCED v2: `/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_214223_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW_SMALL1802-all.log`
- FaC-CofCED v3: `/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_215539_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW_SMALL1802-all.log`
- FaC-CofCED v4: `/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_220018_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW_SMALL1802-all.log`

Result:

| Model | Best val macF1 | Test acc/micF1 | Test macF1 | Test ROUGE-1 | Test ROUGE-2 | Test ROUGE-L | Test sent macF1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CofCED | 7.32 | 15.00 | 6.60 | 19.88 | 3.20 | 14.62 | 1.75 |
| FaC-CofCED v1 | 14.24 | 23.33 | 12.37 | 19.17 | 3.21 | 14.25 | 5.37 |
| FaC-CofCED v2 | 13.65 | 13.33 | 5.60 | 19.80 | 3.39 | 14.67 | 4.35 |
| FaC-CofCED v3 | 4.48 | 20.00 | 10.29 | 19.86 | 3.21 | 14.65 | 1.83 |
| FaC-CofCED v4 | 7.50 | 13.33 | 6.57 | 19.79 | 3.19 | 14.62 | 1.98 |

Delta against CofCED:

| Model | Val macF1 | Test micF1 | Test macF1 | Test sent macF1 |
| --- | ---: | ---: | ---: | ---: |
| FaC-CofCED v1 | +6.92 | +8.33 | +5.77 | +3.62 |
| FaC-CofCED v2 | +6.33 | -1.67 | -1.00 | +2.60 |
| FaC-CofCED v3 | -2.84 | +5.00 | +3.69 | +0.08 |
| FaC-CofCED v4 | +0.18 | -1.67 | -0.03 | +0.23 |

## Preliminary conclusion

On the more credible small balanced run (`SMALL180`), FaC-CofCED v1 is still
the best variant tested so far:

```text
CofCED test macro-F1:        6.60
FaC-CofCED v1 test macro-F1: 12.37
```

It also improves sentence evidence macro-F1:

```text
CofCED:        1.75
FaC-CofCED v1: 5.37
```

The more aggressive variants did not improve over v1:

- v2 improved validation macro-F1 but hurt test classification. The deterministic
  pooling and heuristic label prior likely overfit the validation split and
  pushed the model toward a narrow subset of labels.
- v3 recovered some test macro-F1 by weakening and centering the deterministic
  FAC effects, but validation macro-F1 collapsed. It is not a reliable model
  selection target in this small setting.
- v4 preserved the v1 learned selector/gates and added only small FAC-summary
  residuals, but the auxiliary residual still degraded test macro-F1 to nearly
  the CofCED baseline.

ROUGE remains mixed:

- ROUGE-2 is effectively unchanged/slightly higher.
- ROUGE-1 and ROUGE-L are slightly lower for v1, while v2/v3/v4 recover some
  overlap at the cost of classification quality.

This supports a narrower design conclusion: heuristic conflict-aware evidence
features are useful when the model learns how to use them, but hand-coded
stance pooling and LIAR-label priors are too brittle on the current small
subset. The next serious improvement should focus on better FAC feature quality
offline, not stronger heuristic fusion inside the classifier.

## Caveats

This is still not a final result:

- one seed only;
- small balanced subset, not full LIAR-RAW;
- only 2 epochs;
- heuristic FAC features, not NLI/embedding features;
- model still predicts only a subset of labels in this small setting.

The result is good enough to keep v1 as the current working improvement path,
but it is not enough to claim a robust final improvement. The next step should
replace proxy features with offline NLI + sentence-transformer features and
rerun a larger subset before changing the architecture again.
