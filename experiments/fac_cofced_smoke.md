# FaC-CofCED smoke experiment notes

Owner GitHub username: `pangshuai227`

## Current method

This workspace keeps CofCED as the team baseline and adds an optional lightweight
FaC-CofCED path. It is inspired by FaithfulRAG's fact-level conflict modeling,
but adapts the conflict target from parametric-vs-context conflicts to
claim/report and cross-evidence conflicts.

The current implementation is intentionally lightweight:

- `reader5.py` generates deterministic sentence-level FAC features:
  `[alignment, support_proxy, refute_proxy, neutral_proxy, conflict_proxy]`.
- `model_exp_fc5.py` uses `COFCED_USE_FAC_FEATURES=1` to:
  - add FAC scalar scores into the sentence selector;
  - pool selected evidence into support/refute/conflict representations;
  - classify with `[claim, reports, evidence, support, refute, conflict]`.
- Default behavior remains original CofCED when `COFCED_USE_FAC_FEATURES` is not set.

This is not yet the final NLI/embedding version. The proxy features are a
low-risk first step to verify integration and run small experiments without
downloading additional models.

## Smoke dataset

Create a tiny LIAR-RAW subset:

```bash
python3 scripts/make_smoke_dataset.py --train 4 --val 2 --test 2
```

The smoke dataset is ignored by git and can be recreated.

## Smoke run

Run a short FAC-enabled training pass:

```bash
cd Codes
COFCED_DATA_ROOT=/root/FAC-CofCED/Codes/dataset/LIAR-RAW_SMOKE \
COFCED_USE_FAC_FEATURES=1 \
COFCED_TRANSFORMERS_LOCAL_ONLY=1 \
COFCED_N_EPOCHS=1 \
COFCED_BATCH_SIZE=1 \
COFCED_REPORT_EACH_CLAIM=4 \
COFCED_TRAIN_LIMIT=2 \
COFCED_SKIP_TEST=1 \
python3 train_exp_fc5_LIAR_RAW2.py
```

For a direct baseline smoke run, change `COFCED_USE_FAC_FEATURES=0`.

## Verified smoke results

Both commands below completed on 2026-06-02. The metrics are not meaningful
because the run uses only 2 training examples and 2 validation examples; this is
only an integration check.

FAC-enabled smoke:

```bash
COFCED_DATA_ROOT=/root/FAC-CofCED/Codes/dataset/LIAR-RAW_SMOKE \
COFCED_USE_FAC_FEATURES=1 \
COFCED_TRANSFORMERS_LOCAL_ONLY=1 \
COFCED_N_EPOCHS=1 \
COFCED_BATCH_SIZE=1 \
COFCED_REPORT_EACH_CLAIM=4 \
COFCED_TRAIN_LIMIT=2 \
COFCED_SKIP_TEST=1 \
python3 train_exp_fc5_LIAR_RAW2.py
```

Log:

```text
/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_200607_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW_SMOKE2-all.log
```

Observed validation output:

```text
FaC-CofCED features enabled: True
Finish 1 epoch, Loss: 54.671200
ROUGE-F(1/2/l): 23.56 & 8.60 & 19.81
COFCED_SKIP_TEST=1; skip test evaluation.
```

Original CofCED smoke:

```bash
COFCED_DATA_ROOT=/root/FAC-CofCED/Codes/dataset/LIAR-RAW_SMOKE \
COFCED_USE_FAC_FEATURES=0 \
COFCED_TRANSFORMERS_LOCAL_ONLY=1 \
COFCED_N_EPOCHS=1 \
COFCED_BATCH_SIZE=1 \
COFCED_REPORT_EACH_CLAIM=4 \
COFCED_TRAIN_LIMIT=2 \
COFCED_SKIP_TEST=1 \
python3 train_exp_fc5_LIAR_RAW2.py
```

Log:

```text
/root/FAC-CofCED/Codes/dataset/logs/2026-06-02_200656_DOC_ExplainFC5-DistilBERT_auto_LIAR-RAW_SMOKE2-all.log
```

Observed validation output:

```text
FaC-CofCED features enabled: False
Finish 1 epoch, Loss: 54.883649
ROUGE-F(1/2/l): 22.00 & 8.11 & 18.74
COFCED_SKIP_TEST=1; skip test evaluation.
```

Notes:

- `COFCED_TRANSFORMERS_LOCAL_ONLY=1` avoids HuggingFace network retries and uses
  the local DistilBERT cache.
- The eval script now passes explicit class labels into `classification_report`,
  so tiny validation subsets with missing classes do not crash.
- The environment prints `libgomp: Invalid value for environment variable
  OMP_NUM_THREADS`; the smoke run still completes. Set `OMP_NUM_THREADS=1` if
  you want to silence it.

## Next implementation step

Replace the heuristic FAC feature generator with offline features:

1. sentence embedding alignment, e.g. `all-MiniLM-L6-v2`;
2. NLI probabilities for `entailment / contradiction / neutral`;
3. optional cross-report stance consistency.

The model interface can stay the same if the final feature layout remains
`[alignment, support, refute, neutral, conflict]`.
