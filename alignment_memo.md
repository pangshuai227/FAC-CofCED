# CofCED + FaithfulRAG Alignment Memo

## 1. Problem Framing

Baseline paper:

- **A Coarse-to-fine Cascaded Evidence-Distillation Neural Network for Explainable Fake News Detection** (COLING 2022)
- Method name: **CofCED**
- Task: given a claim and multiple relevant raw reports, predict the veracity label and extract explanatory evidence sentences.

Reference paper:

- **FaithfulRAG: Fact-Level Conflict Modeling for Context-Faithful Retrieval-Augmented Generation** (arXiv 2025)
- Method name: **FaithfulRAG**
- Task: given a question and retrieved context, generate an answer that remains faithful when retrieved context conflicts with the LLM's parametric knowledge.

The useful connection is not "replace CofCED with RAG generation". The useful connection is:

> CofCED distills evidence from noisy raw reports, but does not explicitly model fact-level conflicts among evidence. FaithfulRAG provides a fact-level conflict modeling idea that can be adapted into CofCED's report selection, sentence selection, and final veracity prediction.

## 2. What CofCED Already Does

CofCED formulates explainable fake news detection as a multi-task learning problem:

1. **Report selection**: select top-K check-worthy raw reports.
2. **Sentence selection**: extract explanatory sentences from the selected reports.
3. **Veracity prediction**: predict the claim label using the claim, all reports, and selected evidence.

Its fine-grained sentence selector scores candidate evidence sentences using four features:

- **Claim relevance**: whether the sentence is related to the claim.
- **Richness**: whether the sentence contains enough informative content.
- **Salience**: whether the sentence is important in its report.
- **Non-redundancy**: whether the sentence adds new information compared with previously selected evidence.

Final prediction uses:

```text
h_final = [h_claim; h_reports; h_evidence]
```

where `h_evidence` is aggregated from selected explanatory sentences.

## 3. Main Weakness of CofCED

CofCED assumes that good evidence can be found by relevance, informativeness, salience, and non-redundancy. This is useful, but incomplete for fake news detection because raw reports often contain:

- evidence that supports the claim,
- evidence that refutes the claim,
- mutually conflicting reports,
- copied or paraphrased claims without verification,
- partially correct facts mixed with misleading conclusions.

Therefore, a sentence can be highly relevant and salient while still being misleading or contradicted by other reports. CofCED does not explicitly distinguish:

```text
relevant evidence
vs.
supporting evidence
vs.
refuting evidence
vs.
conflicting evidence
```

This is the opening for improvement.

## 4. What FaithfulRAG Contributes

FaithfulRAG addresses knowledge conflicts in RAG. Its key idea is:

> Do not force the model to blindly trust either parametric knowledge or retrieved context. Instead, externalize facts, align them with context, identify conflicts, and reason over the conflicts.

The method has three transferable components:

1. **Self-Fact Mining**
   - Extract high-level knowledge required by the query.
   - Generate a self-context.
   - Extract fine-grained self-facts.

2. **Contextual Knowledge Alignment**
   - Chunk the retrieved context.
   - Embed self-facts and context chunks.
   - Select context chunks aligned with self-facts.

3. **Self-Think**
   - Analyze facts.
   - Match facts to possible answers.
   - Check whether context is sufficient or conflicting.
   - Produce a final verified answer.

For fake news detection, we should not directly copy the generation pipeline. Instead, we should adapt the fact-level alignment and conflict reasoning into a discriminative, explainable classifier.

## 5. Proposed Direction: Fact-level Conflict-aware CofCED

Suggested method name:

- **Conflict-aware CofCED**
- **Fact-aware CofCED**
- **FaC-CofCED: Fact-level Conflict-aware CofCED**

The core idea:

> Enhance CofCED with fact-level alignment and conflict-aware evidence modeling, so the model can distinguish supporting, refuting, and conflicting evidence from raw reports.

## 6. Module Mapping

| FaithfulRAG Module | Original Role | Adapted CofCED Role |
| --- | --- | --- |
| Self-Knowledge Extraction | Extract knowledge needed to answer a question | Extract key factual aspects implied by the claim |
| Self-Context Generation | Generate self-context from LLM knowledge | Optional; can be skipped for a lighter implementation |
| Self-Fact Extraction | Extract atomic facts from self-context | Extract claim facts and report facts |
| Contextual Alignment | Align self-facts with retrieved context chunks | Align claim facts with report sentences/facts |
| Self-Think | Reason over facts and context before answering | Produce conflict-aware features or a structured verification representation |

## 7. Proposed Architecture

### 7.1 Claim Fact Extraction

For each claim `c`, extract a set of atomic facts:

```text
F_c = {f_c1, f_c2, ..., f_cm}
```

Example:

```text
Claim: "Microwaving fabric masks is a good way to sanitize them for reuse."

Claim facts:
1. Fabric masks can be sanitized by microwaving.
2. Microwaving masks is safe for reuse.
3. The method is recommended/effective.
```

Implementation options:

- lightweight: use an LLM prompt offline;
- non-LLM: use sentence simplification / OpenIE / dependency patterns;
- simplest project version: treat the claim itself as one claim fact and align report sentences to it.

### 7.2 Report Fact Extraction

For every sentence `s_ij` in raw reports, extract or represent report facts:

```text
F_sij = {f_ij1, f_ij2, ..., f_ijn}
```

For a lightweight implementation, we can skip explicit fact text extraction and use sentence-level embeddings as fact proxies.

### 7.3 Fact-level Alignment

Compute alignment between claim facts and report facts/sentences:

```text
align(f_c, f_s) = cosine(emb(f_c), emb(f_s))
```

FaithfulRAG uses `all-MiniLM-L6-v2` for this kind of embedding-based alignment, which is a practical choice for a course project.

We can derive a sentence-level fact alignment score:

```text
A_ij = max over claim facts and sentence facts of align(f_c, f_sij)
```

### 7.4 Stance / Conflict Detection

Use an NLI or stance classifier to estimate whether each report sentence:

- supports the claim,
- refutes the claim,
- is neutral/irrelevant.

For each sentence:

```text
P_stance(s_ij, c) = [p_support, p_refute, p_neutral]
```

Then define:

```text
conflict_score = p_refute
support_score = p_support
neutral_score = p_neutral
```

If explicit fact extraction is available, NLI can be applied between claim facts and report facts. If not, apply NLI between claim and sentence.

### 7.5 Cross-report Agreement

CofCED has non-redundancy, but not agreement. We add a cross-report consistency feature:

```text
agreement(s_ij) = average stance/fact similarity with evidence from other reports
```

Intuition:

- A sentence refuting the claim becomes stronger if multiple independent reports refute the same claim fact.
- A sentence becomes suspicious if it is isolated and contradicted by most other evidence.

### 7.6 Conflict-aware Sentence Selector

Original CofCED sentence selector:

```text
score(s) =
  claim relevance
+ richness
+ salience
- non-redundancy
```

Proposed selector:

```text
score(s) =
  claim relevance
+ richness
+ salience
- non-redundancy
+ fact alignment
+ stance salience
+ cross-report agreement
```

The added features make the selected explanation sentences more useful for veracity prediction, not just more semantically related.

### 7.7 Conflict-aware Veracity Prediction

Instead of aggregating all selected evidence into one `h_evidence`, split evidence into stance-aware groups:

```text
h_support  = pool(selected supporting evidence)
h_refute   = pool(selected refuting evidence)
h_conflict = pool(high-conflict evidence)
```

Final classifier:

```text
h_final = [h_claim; h_reports; h_support; h_refute; h_conflict; h_agreement]
y_hat = softmax(MLP(h_final))
```

This makes the classifier aware of the direction and structure of evidence.

## 8. Training Objectives

Keep CofCED's original multi-task training:

```text
L_all = beta_D * L_report + beta_S * L_sentence + beta_C * L_classification
```

Add optional auxiliary losses:

### Option A: Stance Auxiliary Loss

If stance labels are generated by a teacher model or heuristic:

```text
L_stance = CE(stance_pred, stance_label)
```

Total:

```text
L_all = beta_D * L_report
      + beta_S * L_sentence
      + beta_C * L_classification
      + beta_T * L_stance
```

### Option B: Contrastive Fact Alignment Loss

Pull aligned claim-report facts closer and push unrelated/conflicting facts apart:

```text
L_align = contrastive(f_claim, f_report)
```

This is more complex and may be optional.

## 9. Minimal Implementation Plan

Given that the current workspace only has the two PDFs and no CofCED code/data, the most realistic implementation plan is:

1. Reproduce or obtain CofCED code and RAWFC/LIAR-RAW data.
2. Add preprocessing:
   - encode claims and report sentences using `all-MiniLM-L6-v2`;
   - compute claim-sentence fact alignment scores;
   - compute support/refute/neutral probabilities using an NLI model.
3. Modify the sentence selector to include extra scalar features.
4. Modify final evidence aggregation into support/refute/conflict groups.
5. Train and evaluate on RAWFC and LIAR-RAW.
6. Run ablations.

## 10. Ablation Design

Suggested variants:

| Variant | Description |
| --- | --- |
| CofCED | Original baseline |
| + FA | Add fact alignment score |
| + ST | Add stance/refute/support score |
| + AGR | Add cross-report agreement |
| + CER | Add conflict-aware evidence representation |
| Full | FA + ST + AGR + CER |

Expected finding:

- FA improves evidence relevance.
- ST improves veracity prediction because fake news labels depend on support/refute direction.
- AGR reduces isolated noisy evidence.
- CER improves classification by separating supporting and refuting evidence.

## 11. Evaluation

Follow CofCED:

- Veracity prediction:
  - macro precision,
  - macro recall,
  - macro F1.

- Explanation quality:
  - ROUGE-1,
  - ROUGE-2,
  - ROUGE-L.

Additional useful metrics:

- Evidence sentence classification F1.
- Refuting evidence recall.
- Human evaluation of whether explanations justify the predicted label.

## 12. Risks and Mitigations

### Risk 1: Fact extraction is noisy

Mitigation:

- Start with sentence-level alignment instead of full fact extraction.
- Use claim sentence as a fact proxy.
- Add full fact extraction only if time allows.

### Risk 2: NLI model domain mismatch

Mitigation:

- Use NLI scores as soft features, not hard labels.
- Add ablation to show whether NLI helps.

### Risk 3: LLM usage may be expensive

Mitigation:

- Use LLM only for offline preprocessing on a subset.
- Prefer MiniLM embeddings and open-source NLI for the full dataset.

### Risk 4: Explanation ROUGE may not improve

Mitigation:

- Emphasize macro-F1 and evidence usefulness.
- Add human or qualitative evaluation showing selected evidence is more label-discriminative.

## 13. Recommended Project Claim

Possible concise claim for the report:

> We propose a fact-level conflict-aware extension of CofCED for explainable fake news detection. Inspired by FaithfulRAG, our method aligns claim-level facts with raw-report evidence and explicitly models support, refutation, and cross-report conflict. This allows the model to select more discriminative explanatory evidence and improve veracity prediction under noisy, contradictory raw reports.

## 14. Recommended Next Step

Before implementation, decide the project scope:

1. **Lightweight version**:
   - no explicit fact extraction;
   - use claim-sentence alignment and NLI stance scores;
   - easiest to implement and evaluate.

2. **Medium version**:
   - add LLM/OpenIE fact extraction offline;
   - align claim facts and report facts;
   - stronger connection to FaithfulRAG.

3. **Full version**:
   - add conflict-aware reasoning/self-think module;
   - possibly use LLM-generated structured verification;
   - highest novelty, highest implementation cost.

For a final course project, the medium version is the best tradeoff if time and compute allow. The lightweight version is the safest baseline improvement.
