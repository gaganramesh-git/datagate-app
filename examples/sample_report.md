# DataGate report — candidate_dirty.csv

**BLOCK** — Blocked — do not train on this dataset as-is

- 20 rows contain SSN-shaped IDs — credentials and government/payment identifiers must not enter training.
- 8 rows contain bidirectional-override characters — text a reviewer sees differently from what a model reads.
- 1 suspected backdoor trigger(s): a token that forces a label the rest of the text contradicts (40 rows).

- Rows × columns: 2,090 × 2
- task: classification · modality: text · target: label · domain: e-commerce
- Reference data: reference.csv (400 rows)

| Check | Score | Confidence | Verdict |
|---|---:|---:|---|
| Safety | 0.44 | 0.80 | critical findings — blocked |
| Fit | 0.92 | 0.66 | strong fit |

Predicted marginal gain: **+0.0 to +0.0 pts balanced accuracy (linear-proxy estimate on your task data)**

## What to do next

1. Redact or drop the 20 rows containing personal data or secrets (the cleaned export can mask them in place instead).
2. Inspect rows containing the suspected trigger token(s) 'zxq7' → 'positive' (40 rows). If you did not put them there, treat the dataset as poisoned.
3. Review 30 rows that sit far from their own class — typical of label-flipped or injected off-distribution samples.
4. Strip invisible/control characters and review rows with injection phrases or active content; the cleaned export drops the affected rows.
5. Review the 86 rows with suspected label errors (4.1% of those checked); relabel or drop them before training.
6. Remove the 60 duplicate rows (2.9%) — they inflate apparent dataset size and leak between train and validation splits.

## Evidence

| Dimension | Family | Score | Confidence | Summary |
|---|---|---:|---:|---|
| Record quality | security | 1.00 | 0.90 | all 2,090 rows are structurally sound |
| Label quality | security | 0.75 | 0.86 | 86 suspected label errors in 2,090 checked rows (4.1%) |
| Duplication | security | 0.94 | 0.90 | 60 exact, 0 same-content and 0 near-duplicate rows (2.9% of 2,090) |
| PII & secrets | security | 0.00 | 0.80 | personal data or secrets in 20 of 2,090 rows (1.0%): email addresses (20), SSN-shaped IDs (20) |
| Hidden content | security | 0.23 | 0.85 | hidden or adversarial content in 14 of 2,090 rows (0.7%): bidirectional-override characters (8), zero-width characters (8), prompt-injection phrases or chat control tokens (6) |
| Poisoning & backdoors | security | 0.10 | 0.75 | 30 class-geometry outliers (1.4% of 2,090 analyzed rows) and 1 suspected backdoor trigger(s) — heuristic, cannot rule out stealthy attacks |
| Eval contamination | security | 1.00 | 0.90 | no overlap between the candidate and 400 reference rows |
| Provenance | security | — | — | Skipped: no known datasets registered to compare against (`datagate registry add`) |
| Task structure | fit | 1.00 | 0.85 | structure matches the declared task |
| Learnable signal | fit | 0.90 | 0.70 | a quick proxy model reaches balanced accuracy 0.95 on held-out folds (chance 0.50) — strong learnable signal about 'label' |
| Leakage risk | fit | — | — | Skipped: no structured feature columns to check |
| Gap coverage | fit | 0.94 | 0.70 | candidate covers 99% of your task data's regions; 86% of candidate rows are relevant to it |
| Distribution match | fit | 1.00 | 0.55 | a classifier separates the candidate from your task data with AUC 0.52 (indistinguishable) |
| Directional value | fit | 0.92 | 0.70 | learning direction is aligned with your task (gradient cosine +0.95); a taste of the candidate achieves 83% of the loss reduction your own data would |
| Data value | fit | 0.87 | 0.70 | nearest-neighbor utility on your task data is 0.93 (random would be 0.50); 93% of candidate rows help, 7% hurt |
| Marginal gain | fit | 0.80 | 0.75 | trained on the candidate alone, a proxy scores balanced accuracy 1.00 on your task data (chance 0.50); adding it to a sample of your data changes the score by +0.0 to +0.0 pts balanced accuracy |

## Quarantine — 250 rows

| Reason | Rows |
|---|---:|
| suspected_mislabel | 86 |
| exact_duplicate | 60 |
| backdoor_trigger | 40 |
| class_outlier | 30 |
| pii | 20 |
| hidden | 14 |

---

Scores are calibrated estimates with confidence, not guarantees. A clean result lowers risk; it does not prove a dataset is safe.

Engine 0.2.0 · seed 7 · dataset SHA-256 `de1ce17c169a5bb7e6900a6c990e105b78157207fcdd1dc81e837aeda623c77f` · report digest `0d2ba90a43073494bf2cc2ffec294b09edd471aaa641f5f0278ecc868f98b48f`
