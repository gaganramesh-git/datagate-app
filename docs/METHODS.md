# Methods and limits

How each check works, what it is based on, and where it stops working. Thresholds quoted here are the defaults in the code; the eval harness (`evals/harness.py`) is what keeps them honest.

## Shared machinery

**Column kinds.** Every column is classified as text, categorical, numeric, identifier or other. Identifier-like columns (unique integers, unique short strings) are ignored by model-based checks. Content scanners read every string column regardless of kind.

**One feature space.** Text becomes TF-IDF (up to 20,000 terms), numerics are standardized, categoricals are one-hot encoded. When you supply your own task data, both datasets are featurized together, so they are always compared in the same space. Geometry-based checks use a 64-dimensional LSA projection of that space. A neural encoder would slot in at `features.embed`; nothing else would change.

**Proxy models.** Logistic regression for text, gradient-boosted trees for purely tabular data. Out-of-fold predictions (3-fold) are computed once and shared.

**Sampling.** Model-based checks use a seeded sample of at most 20,000 candidate rows and 5,000 reference rows. Content scans are exhaustive.

**Scoring.** Each check maps a problem rate to a score with an `ok` rate (no penalty) and a `bad` rate (score 0), linear in between.

## Security checks

### Label quality — confident learning
Following Northcutt et al. (2021): each class gets a threshold equal to the mean predicted probability of that class among rows labeled with it. A row is a suspected mislabel when the model is confidently in a different class (and beats the given label by at least 0.10). Scored with ok = 0.5%, bad = 15%.

*Limits.* It abstains when the proxy model barely beats chance — a model that knows nothing cannot tell you which labels are wrong. On inherently noisy tabular targets, genuinely ambiguous rows are flagged alongside wrong ones (measured precision ≈ 60%). Classification only.

### Duplication
Exact repeats; repeats after normalizing case, punctuation and spacing; near-duplicates by MinHash/LSH (3-word shingles, Jaccard ≥ 0.85, up to 20,000 rows); and identical text carrying different labels. In low-cardinality tabular data, repeated rows are treated as natural and not penalized.

### PII and secrets
Regular expressions run inside polars over the full dataset: emails, phone numbers, SSN-shaped IDs, payment cards, IPv4 addresses, AWS keys, private-key headers, common API-token shapes, JWTs and `password = …`-style assignments. Card candidates must be card-shaped (issuer digit, 4-4-4-N or Amex grouping) *and* pass the Luhn checksum. Any critical kind is a blocker. Matched values are never written to a report.

*Limits.* Pattern-based: names, addresses and free-form identifiers are not detected. US-centric phone and ID formats.

### Hidden content
Bidirectional-override characters (U+202A–202E, U+2066–2069) and Unicode tag characters (U+E0000–E007F) are critical and block. Zero-width space, word joiner, BOM and control characters are warnings, as are prompt-injection phrases, chat-template control tokens and active web content. Zero-width joiner and non-joiner are deliberately not flagged — they are legitimate in emoji and in Arabic and Indic scripts.

*Limits.* Injection phrases are a short English list. In a dataset *about* prompt injection these hits are expected.

### Poisoning and backdoors
Two signals, neither requiring a training run on the suspect data.

**Class geometry.** Within each class, rows are scored by distance from the class centroid and by projection onto the class's eight leading singular directions (the idea behind spectral signatures, Tran et al. 2018), using median/MAD z-scores. Each axis is then classified:
- a handful of isolated extreme rows → outliers, quarantined;
- a small, *uniformly* extreme group (up to 6% of the class) → an injected-looking cluster, quarantined;
- the same tight shape but larger → reported as likely boilerplate (a repeated signature or template), not quarantined;
- a spread-out group whose tail merely crosses the threshold → a natural sub-population, ignored.

**Backdoor trigger tokens.** A dirty-label backdoor stamps a token onto rows of other classes and relabels them. Candidate tokens are those almost perfectly predictive of one label. The test is counterfactual: mask the token (and any tokens that always co-occur with it) and re-read the rows. For an honest strong word the rest of the text still supports the label; for a trigger, the predicted probability collapses and the remaining text argues for another class. A trigger with enough support is a blocker.

*Limits — these matter.* Spectral signatures were designed for a trained model's representations; this uses TF-IDF/LSA, a weaker stand-in. **Clean-label backdoors** (trigger added to rows that already carry the target label) pass the counterfactual test by construction and are not detected. Tabular feature-space attacks are not covered. An attacker who knows the method can poison below the support threshold or above the cluster-size limit. Reported confidence is 0.5 without a trigger and 0.75 with one for these reasons.

### Eval contamination
Rows of your reference data found in the candidate, by normalized content match plus MinHash near-matching. Overlapping candidate rows are quarantined, so the cleaned export is de-contaminated. Scored on the share of *reference* rows affected (bad = 25%).

### Provenance
Each registered dataset is stored as a bottom-k sketch of 64-bit row hashes (4,096 values, no raw rows; hashing ignores column order). Two sketches estimate containment in both directions: how much of the candidate is the known dataset, and how much of the known dataset is inside the candidate. Half or more of a registered benchmark or eval set in either direction is a blocker.

*Limits.* Detects row-level reuse, not paraphrase or derivation. Only knows datasets you registered or uploaded; there is no public index. This is "matches", never "ownership proven".

## Fit checks

### Task structure
Modality, target presence and type, class count, balance (normalized entropy), per-class support, volume, and — with reference data — whether the label vocabularies overlap. A missing or unusable target caps fit.

### Learnable signal
Balanced accuracy of the proxy model on held-out folds, as lift over chance (R² for regression). It answers "do the features say anything about the target", and makes no claim about your final accuracy.

### Leakage risk
A single categorical column whose values determine the label (purity ≥ 0.985), a single numeric column on which a depth-3 tree reaches ≥ 0.985 balanced accuracy, or a numeric column correlated ≥ 0.995 with a regression target. A "verify this" signal — decisive features do exist.

### Gap coverage (needs your data)
Both datasets are embedded together. A reference row is *covered* when its five nearest candidate neighbors are, on average, within the median distance of a reference row to its own five nearest reference neighbors. *Relevance* is the share of candidate rows within that radius of some reference row. Score = 0.6 × coverage + 0.4 × relevance.

### Distribution match (needs your data)
Adversarial validation: a classifier is trained to tell candidate rows from reference rows. AUC ≈ 0.5 means indistinguishable. The score only starts falling above AUC 0.6, and the strongest features on each side are reported. Data from another source is nearly always somewhat distinguishable, so this carries modest confidence (0.55).

### Directional value (needs labeled data on both sides)
For a linear softmax (or least-squares) model at zero initialization, the first-step gradient is computed on the candidate and on your data; their cosine says whether learning from the candidate moves toward your task (aligned), nowhere (orthogonal) or away (opposed). A "taste test" then takes 25 gradient steps on the candidate alone and reports how much your task's loss drops relative to 25 steps on your own data. An opposed direction caps fit.

### Data value (needs labeled data on both sides)
Exact KNN-Shapley (Jia et al. 2019): with your data as the validation set, each candidate row's Shapley value under a 5-nearest-neighbor utility has a closed form. Values sum to the KNN utility, which is scored against what random neighbors would achieve. Negative-value rows — near your data but with a conflicting label — are listed, not quarantined. Uses up to 4,000 candidate and 400 reference rows.

### Marginal gain (needs at least 40 labeled reference rows)
*Transfer*: train a linear proxy on the candidate alone, score on your data. *Marginal gain*: across up to four folds of your data, compare training on your fold alone against your fold plus the candidate; the min and max difference is the range the report quotes.

*Limits.* A linear proxy on a small sample of your data. It gives direction and rough size. When the task is already saturated on your sample the range reads "+0.0 to +0.0", which says nothing about harder data.

## The gate

**Safety** = 0.6 × confidence-weighted mean + 0.4 × the worst confidence-adjusted signal (`1 − confidence × (1 − score)`). Any blocker caps it at 0.49.

**Fit** = confidence-weighted mean; capped at 0.30 by a fit-breaking finding, and at 0.70 ("plausible — not verified against your task data") when no reference data was supplied.

**Composite confidence** = mean analyzer confidence × (0.5 + 0.5 × share of checks that could run).

**Decision.** BLOCK on any blocker or safety < 0.50. REVIEW when safety < 0.80, fit < 0.50, confidence is too low, or the task spec is ambiguous. Otherwise PASS. Thresholds are a `GatePolicy`, stored in every report.

## Reports

A report's digest is the SHA-256 of its canonical JSON, excluding timings, the digest and the signature; integral floats are normalized so the digest survives a round trip through any JSON implementation. Signatures are Ed25519 over the digest. A signature verified against the key embedded in the report proves integrity only; to establish who issued it, compare against a public key obtained separately.

A blind (score-only) report removes row indices, the quarantine list, per-row flags, content-bearing metrics, recommendations that quote the data, and the storage path, then gets its own digest and signature.
