# DataGate

A checkpoint for training data. Before a dataset goes anywhere near your model, DataGate answers two questions:

- **Is it safe?** Mislabels, duplicates, personal data and secrets, hidden characters, poisoning and backdoor triggers, overlap with your eval set.
- **Will it actually help *my* model?** Measured against a sample of your own task data — without a full training run.

It ends with a decision, not just scores: **PASS**, **REVIEW** or **BLOCK**, with the reasons, the rows to pull, and a cleaned file to use instead.

```text
BLOCK — Blocked — do not train on this dataset as-is
  • 20 rows contain SSN-shaped IDs — credentials and government/payment identifiers must not enter training.
  • 8 rows contain bidirectional-override characters — text a reviewer sees differently from what a model reads.
  • 1 suspected backdoor trigger(s): a token that forces a label the rest of the text contradicts (40 rows).

Safety 0.44 (critical findings — blocked) · Fit 0.92 (strong fit) · 250 rows quarantined
```

Everything runs locally. The web UI makes no third-party requests.

Design background: [docs/01_PROJECT_AND_ARCHITECTURE.md](docs/01_PROJECT_AND_ARCHITECTURE.md) · [docs/02_TECH_STACK.md](docs/02_TECH_STACK.md) · how each check works and where it stops working: [docs/METHODS.md](docs/METHODS.md)

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[dev,service]"
uv run datagate demo                      # writes sample datasets with planted problems to examples/
```

### Web app

```bash
uv run datagate serve
```

Open http://127.0.0.1:8000. Upload the candidate dataset, optionally a sample of your own data, say what you want to train, review the parsed task spec, and read the report. API reference at `/docs`.

### Command line

```bash
# The gate: security scan + fit check
uv run datagate check examples/candidate_dirty.csv \
    -p "classify product reviews by sentiment" \
    -r examples/reference.csv

# Get the usable part back, and the rows that were pulled
uv run datagate check data.csv -p "..." -r my_eval_set.csv \
    --export-clean clean.csv --export-quarantine quarantine.csv --redact-pii

# In a pipeline: non-zero exit unless the gate passes
uv run datagate check data.csv -p "..." -r my_eval_set.csv --fail-on review --quiet

# Reports you can hand to someone else
uv run datagate check data.csv -p "..." --sign --json report.json --md report.md
uv run datagate verify report.json --pubkey "$(uv run datagate keys show)"

# Teach it which datasets must never leak into training
uv run datagate registry add my_eval_set.csv --name my-eval --kind eval
```

Formats: CSV, TSV, Parquet, JSON, JSONL/NDJSON, plain text (one record per line).

## The two inputs that matter

**What you want to train** (`--purpose`) and, optionally, a **project brief** (`--brief`, text or a `.md` file) are compiled into a task spec — task type, modality, target column, domain, deployment context, success metric. The UI shows the parsed spec for correction before anything runs.

**A sample of your own task data** (`--reference`) — ideally your evaluation set — is what turns "this dataset looks learnable" into "this dataset helps *your* task". Without it, DataGate still runs the full security scan, but fit is capped at *plausible, not verified* and it tells you so.

## What it checks

**Security** — feeds the Safety score and the quarantine list

| Check | What it finds | How |
|---|---|---|
| Record quality | missing labels, mostly-empty rows, blank text, non-finite values | full scan |
| Label quality | suspected mislabels | confident learning on out-of-fold predictions, per-class thresholds |
| Duplication | exact, same-content and near-duplicate rows; identical text with different labels | normalization + MinHash/LSH |
| PII & secrets | emails, phones, SSN-shaped IDs, payment cards (Luhn-checked), keys, tokens, JWTs, inline passwords | full scan in Rust regex; values never echoed |
| Hidden content | bidi-override and invisible tag characters, zero-width and control characters, prompt-injection phrases, active web content | full scan |
| Poisoning & backdoors | injected off-distribution clusters; **backdoor trigger tokens** | spectral class outliers; counterfactual masking test |
| Eval contamination | rows of your reference data inside the candidate | exact + near matching |
| Provenance | overlap with registered benchmarks, eval sets or public datasets | bottom-k row sketches, both directions |

**Fit** — feeds the Fit score

| Check | Question | Needs your data |
|---|---|:-:|
| Task structure | Does the data match the declared task — target, classes, balance, volume, label vocabulary? | |
| Learnable signal | Do the features say anything about the target? | |
| Leakage risk | Does one column give the answer away? | |
| Gap coverage | Does it reach the regions your task lives in? How much of it is on-topic? | ✓ |
| Distribution match | Can a classifier tell it from your data, and by what? | ✓ |
| Directional value | Does learning from it move a model toward your task, or away? | ✓ |
| Data value | Per-row value against your task (exact KNN-Shapley) | ✓ |
| Marginal gain | What does adding it actually do to a proxy model on your data? | ✓ |

Every check is a plugin with one contract — `run(ctx) -> ScoredEvidence` — and the aggregator only ever consumes that.

## How the decision is made

- **Scores are risk-calibrated, not clean-fractions.** A dataset with 5% wrong labels is not "95% fine"; it scores about 0.7 on label quality.
- **Safety is a weakest-link property.** The composite blends the confidence-weighted mean with the single worst signal, so one serious finding cannot be averaged away.
- **Blockers override everything.** Secrets or government IDs, a suspected backdoor trigger, smuggled invisible text, or a swallowed eval set cap safety below the block threshold and force **BLOCK**.
- **Fit is an average, and never blocks alone** — a poor fit is a **REVIEW**. A fit-breaking finding (no usable target, labels that never match, a learning direction opposed to your task) caps it.
- **Confidence is reported, and it shrinks** when few checks could run.

Thresholds live in a versioned `GatePolicy` recorded in every report. Reports are deterministic: the same inputs, seed and policy give the same SHA-256 digest, and can be signed (Ed25519).

## Measured, not asserted

`evals/harness.py` runs the engine against data with known planted problems and against real, clean datasets. Current results (also run in CI):

| | Result |
|---|---|
| Recall — mislabels / duplicates / PII / backdoor rows / injected junk / hidden chars / injection phrases | 91% / 100% / 100% / 100% / 100% / 100% / 100% |
| False-flag rate on untouched rows (dirty set) and on a clean set | 0.0% / 0.0% |
| Safety as label noise rises 0 → 2 → 5 → 10 → 20% | 1.00 → 0.96 → 0.89 → 0.78 → 0.61 (monotone) |
| Fit: on-task vs off-topic vs inverted labels | 0.96 vs 0.22 vs 0.30 (direction read as *opposed*) |
| Tabular: mislabel recall / precision; leaked column named | 74% / 62%; yes |
| Eval contamination: planted reference rows found | 100% |
| Real clean tabular datasets (breast cancer, wine, digits) | all pass; ≤ 1.9% of rows flagged |
| Real text (20 Newsgroups, 2,260 posts) | no false backdoor triggers; 0.0% class outliers; 0.1% suspected mislabels. Blocked — correctly — for a real inline password and ~380 real email addresses |

```bash
uv run python evals/harness.py     # exit 1 if any check regresses
uv run pytest                      # 80 tests
```

Running on real text found two false positives that synthetic data could not — a ZIP code beside a phone number passing the card checksum, and one author's signature block read as an injected cluster. Both are fixed and are now regression tests. Expect more of these on your data; the harness is where they should end up.

## What it does not do

- **It cannot prove a dataset is safe.** Poisoning detection is an open research problem. Clean-label backdoors (a trigger added to rows that already carry the target label) and feature-space attacks on tabular data are not detected. A clean result lowers risk; it is not a guarantee.
- **The fit estimate is a proxy.** Checks use TF-IDF/LSA features and small linear or tree models. The marginal-gain range is a direction and rough size against a sample of your data, not a forecast for your production model. On saturated tasks it reads "+0.0".
- **Images are not analyzed** — only the tabular and text columns of a dataset.
- **Model-based checks sample** at most 20,000 rows; content scans (PII, hidden content, duplicates, quality) read everything.
- **Spec parsing is heuristic** by default. An optional LLM backend exists (`DATAGATE_USE_LLM=1` plus an API key) and is boxed to parsing intent — it never influences a score. It has only been exercised with a fake backend in tests, not against a live API.

## The service

| Endpoint | Purpose |
|---|---|
| `POST /datasets` · `GET /datasets` · `DELETE /datasets/{id}` | upload (streamed, size-capped), list, purge the stored data |
| `POST /specs` · `PATCH /specs/{id}` | compile and correct the task spec |
| `POST /analyses` · `GET /analyses/{id}` · `GET /analyses` | start a job, poll progress, history |
| `GET /reports/{id}` · `/markdown` · `/clean` · `/quarantine` · `/rows` | report, exports, row previews |
| `POST /reports/verify` | check a report's digest and signature |
| `POST /deals` … | blind evaluation preview (below) |

Infrastructure sits behind small seams, with zero-setup defaults:

| Concern | Default | Swap |
|---|---|---|
| Database | SQLite (WAL) via SQLAlchemy | Postgres — `DATAGATE_DATABASE_URL` |
| Blob storage | local, content-addressed | S3/MinIO — implement `Storage` |
| Job queue | in-process thread pool | Celery/Arq — reimplement `worker.submit()` |
| Auth | open, one local user | `DATAGATE_REQUIRE_LOGIN=1` (accounts: sign up, sign in, private data) · `DATAGATE_API_KEYS=key1,key2` (API clients, no-accounts mode only) |

Configuration: `DATAGATE_HOME` (default `./var`), `DATAGATE_DATABASE_URL`, `DATAGATE_STORAGE_DIR`, `DATAGATE_WORKERS`, `DATAGATE_MAX_UPLOAD_BYTES`, `DATAGATE_API_KEYS`, `DATAGATE_REQUIRE_LOGIN`, `DATAGATE_SESSION_DAYS`, `DATAGATE_USE_LLM`, `DATAGATE_LLM_API_KEY`, `DATAGATE_LLM_MODEL`.

A `Dockerfile` is included but was written without a Docker daemon available — build and test it before relying on it. **Putting it on the internet.** Set `DATAGATE_REQUIRE_LOGIN=1` (the Docker image already does). Anyone can then create an account with an email and password, and sees only their own datasets, specs, analyses, reports and history; another person's link answers "not found". Passwords are hashed with scrypt, sessions are random tokens in an HttpOnly cookie (only their hash is stored), and failed sign-ins are rate-limited. Marketplace listings are visible to everyone signed in; a buyer's score, report and decision are visible only to that buyer and the seller. Not included: email verification and password reset (there is no mail service, so a forgotten password is reset by the operator), and deleting an account (datasets can be purged one at a time). Serve it over HTTPS — Render and most hosts do — because the cookie is marked Secure only when the request arrives over HTTPS.

### Blind evaluation (marketplace preview)

The second mode in the design docs: a buyer vets a seller's dataset before purchase, and *only the score leaves*. A seller lists a dataset; a buyer describes their task (optionally with their own data) and receives a **signed, score-only report** — verdict, scores, counts by problem type — with no row indices, no row contents, no flagged tokens and no filename. Exports and row previews are refused for these reports.

This is the engine side of that mode, working end to end. There are **no payments, identity checks, escrow or access control** — anyone who can reach the service can act as either party. Treat it as a demonstration of the report contract, not a marketplace.

## Layout

```
src/datagate/
  types.py            # ProjectSpec, ScoredEvidence, GatePolicy, Report — the contracts
  ingest/             # loading, column-kind inference, profiling
  spec/               # task-spec compiler (heuristic) + optional boxed LLM backend
  features.py         # one shared feature space: TF-IDF + scaled numerics + one-hot
  context.py          # per-run cache: sampled views, joint space, out-of-fold predictions
  analyzers/          # security/ and fit/ plugins
  aggregate.py        # composite scores, quarantine, the gate decision
  orchestrator.py     # runs it all, in parallel, deterministically
  export.py           # cleaned dataset, quarantine list, PII masking
  signing.py          # digests and Ed25519 signatures
  registry.py         # known-dataset sketches (provenance)
  blind.py            # score-only reports
  demo.py             # seeded datasets with planted problems
  report.py · cli.py
  service/            # FastAPI app, persistence, storage, worker, the web UI
evals/harness.py      # detection, calibration, fit and false-positive benchmark
tests/
```

## Roadmap

- [x] Analysis engine, gate decision, exports, signing, CLI
- [x] Service and web UI
- [x] Blind evaluation with score-only signed reports
- [ ] Neural embeddings behind `features.embed` (sentence-transformers, CLIP) and image support
- [ ] Clean-label and tabular backdoor detection; a public-dataset provenance index
- [ ] Calibrating the marginal-gain estimate against real training runs
- [ ] Production infrastructure: Postgres, object storage, a real queue, hosted auth
- [ ] Marketplace: identity, payments and escrow, per-deal isolation, then enclaves/MPC
