# DataGate — Complete Tech Stack (Start to End)

> Companion to `01_PROJECT_AND_ARCHITECTURE.md`. Opinionated choices with rationale and alternatives. Everything needed to build Mode A (open-source checker) first, then Mode B (marketplace) on the same foundation.

Guiding principles: **Python for the ML/data core**, **TypeScript for the UI**, **boring, battle-tested infra**, and **don't adopt Mode B's heavy machinery until Mode A is proven**.

---

## 1. Core language & runtime

- **Python 3.11+** — the analysis engine, analyzers, API. Non-negotiable: the entire data/ML ecosystem lives here.
- **TypeScript** — the web frontend.
- Rationale: one strong language per concern beats a clever polyglot mix. Python owns the science; TS owns the interface.

## 2. Data handling & profiling

- **Polars** (primary) + **pandas** (compatibility) — dataframes. Polars for speed/memory on large files; pandas where a library needs it.
- **PyArrow** — Parquet/Arrow I/O, zero-copy interchange.
- **DuckDB** — in-process SQL analytics over files without a server; great for fast profiling queries.
- **ydata-profiling** — quick statistical profiling to bootstrap; replace hot paths with custom code later.
- Alternatives: Dask/Ray if datasets outgrow single-node memory (defer until needed).

## 3. ML & the fit engine

- **PyTorch** — probe models and, critically, **gradient/influence computation** for the directional-value check. You need gradients; PyTorch gives them cleanly.
- **scikit-learn** — classical baseline/probe models, metrics, calibration.
- **Hugging Face `transformers` + `datasets`** — loading/handling model and dataset artifacts, especially for text/LLM use cases.
- **sentence-transformers** (text) and **open-CLIP** (images) — embeddings for gap-coverage and distribution checks.
- **FAISS** — fast nearest-neighbor search over embeddings (coverage analysis, dedup, contamination).
- Fit-check specifics: implement **gradient alignment** and **influence-function proxies** on sampled data; borrow ideas from data-valuation literature (Shapley/influence) but keep it *cheap* (subset-based) — that cheapness is the point.

## 4. Security scan stack

- **cleanlab** — label errors and data-quality issues (confident learning).
- **datasketch** (MinHash / LSH) — near-duplicate detection and benchmark contamination overlap at scale.
- **Presidio** (Microsoft) — PII detection/redaction.
- **Custom poisoning/backdoor detectors** — activation/embedding clustering, intra-class distance outliers, trigger localization. This is your research edge; build it in-house on top of PyTorch + FAISS.
- **Dataset fingerprinting** — content hashing + perceptual/semantic fingerprints to detect re-uploads of known public datasets (feeds Mode B provenance).

## 5. Orchestration of the analysis pipeline

- **Custom plugin interface** for analyzers (each: `run(dataset, spec) -> ScoredEvidence`). Keep the contract tiny and stable.
- **Prefect** *or* **Dagster** — pipeline orchestration for multi-step, retriable analysis DAGs with observability. Start simpler if you like and graduate to one of these.
- For plain async job execution (upload → analyze → report): **Celery + Redis** (mature) or **Arq** (async-native, lighter). Pick Celery for ecosystem, Arq for simplicity.

## 6. Backend API

- **FastAPI** — async, typed, auto OpenAPI docs. The default modern Python API framework.
- **Pydantic v2** — request/response and `ProjectSpec` schema validation.
- **Uvicorn + Gunicorn** — ASGI server + process management.
- **Task-Spec Compiler / Report Generator** — call an LLM via API (provider-agnostic wrapper). Keep the LLM boxed in: it parses intent and phrases reports; it never computes the scores.

## 7. Datastores

- **PostgreSQL** — system of record: users, datasets metadata, specs, jobs, reports, deals. Reach for **pgvector** if you want embeddings in Postgres instead of a separate store.
- **S3-compatible object storage** — **AWS S3** in cloud, **MinIO** for self-host/dev — raw datasets and rendered reports (large, blob-shaped).
- **Redis** — queue broker + caching + rate limiting.
- **Vector index** — **FAISS** in-process to start; **Qdrant** or **pgvector** if you need a persistent, queryable vector service.

## 8. Frontend

- **React + TypeScript + Vite** — SPA. Use **Next.js** instead if you want SSR/marketing pages in one app.
- **Tailwind CSS** — styling.
- **TanStack Query** — server state, polling long-running analysis jobs.
- **Recharts** or **visx** — the report's scorecards and charts.
- Deliverable focus: an upload flow, a spec-review/edit screen (surface the parsed `ProjectSpec` for correction), and a clear report view (Safety score, Fit score, predicted-gain range, flagged/quarantined samples, confidence).

## 9. Auth & identity

- **Mode A:** API keys for programmatic use + a hosted auth provider (**Clerk** or **Auth0**) for the web app. Keep it light.
- **Mode B:** add **KYC identity verification** — **Stripe Identity** or **Persona** — so sellers are accountable (this underpins "verified provenance").

## 10. Deployment & infrastructure

- **Docker** — containerize everything (API, workers, frontend).
- **Orchestration:** start on a PaaS — **Render**, **Fly.io**, or **AWS ECS/Fargate** — to move fast. Graduate to **Kubernetes (EKS/GKE)** only when scale/GPU scheduling demands it.
- **GPU workers** — a separate worker pool for embedding generation and probe-model training (managed GPU nodes or a provider like Modal/RunPod early on).
- **Terraform** — infrastructure as code from day one; cheap insurance.
- **Secrets** — cloud KMS / Secrets Manager (never in env files in prod).

## 11. Observability, quality, CI/CD

- **OpenTelemetry** + **Prometheus/Grafana** — tracing and metrics.
- **Sentry** — error tracking.
- **Structured logging** (structlog) — and, for Mode B, **append-only audit logs**.
- **pytest** + **Hypothesis** — unit/property tests. Plus an **ML evaluation harness**: run analyzers against datasets with *known* injected problems (label noise, planted backdoors, known-useless data) and track detection rates and score calibration release-over-release. This harness is how you keep the "honesty" promise measurable.
- **GitHub Actions** — CI/CD.
- **Ruff + Black + mypy** (Python), **ESLint + Prettier** (TS) — linting/formatting/typing.

## 12. Mode B: additional stack (defer until Phase 2+)

- **Payments & escrow** — **Stripe Connect** (marketplace payments with hold/release). Handles the money, disputes, payouts.
- **Per-deal isolation** — ephemeral containers/namespaces per evaluation; destroy inputs after scoring; strict IAM boundaries.
- **Privacy Tier 1 (escrow):** encryption at rest/in transit (KMS), tight access control, signed reports (sign with a platform key so both parties can verify integrity).
- **Privacy Tier 2 (zero-trust, Phase 3):**
  - **Secure enclaves** — **AWS Nitro Enclaves** or **Azure Confidential Computing** (run the engine where even the operator can't see inputs), or
  - **Secure multi-party computation** — **CrypTen** or **MP-SPDZ**, and/or **FHE** for narrow scoring steps. All slow and specialized — a premium tier, not a default.
- **Provenance index** — a service that fingerprints incoming datasets and checks them against a corpus of known public datasets (Kaggle/HF/web) to catch re-uploads; pairs with watermarking.

## 13. Recommended build order (start → end)

1. **Engine as a library** — ingestion, profiling, embeddings, the security analyzers, the fit estimator (gap coverage + directional value), the aggregator. Pure Python, testable in isolation.
2. **ML eval harness** — datasets with planted problems; measure detection + calibration. Build this *alongside* the engine, not after.
3. **CLI** — run Mode A locally end to end. Prove the science before any UI.
4. **FastAPI + Postgres + Redis + object storage + async workers** — wrap the engine as a service.
5. **React frontend** — upload, spec review/edit, report view. **Ship Mode A.**
6. **Provenance + KYC + Stripe escrow + per-deal isolation + signed reports** — **Mode B trusted-escrow marketplace** (Privacy Tier 1).
7. **Secure enclaves / MPC + verifiable-computation proofs** — **Mode B zero-trust premium tier.**

---

### One-line summary of the stack

Python 3.11 · Polars/PyArrow/DuckDB · PyTorch/scikit-learn/HF · sentence-transformers/CLIP/FAISS · cleanlab/datasketch/Presidio · FastAPI/Pydantic · Postgres(+pgvector)/Redis/S3(MinIO) · Prefect or Celery · React/TS/Vite/Tailwind/TanStack Query · Docker/Terraform/GitHub Actions · OpenTelemetry/Prometheus/Sentry — then, for the marketplace: Stripe Connect/Identity · Nitro Enclaves or MP-SPDZ/CrypTen.
