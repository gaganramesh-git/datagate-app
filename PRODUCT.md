# Product


## Platform

web

## Users

Three audiences use the same app, and all three are confirmed primary:

- **ML / data engineers** about to train or fine-tune, checking a downloaded or purchased dataset at their desk. They read the report closely — evidence, metrics, individual flagged rows.
- **Team leads and reviewers** who need the verdict and its reasons quickly, then approve the data or send it back.
- **Dataset buyers and sellers** using blind evaluation before a purchase, for whom the signed, score-only report and its trustworthiness matter most.

## Product Purpose

DataGate is a pre-ingestion checkpoint for training data. Given a dataset and what the user wants to train (optionally a project brief and a sample of their own task data), it answers *is it safe?* and *will it actually help my model?*, and ends with a decision — PASS, REVIEW or BLOCK — plus the reasons, the rows to pull, and a cleaned file to use instead. Success: a user can trust the verdict, understand why in seconds, and act on it (export the cleaned data, review flagged rows, hand over a signed report).

## Positioning

A gate, not a dashboard: it issues a decision with reasons and a cleaned dataset, measured against the user's own task data, with every score shown as an estimate with confidence. Its detection quality is measured by an eval harness against planted problems and real datasets, and reports are deterministic and signable.

## Operating Context

- Flow: upload candidate dataset (+ optional own task data) → state the objective and optional brief → review the parsed task spec → analysis with live progress → report.
- Report contents: verdict banner with reasons, safety and fit scores with confidence, per-check breakdown, prioritized recommendations, evidence table with raw metrics, quarantine list with filters and row previews, downloads (cleaned dataset, PII-masked cleaned dataset, quarantined rows, JSON, Markdown), signature verification.
- Also: history of past analyses; a marketplace preview for blind evaluation (list a dataset, evaluate blind, accept/decline).
- Runs locally (FastAPI serving a zero-build Preact + htm SPA); the page makes no third-party requests under a strict CSP.

## Capabilities and Constraints

- 16 analyzers: 8 security (record quality, label quality, duplication, PII & secrets, hidden content, poisoning & backdoors, eval contamination, provenance) and 8 fit (task structure, learnable signal, leakage risk, gap coverage, distribution match, directional value, data value, marginal gain). Five fit checks need the user's own task data.
- No build step and no Node on the development machine; all frontend assets are served from the project (vendored library, local fonts only).
- Light and dark themes with a toggle must remain.
- Every existing feature must remain.
- Images are not analyzed; the marketplace has no payments, identity or escrow and is labeled a preview.

## Brand Commitments

- Name **DataGate** and the existing shield-with-check mark.
- Honest, hedged voice in report text: scores are "estimates with confidence, not guarantees"; provenance "matches", never "proven ownership".
- Must not read as: a generic SaaS dashboard (purple gradients, card soup, stock icon tiles), a hacker/terminal costume (neon on black, monospace everything), anything playful or gamified, or a sparse layout that hides numbers.

## Evidence on Hand

- Measured detection results in README.md (eval harness), sample report in `examples/sample_report.md`, demo datasets in `examples/`.
- No customers, testimonials, pricing or deployments exist; none may be invented.

## Product Principles

1. The verdict comes first; the evidence is always one step away and never hidden.
2. Dense and scannable beats spacious and decorative — the report is read closely.
3. Every number carries its uncertainty; the interface never overclaims.
4. Trust is earned through provenance: digests, signatures and stated limits are first-class, not footnotes.
5. One app serves the reader who skims and the reader who audits.
