"""Report rendering: rich console view, Markdown, and JSON.

Everything shown is grounded in analyzer evidence — the renderer phrases,
it never invents a verdict. All copy uses "likely/confidence", never
guarantees (honesty by design).
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from datagate.types import Report, Severity, Verdict

_VERDICT_STYLE = {Verdict.PASS: "green", Verdict.REVIEW: "yellow", Verdict.BLOCK: "red"}
DISCLAIMER = ("Scores are calibrated estimates with confidence, not guarantees. "
              "A clean result lowers risk; it does not prove a dataset is safe.")


def _context_bits(report: Report) -> list[str]:
    spec = report.spec
    bits = [f"task: {spec.task_type.value}", f"modality: {spec.modality.value}"]
    for label, value in [("target", spec.target_column), ("domain", spec.domain),
                         ("deployment", spec.deployment_context),
                         ("metric", spec.success_metric)]:
        if value:
            bits.append(f"{label}: {value}")
    return bits


def render_console(report: Report, console: Console | None = None) -> None:
    c = console or Console()
    style = _VERDICT_STYLE[report.decision.verdict]

    header = (f"[bold]{report.dataset_name or report.dataset_path}[/bold]  "
              f"({report.dataset_rows:,} rows × {report.dataset_columns} cols)\n"
              + " · ".join(_context_bits(report)))
    if report.reference_name:
        header += (f"\nreference: {report.reference_name} "
                   f"({report.reference_rows:,} rows)")
    c.print(Panel(header, title="DataGate Report", border_style="cyan"))

    reasons = "\n".join(f"  • {r}" for r in report.decision.reasons)
    c.print(Panel(f"[bold {style}]{report.decision.verdict.value.upper()}[/bold {style}] — "
                  f"{report.decision.headline}\n{reasons}", border_style=style))

    scores = Table(show_header=True, header_style="bold")
    for col in ("Check", "Score", "Confidence", "Verdict"):
        scores.add_column(col, justify="right" if col in ("Score", "Confidence") else "left")
    for name, comp in (("Safety", report.safety), ("Fit", report.fit)):
        scores.add_row(name, f"{comp.score:.2f}", f"{comp.confidence:.2f}",
                       _color(comp.label, comp.score))
    c.print(scores)
    if report.predicted_gain_range:
        c.print(f"  Predicted marginal gain: [bold]{report.predicted_gain_range}[/bold]")

    ev = Table(title="Evidence", show_header=True, header_style="bold")
    ev.add_column("Dimension")
    ev.add_column("Family")
    ev.add_column("Score", justify="right")
    ev.add_column("Conf", justify="right")
    ev.add_column("Summary", overflow="fold")
    for e in report.evidence:
        ev.add_row(e.dimension or e.analyzer, e.family,
                   "—" if e.skipped else f"{e.score:.2f}",
                   "—" if e.skipped else f"{e.confidence:.2f}",
                   e.summary, style="dim" if e.skipped else "")
    c.print(ev)

    if report.recommendations:
        c.print("[bold]What to do next[/bold]")
        for i, rec in enumerate(report.recommendations, 1):
            c.print(f"  {i}. {rec}")

    if report.quarantine_total:
        n_crit = sum(1 for f in report.quarantine_list if f.severity is Severity.CRITICAL)
        by_reason = ", ".join(f"{k} ({v:,})" for k, v in report.quarantine_by_reason.items())
        q = Table(title=f"Quarantine — {report.quarantine_total:,} rows "
                        f"({n_crit:,} critical). {by_reason}",
                  show_header=True, header_style="bold red")
        q.add_column("Row", justify="right")
        q.add_column("Severity")
        q.add_column("Reason", overflow="fold")
        q.add_column("Detail", overflow="fold")
        for f in report.quarantine_list[:15]:
            q.add_row(str(f.row_index), f.severity.value, f.reason, f.detail or "")
        c.print(q)
    else:
        c.print("[green]No rows quarantined.[/green]")

    c.print(f"\n[dim]{DISCLAIMER}  Digest {str(report.digest)[:16]}… · "
            f"engine {report.engine_version}[/dim]")


def render_markdown(report: Report) -> str:
    d = report.decision
    lines = [
        f"# DataGate report — {report.dataset_name or report.dataset_path}",
        "",
        f"**{d.verdict.value.upper()}** — {d.headline}",
        "",
        *[f"- {r}" for r in d.reasons],
        "",
        f"- Rows × columns: {report.dataset_rows:,} × {report.dataset_columns}",
        f"- {' · '.join(_context_bits(report))}",
    ]
    if report.reference_name:
        lines.append(f"- Reference data: {report.reference_name} "
                     f"({report.reference_rows:,} rows)")
    lines += [
        "",
        "| Check | Score | Confidence | Verdict |",
        "|---|---:|---:|---|",
        f"| Safety | {report.safety.score:.2f} | {report.safety.confidence:.2f} "
        f"| {report.safety.label} |",
        f"| Fit | {report.fit.score:.2f} | {report.fit.confidence:.2f} "
        f"| {report.fit.label} |",
    ]
    if report.predicted_gain_range:
        lines += ["", f"Predicted marginal gain: **{report.predicted_gain_range}**"]

    if report.recommendations:
        lines += ["", "## What to do next", ""]
        lines += [f"{i}. {rec}" for i, rec in enumerate(report.recommendations, 1)]

    lines += ["", "## Evidence", "",
              "| Dimension | Family | Score | Confidence | Summary |",
              "|---|---|---:|---:|---|"]
    for e in report.evidence:
        score = "—" if e.skipped else f"{e.score:.2f}"
        conf = "—" if e.skipped else f"{e.confidence:.2f}"
        summary = e.summary.replace("|", "\\|")
        lines.append(f"| {e.dimension or e.analyzer} | {e.family} | {score} | {conf} "
                     f"| {summary} |")

    lines += ["", f"## Quarantine — {report.quarantine_total:,} rows", ""]
    if report.quarantine_by_reason:
        lines += ["| Reason | Rows |", "|---|---:|"]
        lines += [f"| {k} | {v:,} |" for k, v in report.quarantine_by_reason.items()]
    else:
        lines.append("No rows quarantined.")

    lines += ["", "---", "", DISCLAIMER, "",
              f"Engine {report.engine_version} · seed {report.seed} · "
              f"dataset SHA-256 `{report.dataset_fingerprint}` · "
              f"report digest `{report.digest}`"]
    if report.signature:
        lines.append(f"Signed (Ed25519), public key `{report.signature['public_key']}`")
    return "\n".join(lines) + "\n"


def write_json(report: Report, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report.model_dump(mode="json"), indent=2))


def write_markdown(report: Report, path: str | Path) -> None:
    Path(path).write_text(render_markdown(report), encoding="utf-8")


def _color(label: str, score: float) -> str:
    style = "green" if score >= 0.8 else ("yellow" if score >= 0.5 else "red")
    return f"[{style}]{label}[/{style}]"
