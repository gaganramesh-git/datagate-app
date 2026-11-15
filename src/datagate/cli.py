"""DataGate CLI — the gate, fully local.

    datagate check data.csv -p "classify product reviews by sentiment"
    datagate check data.csv -p "..." -r my_eval_set.csv --export-clean clean.csv
    datagate check data.csv -p "..." --fail-on review        # for CI pipelines
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from datagate import export, signing
from datagate.demo import write_demo
from datagate.ingest.loader import load_dataset
from datagate.ingest.profiler import profile_dataset
from datagate.orchestrator import analyze
from datagate.registry import VALID_KINDS, Registry
from datagate.report import render_console, write_json, write_markdown
from datagate.spec.compiler import compile_spec
from datagate.types import Severity, Verdict

app = typer.Typer(help="DataGate: is this dataset safe, and will it help my model?",
                  no_args_is_help=True)
registry_app = typer.Typer(help="Known-dataset registry (benchmarks, eval sets, public data).",
                           no_args_is_help=True)
keys_app = typer.Typer(help="Report-signing keys.", no_args_is_help=True)
app.add_typer(registry_app, name="registry")
app.add_typer(keys_app, name="keys")
console = Console()

EXIT_GATE_FAILED = 1
EXIT_AMBIGUOUS_SPEC = 2


def _read_brief(brief: Optional[str]) -> Optional[str]:
    """A --brief value may be a path to a .md/.txt file or inline text."""
    if not brief:
        return None
    candidate = Path(brief)
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8", errors="replace")
    return brief


@app.command()
def check(
    dataset: str = typer.Argument(..., help="Path to CSV/TSV/Parquet/JSON/JSONL/TXT dataset"),
    purpose: str = typer.Option(..., "--purpose", "-p",
                                help="What you want to train (free text)"),
    brief: Optional[str] = typer.Option(
        None, "--brief", "-b",
        help="Project brief: a .md/.txt path or inline text (the idea, the area of use)"),
    target: Optional[str] = typer.Option(None, "--target", "-t",
                                         help="Label/target column (overrides inference)"),
    reference: Optional[str] = typer.Option(
        None, "--reference", "-r",
        help="A sample of your own task data (ideally your eval set). Unlocks coverage, "
             "directional value, data value, marginal gain and contamination checks"),
    json_out: Optional[str] = typer.Option(None, "--json", "-j", help="Write the report as JSON"),
    md_out: Optional[str] = typer.Option(None, "--md", help="Write the report as Markdown"),
    export_clean: Optional[str] = typer.Option(
        None, "--export-clean", help="Write the dataset with quarantined rows removed"),
    export_quarantine: Optional[str] = typer.Option(
        None, "--export-quarantine", help="Write the quarantined rows with reasons"),
    redact_pii: bool = typer.Option(
        False, "--redact-pii",
        help="In the cleaned export, mask PII in place instead of dropping those rows"),
    sign: bool = typer.Option(False, "--sign", help="Sign the report (Ed25519)"),
    fail_on: str = typer.Option(
        "never", "--fail-on",
        help="Exit non-zero when the gate says: block | review (review or block) | never"),
    seed: int = typer.Option(7, "--seed", help="Seed for every sample the analysis takes"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Don't stop on clarifying questions"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Print only the verdict line"),
) -> None:
    """Run the full security + fit analysis and print the gate decision."""
    if fail_on not in ("never", "block", "review"):
        raise typer.BadParameter("--fail-on must be one of: never, block, review")

    ds = load_dataset(dataset)
    profile_dataset(ds)
    spec = compile_spec(purpose, ds, target_column=target, project_brief=_read_brief(brief))

    if spec.clarifying_questions and not yes:
        console.print("[yellow]The task spec is ambiguous:[/yellow]")
        for q in spec.clarifying_questions:
            console.print(f"  • {q}")
        console.print(f"[dim]Parsed:[/dim] task={spec.task_type.value}, "
                      f"modality={spec.modality.value}, target={spec.target_column}")
        console.print("Refine --purpose/--target, or pass --yes to proceed as parsed.")
        raise typer.Exit(code=EXIT_AMBIGUOUS_SPEC)

    with console.status("Running security scan and fit check…", spinner="dots"):
        result = analyze(dataset, purpose, reference_path=reference, spec=spec, seed=seed)
    report = result.report
    if sign:
        signing.sign_report(report)

    if quiet:
        d = report.decision
        console.print(f"{d.verdict.value.upper()}  safety={report.safety.score:.2f}  "
                      f"fit={report.fit.score:.2f}  quarantined={report.quarantine_total}")
    else:
        render_console(report, console)

    if json_out:
        write_json(report, json_out)
        console.print(f"[dim]Report (JSON) → {json_out}[/dim]")
    if md_out:
        write_markdown(report, md_out)
        console.print(f"[dim]Report (Markdown) → {md_out}[/dim]")
    if export_quarantine:
        export.write_frame(export.quarantine_frame(result.dataset, result.quarantine),
                           export_quarantine)
        console.print(f"[dim]Quarantined rows → {export_quarantine}[/dim]")
    if export_clean:
        cleaned = export.clean_frame(result.dataset, result.quarantine,
                                     min_severity=Severity.WARNING, redact_pii=redact_pii)
        export.write_frame(cleaned, export_clean)
        console.print(f"[dim]Cleaned dataset ({cleaned.height:,} of "
                      f"{result.dataset.n_rows:,} rows) → {export_clean}[/dim]")

    verdict = report.decision.verdict
    if (fail_on == "block" and verdict is Verdict.BLOCK) or \
            (fail_on == "review" and verdict is not Verdict.PASS):
        raise typer.Exit(code=EXIT_GATE_FAILED)


@app.command()
def profile(dataset: str = typer.Argument(..., help="Path to dataset")) -> None:
    """Print the statistical profile of a dataset (no analysis)."""
    ds = load_dataset(dataset)
    console.print_json(data=profile_dataset(ds))


@app.command()
def verify(
    report: str = typer.Argument(..., help="Path to a report JSON file"),
    pubkey: Optional[str] = typer.Option(
        None, "--pubkey", help="Base64 public key you trust (from `datagate keys show`)"),
) -> None:
    """Check that a report is intact, and who signed it."""
    payload = json.loads(Path(report).read_text())
    result = signing.verify_report(payload, trusted_public_key=pubkey)
    style = "green" if result.ok and result.signature_ok else ("yellow" if result.digest_ok
                                                                 else "red")
    console.print(f"[{style}]{result.detail}[/{style}]")
    if not result.ok or not result.digest_ok:
        raise typer.Exit(code=1)


@app.command()
def demo(out: str = typer.Option("examples", "--out", "-o", help="Output directory")) -> None:
    """Write sample datasets with planted problems to try the gate on."""
    paths = write_demo(out)
    for name in paths:
        console.print(f"  wrote {out}/{name}")
    console.print(
        f"\nTry:\n  datagate check {out}/candidate_dirty.csv "
        f'-p "classify product reviews by sentiment" -r {out}/reference.csv')


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: Optional[int] = typer.Option(
        None, help="Port (defaults to the PORT env var, then 8000)"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
) -> None:
    """Run the DataGate web service and UI."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Service extras not installed.[/red] "
                      'Run: uv pip install -e ".[service]"')
        raise typer.Exit(code=1)

    # An explicit --port wins; otherwise honor $PORT, falling back to 8000.
    resolved_port = port if port is not None else int(os.environ.get("PORT", "8000"))
    console.print(f"[green]DataGate[/green] → http://{host}:{resolved_port}  "
                  f"(API docs at http://{host}:{resolved_port}/docs)")
    uvicorn.run("datagate.service.api:app", host=host, port=resolved_port, reload=reload)


@registry_app.command("add")
def registry_add(
    dataset: str = typer.Argument(..., help="Dataset to register"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Display name"),
    kind: str = typer.Option("internal", "--kind", "-k",
                             help=f"One of: {', '.join(VALID_KINDS)}"),
) -> None:
    """Register a known dataset so future checks can detect overlap with it."""
    if kind not in VALID_KINDS:
        raise typer.BadParameter(f"--kind must be one of: {', '.join(VALID_KINDS)}")
    entry = Registry().add(load_dataset(dataset), name=name, kind=kind)
    console.print(f"registered [bold]{entry.name}[/bold] ({entry.kind}, "
                  f"{entry.rows:,} rows, {entry.fingerprint[:12]})")


@registry_app.command("list")
def registry_list() -> None:
    """List registered datasets."""
    entries = Registry().entries()
    if not entries:
        console.print("[dim]registry is empty[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    for col in ("Name", "Kind", "Rows", "Fingerprint"):
        table.add_column(col)
    for e in entries:
        table.add_row(e.name, e.kind, f"{e.rows:,}", e.fingerprint[:12])
    console.print(table)


@registry_app.command("remove")
def registry_remove(name: str = typer.Argument(..., help="Name or fingerprint prefix")) -> None:
    """Remove a dataset from the registry."""
    removed = Registry().remove(name)
    console.print(f"removed {removed} {'entry' if removed == 1 else 'entries'}")


@keys_app.command("show")
def keys_show() -> None:
    """Print this installation's report-signing public key (creates one if needed)."""
    console.print(signing.public_key_b64())


if __name__ == "__main__":
    app()
