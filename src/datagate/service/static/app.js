// DataGate single-page app. Preact + htm, vendored locally: no build step and
// no third-party requests — the page works offline and the CSP stays strict.
import { html, render, useState, useEffect, useRef }
  from "/static/vendor/htm-preact-standalone.module.js";

const TASK_TYPES = ["classification", "regression", "generation", "retrieval", "unknown"];
const MODALITIES = ["tabular", "text", "image", "mixed", "unknown"];
const DATA_ACCEPT = ".csv,.tsv,.parquet,.json,.jsonl,.ndjson,.txt";
const TOTAL_CHECKS = 16;

/* ---------- helpers ---------- */
async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (res.status === 401 && !path.startsWith("/auth/")) {
    window.dispatchEvent(new Event("dg-signed-out"));   // session ended: show sign-in
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* non-JSON body */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}
const postJSON = (path, body, method = "POST") => api(path, {
  method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});
async function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  return api("/datasets", { method: "POST", body: form });
}
async function waitForJob(jobId, onTick) {
  for (;;) {
    const job = await api(`/analyses/${jobId}`);
    if (onTick) onTick(job);
    if (job.status === "succeeded" || job.status === "failed") return job;
    await new Promise((r) => setTimeout(r, 800));
  }
}

// State colour only where it carries meaning: a healthy check stays ink.
const tone = (s) => (s >= 0.8 ? "ink" : s >= 0.5 ? "review" : "block");
const fitTone = (s) => (s >= 0.75 ? "ink" : s >= 0.35 ? "review" : "block");
const toneColor = (t) => (t === "ink" ? "var(--ink)" : `var(--${t})`);
// Confidence is drawn as the notation's level: solid, hatched, dotted.
const level = (c) => (c >= 0.8 ? "lvl-solid" : c >= 0.6 ? "lvl-hatch" : "lvl-dot");
const kb = (b) => (b >= 1048576 ? (b / 1048576).toFixed(1) + " MB" : Math.max(1, Math.round(b / 1024)) + " KB");
const when = (iso) => new Date(iso).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
const go = (hash) => { window.location.hash = hash; };
const cap = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const worstFirst = (a, b) => (a.skipped - b.skipped) || (a.score - b.score);

function useRoute() {
  const [hash, setHash] = useState(window.location.hash || "#/");
  useEffect(() => {
    const on = () => { setHash(window.location.hash || "#/"); window.scrollTo(0, 0); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash.replace(/^#\/?/, "").split("/");
}

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try { const saved = localStorage.getItem("datagate-theme"); if (saved) return saved; } catch (e) { /* storage blocked */ }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("datagate-theme", theme); } catch (e) { /* ignore */ }
  }, [theme]);
  return [theme, () => setTheme((t) => (t === "dark" ? "light" : "dark"))];
}

/* ---------- icons: one stroke, one weight ---------- */
const icon = (body) => html`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
  stroke-width="1.5" stroke-linecap="square" aria-hidden="true">${body}</svg>`;
const Mark = html`<svg viewBox="0 0 26 30" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
  <path d="M13 1.5 24 5.4v8.2c0 6.6-4.6 11.1-11 14.9C6.6 24.7 2 20.2 2 13.6V5.4Z"/>
  <path d="M8 15.2l3.4 3.4L18.5 11.5"/></svg>`;
const I = {
  check: icon(html`<path d="M4 4h16v16H4z"/><path d="M8 12.5l2.6 2.6L16 9.7"/>`),
  history: icon(html`<path d="M9 3v18M15 3v18"/><path d="M9 6h6M9 11h6M9 16h6" />`),
  market: icon(html`<path d="M3 8h18M3 8l2-4h14l2 4M5 8v12h14V8"/><path d="M10 20v-6h4v6"/>`),
  sun: icon(html`<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>`),
  moon: icon(html`<path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z"/>`),
  down: icon(html`<path d="M12 4v11M7 10.5l5 5 5-5M4 20h16"/>`),
  file: icon(html`<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4"/>`),
  next: icon(html`<path d="M4 12h15M13 6l6 6-6 6"/>`),
  back: icon(html`<path d="M20 12H5M11 6l-6 6 6 6"/>`),
};
// The staff glyph: the double rule with one mark on it. Reserved for running the checks.
const Staff = html`<svg class="staff" viewBox="0 0 12 18" aria-hidden="true">
  <path d="M3.5 0v18M8.5 0v18" stroke="currentColor" stroke-width="1.3"/>
  <rect x="4.2" y="4" width="3.6" height="6" fill="currentColor"/></svg>`;
const DropGlyph = html`<svg class="glyph" viewBox="0 0 34 46" aria-hidden="true" fill="none">
  <path d="M14 0v46M20 0v46" stroke="currentColor" stroke-width="1.4"/>
  <rect x="21.5" y="8" width="8" height="12" stroke="currentColor" stroke-width="1.4"/>
  <rect x="4.5" y="22" width="8" height="16" fill="currentColor"/></svg>`;

/* ---------- shared pieces ---------- */
function Loading({ text = "Loading" }) {
  return html`<div class="loading" role="status"><span class="beat-loader"></span>${text}…</div>`;
}

function Stepper({ stage }) {
  const order = [["upload", "Upload"], ["spec", "Confirm the task"], ["running", "Run the checks"]];
  const current = order.findIndex(([k]) => k === stage);
  return html`<nav class="stepper" aria-label="Progress"><ol>
    ${order.map(([key, label], i) => html`<li key=${key}
        class=${i === current ? "now" : i < current ? "done" : ""}
        aria-current=${i === current ? "step" : null}>
      <span class="n">${i + 1}</span><span class="t">${label}</span></li>`)}
  </ol></nav>`;
}

function FieldLabel({ text, opt }) {
  return html`<span class="label flabel">${text}${opt ? html` <span class="opt">${opt}</span>` : null}</span>`;
}

function FileDrop({ file, onFile, title, hint, label }) {
  const [drag, setDrag] = useState(false);
  const input = useRef(null);
  return html`<div>
    <button type="button" class=${"drop " + (drag ? "drag " : "") + (file ? "filled" : "")}
      aria-label=${`${label}: ${file ? `${file.name} selected. Choose a different file` : "choose a file"}`}
      onClick=${() => input.current.click()}
      onDragOver=${(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave=${() => setDrag(false)}
      onDrop=${(e) => { e.preventDefault(); setDrag(false); if (e.dataTransfer.files[0]) onFile(e.dataTransfer.files[0]); }}>
      ${DropGlyph}
      <span><span class="name">${file ? file.name : title}</span>
        <span class="sub">${file ? `${kb(file.size)} · choose another file` : hint}</span></span>
    </button>
    <input ref=${input} type="file" class="hidden" accept=${DATA_ACCEPT} tabindex="-1"
      onChange=${(e) => { if (e.target.files[0]) onFile(e.target.files[0]); }} />
  </div>`;
}

/* An irreversible choice asks once, inline, and says what it is agreeing to. */
function Confirm({ prompt, confirmLabel, onConfirm, onCancel, danger, busy }) {
  return html`<div class="confirm" role="group" aria-label=${prompt}>
    <p>${prompt}</p>
    <div class="actions">
      <button class=${"btn small " + (danger ? "danger solid" : "primary")} disabled=${busy} onClick=${onConfirm}>${confirmLabel}</button>
      <button class="btn small" disabled=${busy} onClick=${onCancel}>Cancel</button>
    </div>
  </div>`;
}

/* ---------- check flow ---------- */
function UploadScreen({ initial, onReady, setError }) {
  const [file, setFile] = useState(initial.file || null);
  const [reference, setReference] = useState(initial.reference || null);
  const [objective, setObjective] = useState(initial.objective || "");
  const [brief, setBrief] = useState(initial.brief || "");
  const [busy, setBusy] = useState(false);
  const briefInput = useRef(null);

  function loadBrief(f) {
    const reader = new FileReader();
    reader.onload = () => setBrief(String(reader.result));
    reader.readAsText(f);
  }

  async function submit(e) {
    e.preventDefault();
    if (!file || !objective.trim()) return;
    setBusy(true); setError(null);
    try {
      const dataset = await uploadFile(file);
      const ref = reference ? await uploadFile(reference) : null;
      const spec = await postJSON("/specs", {
        description: objective, project_brief: brief.trim() || null, dataset_id: dataset.id,
      });
      onReady({ file, reference, objective, brief }, dataset, ref, spec);
    } catch (err) { setError(err.message); setBusy(false); }
  }

  return html`<form class="form" onSubmit=${submit}>
    <div class="drops">
      <div><${FieldLabel} text="Dataset to check" />
        <${FileDrop} file=${file} onFile=${setFile} label="Dataset to check" title="Drop a file or choose one"
          hint="CSV, TSV, Parquet, JSON, JSONL or text · up to 512 MB" /></div>
      <div><${FieldLabel} text="Your own task data" opt="optional" />
        <${FileDrop} file=${reference} onFile=${setReference} label="Your own task data" title="Add a sample of your data"
          hint="Ideally your eval set · unlocks the fit checks" /></div>
    </div>

    <label class="field"><${FieldLabel} text="What do you want to train?" />
      <textarea style=${{ minHeight: "64px" }} placeholder="e.g. classify product reviews by sentiment"
        value=${objective} onInput=${(e) => setObjective(e.target.value)}></textarea></label>

    <label class="field"><${FieldLabel} text="Project brief" opt="optional · the domain, where the model runs, how success is measured" />
      <textarea placeholder="e.g. an e-commerce marketplace; the model routes reviews in a real-time API; we track macro F1"
        value=${brief} onInput=${(e) => setBrief(e.target.value)}></textarea>
    </label>
    <div class="actions" style=${{ marginTop: "-10px", marginBottom: "24px" }}>
      <button type="button" class="linkbtn" onClick=${() => briefInput.current.click()}>Load the brief from a .md or .txt file</button>
      ${brief ? html`<span class="muted small">${brief.length.toLocaleString()} characters</span>` : null}
      <input ref=${briefInput} type="file" class="hidden" accept=".md,.txt,.markdown,.text" tabindex="-1"
        onChange=${(e) => { if (e.target.files[0]) loadBrief(e.target.files[0]); }} />
    </div>

    <div class="actions">
      <button class="btn primary" type="submit" disabled=${busy || !file || !objective.trim()}>
        ${busy ? html`<span class="spin"></span>Profiling` : html`Continue${I.next}`}</button>
      ${!reference ? html`<span class="muted small">Without your own data the full security scan still runs; fit stays unverified.</span>` : null}
    </div>
  </form>`;
}

function SpecScreen({ dataset, reference, specResponse, onStarted, onBack, setError }) {
  const [spec, setSpec] = useState(specResponse.spec);
  const [busy, setBusy] = useState(false);
  const columns = Object.keys(dataset.columns || {});
  const assisted = new Set(spec.llm_assisted_fields || []);
  const set = (field, value) => setSpec({ ...spec, [field]: value === "" ? null : value });
  const ai = (f) => (assisted.has(f) ? html`<span class="ai" title="Filled in by the language model">AI</span>` : null);
  const seen = dataset.seen_before || [];

  async function start() {
    setBusy(true); setError(null);
    try {
      await postJSON(`/specs/${specResponse.id}`, { spec }, "PATCH");
      onStarted(await postJSON("/analyses", {
        dataset_id: dataset.id, spec_id: specResponse.id,
        reference_dataset_id: reference ? reference.id : null,
      }));
    } catch (err) { setError(err.message); setBusy(false); }
  }

  return html`<div class="form">
    <p class="meta">${dataset.filename} · ${dataset.n_rows.toLocaleString()} rows × ${dataset.n_cols} columns · ${kb(dataset.size_bytes)} · ${dataset.format}
      ${reference ? html` · compared against ${reference.filename} (${reference.n_rows.toLocaleString()} rows)` : null}</p>
    ${seen.length ? html`<div class="notice warn" style=${{ marginTop: "14px" }}>Seen before: ${seen.map((m) =>
      `${Math.round(m.containment * 100)}% of rows match “${m.name}”${m.exact ? " (identical file)" : ""}`).join("; ")}.</div>` : null}
    <div class="cols" style=${{ marginTop: "18px" }}>
      ${columns.map((name) => { const c = dataset.columns[name]; return html`<div key=${name}>
        <b>${name}</b><span>${c.kind} · ${c.n_unique.toLocaleString()} distinct${c.null_count ? ` · ${c.null_count.toLocaleString()} missing` : ""}</span></div>`; })}
    </div>

    ${specResponse.needs_clarification ? html`<div class="notice warn"><b>The description left something open.</b>
      <ul>${spec.clarifying_questions.map((q) => html`<li key=${q}>${q}</li>`)}</ul></div>` : null}

    <div class="grid3">
      <label class="field"><span class="label flabel">Task type ${ai("task_type")}</span>
        <select value=${spec.task_type} onChange=${(e) => set("task_type", e.target.value)}>
          ${TASK_TYPES.map((t) => html`<option key=${t} value=${t}>${t}</option>`)}</select></label>
      <label class="field"><span class="label flabel">Modality ${ai("modality")}</span>
        <select value=${spec.modality} onChange=${(e) => set("modality", e.target.value)}>
          ${MODALITIES.map((m) => html`<option key=${m} value=${m}>${m}</option>`)}</select></label>
      <label class="field"><span class="label flabel">Target column ${ai("target_column")}</span>
        <select value=${spec.target_column || ""} onChange=${(e) => set("target_column", e.target.value)}>
          <option value="">None</option>
          ${columns.map((c) => html`<option key=${c} value=${c}>${c}</option>`)}</select></label>
      <label class="field"><span class="label flabel">Domain ${ai("domain")}</span>
        <input type="text" placeholder="e.g. e-commerce" value=${spec.domain || ""} onInput=${(e) => set("domain", e.target.value)} /></label>
      <label class="field"><span class="label flabel">Deployment ${ai("deployment_context")}</span>
        <input type="text" placeholder="e.g. real-time production" value=${spec.deployment_context || ""} onInput=${(e) => set("deployment_context", e.target.value)} /></label>
      <label class="field"><span class="label flabel">Success metric ${ai("success_metric")}</span>
        <input type="text" placeholder="e.g. macro F1" value=${spec.success_metric || ""} onInput=${(e) => set("success_metric", e.target.value)} /></label>
    </div>
    <div class="actions">
      <button class="btn primary" onClick=${start} disabled=${busy}>
        ${busy ? html`<span class="spin"></span>Starting` : html`${Staff}Run the 16 checks`}</button>
      <button class="btn" onClick=${onBack} disabled=${busy}>${I.back}Back</button>
    </div>
  </div>`;
}

/* The progress screen is the score being written: the same staff and the same
   16 named checks, each mark inked the moment its check finishes. */
function RunningScreen({ job }) {
  const [checks, setChecks] = useState(null);
  useEffect(() => { api("/meta").then((m) => setChecks(m.analyzers)).catch(() => setChecks([])); }, []);
  const p = job.progress || { done: 0, total: TOTAL_CHECKS };
  const total = p.total || TOTAL_CHECKS;
  const finished = new Set(p.finished || (p.last ? [p.last] : []));
  // Announce quarters, not every check: a screen reader shouldn't chatter.
  const quarter = Math.floor((p.done / total) * 4);
  const spoken = ["Starting the checks", "A quarter of the checks are done", "Half of the checks are done",
    "Three quarters of the checks are done", "All checks are done"][quarter];
  const lane = (family, side) => (checks || []).filter((c) => c.family === family).map((c, i) => {
    const done = finished.has(c.name);
    const name = html`<span class="nm">${c.dimension}</span>`;
    const cnt = html`<span class="cnt">${i + 1}</span>`;
    const mark = html`<span class="bars"><i class=${done ? "lvl-solid pmark on" : "pmark"}></i></span>`;
    const state = html`<span class="v state">${done ? "done" : ""}</span>`;
    return html`<div key=${c.name} class=${"mark static " + (done ? "is-done" : "")}>
      ${side === "l" ? [cnt, name, mark, state] : [state, mark, name, cnt]}</div>`;
  });
  return html`<div class="running">
    <p class="sr" role="status" aria-live="polite">${spoken}</p>
    <div class="running-head" aria-hidden="true">
      <span class="count">${p.done}</span><span class="of">of ${total} checks</span>
      <span class="lede">${job.status === "queued" ? "Queued — waiting for a worker." : p.last
        ? `Last finished: ${p.last.replace(/_/g, " ")}.` : "Loading the data and building the shared feature space."}</span>
    </div>
    ${checks && checks.length ? html`<div class="staff progress-staff" aria-hidden="true">
      <div class="lane l">${lane("security", "l")}</div><span class="gap"></span>
      <div class="lane r">${lane("fit", "r")}</div>
    </div>` : null}
    <p class="muted small running-foot">Seconds to a few minutes, depending on size. You can leave this page; the report stays in History.</p>
  </div>`;
}

function CheckFlow() {
  const [stage, setStage] = useState("upload");
  const [draft, setDraft] = useState({});
  const [dataset, setDataset] = useState(null);
  const [reference, setReference] = useState(null);
  const [specResponse, setSpecResponse] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);

  async function started(first) {
    setJob(first); setStage("running");
    try {
      const final = await waitForJob(first.id, setJob);
      if (final.status === "succeeded") go(`#/report/${final.report_id}`);
      else { setError(final.error || "The analysis failed."); setStage("spec"); }
    } catch (err) { setError(err.message); setStage("spec"); }
  }

  const titles = {
    upload: ["Check a dataset", "Find out whether it is safe to train on, and whether it helps your task."],
    spec: ["Confirm the task", "This is what DataGate understood. Every fit check depends on it."],
    running: ["Running the checks", null],
  };
  const [title, lede] = titles[stage];
  return html`<div>
    <h1 class="title">${title}</h1>
    ${lede ? html`<p class="lede">${lede}</p>` : null}
    <${Stepper} stage=${stage} />
    ${error ? html`<div class="notice error" role="alert"><b>Could not continue.</b> ${error}</div>` : null}
    ${stage === "upload" ? html`<${UploadScreen} initial=${draft} setError=${setError}
      onReady=${(d0, d, r, s) => { setDraft(d0); setDataset(d); setReference(r); setSpecResponse(s); setStage("spec"); }} />` : null}
    ${stage === "spec" ? html`<${SpecScreen} dataset=${dataset} reference=${reference}
      specResponse=${specResponse} setError=${setError} onStarted=${started}
      onBack=${() => { setError(null); setStage("upload"); }} />` : null}
    ${stage === "running" && job ? html`<${RunningScreen} job=${job} />` : null}
  </div>`;
}

/* ---------- report: the score ---------- */
/* Each mark's length is the shortfall from a perfect 1.00, so the problems are
   the long marks and a clean dataset reads as a quiet staff. */
function ScoreMark({ e, i, side, count, hot, setHot, onPick }) {
  const t = tone(e.score);
  const shortfall = 1 - e.score;
  const bar = e.skipped
    ? html`<em>${e.needs_reference ? "needs your data" : "not run"}</em>`
    : shortfall < 0.005
      ? html`<i class="clear" aria-hidden="true"></i>`
      : html`<i class=${level(e.confidence)} style=${{ width: `${shortfall * 100}%`, color: toneColor(t), "--i": i }}></i>`;
  const v = html`<span class="v" style=${{ color: e.skipped ? "var(--ink-3)" : toneColor(t) }}>
    ${e.skipped ? "—" : e.score.toFixed(2)}</span>`;
  const family = side === "l" ? "Safety" : "Fit";
  const name = e.dimension || e.analyzer;
  const label = `${family} check ${i + 1} of ${count}, ${name}: ${e.skipped ? "not run" :
    `score ${e.score.toFixed(2)}, confidence ${e.confidence.toFixed(2)}`}. ${e.summary}. Opens the evidence.`;
  const parts = side === "l"
    ? [html`<span class="cnt">${i + 1}</span>`, html`<span class="nm">${name}</span>`, html`<span class="bars">${bar}</span>`, v]
    : [v, html`<span class="bars">${bar}</span>`, html`<span class="nm">${name}</span>`, html`<span class="cnt">${i + 1}</span>`];
  return html`<button type="button" class=${"mark " + (hot === e.analyzer ? "on" : "")}
    aria-label=${label}
    onMouseEnter=${() => setHot(e.analyzer)} onMouseLeave=${() => setHot(null)}
    onFocus=${() => setHot(e.analyzer)} onBlur=${() => setHot(null)}
    onClick=${() => onPick(e.analyzer)}>${parts}</button>`;
}

function ScoreHead({ name, comp, t, side, ran, of, weak }) {
  // A composite built from less than half its checks is provisional, and says so.
  const provisional = ran < of / 2;
  // "Strong" never sits bare above a failing mark on its own side.
  const label = provisional ? `${ran} of ${of} checks ran — add your own data to score this`
    : weak && /^(strong|very likely|likely)/.test(comp.label)
      ? `${comp.label.split(" —")[0]} overall — ${weak === 1 ? "one weak check" : `${weak} weak checks`} below`
      : comp.label;
  const num = html`<span class=${"big " + (provisional ? "provisional" : "")}
    style=${{ color: provisional ? "var(--ink-3)" : toneColor(t) }}>${comp.score.toFixed(2)}</span>`;
  const what = html`<span class="what"><b>${name}${provisional ? " · provisional" : ""}</b><span>${label}</span>
    <span>confidence ${comp.confidence.toFixed(2)}</span></span>`;
  return html`<div class=${"side " + side}>${side === "l" ? [what, num] : [num, what]}</div>`;
}

/* A degenerate range is not a finding: say "no measurable gain" in plain words. */
function GainLine({ report }) {
  if (!report.predicted_gain_range) return null;
  const mg = report.evidence.find((e) => e.analyzer === "marginal_gain");
  const range = mg && mg.metrics && mg.metrics.predicted_gain_range;
  if (range && Math.abs(range[0]) < 0.0005 && Math.abs(range[1]) < 0.0005) {
    return html`<p class="gain">Adding this data to your sample: no measurable gain predicted (linear-proxy estimate on your task data).</p>`;
  }
  return html`<p class="gain">Adding this data to your own: <b>${report.predicted_gain_range}</b></p>`;
}

/* Reasons beside the verdict: the engine's, plus — on a pass or review — the
   worst marks on the score, so the page never shows a long red mark nobody mentions. */
function decisionReasons(r) {
  const reasons = [...r.decision.reasons];
  if (r.decision.verdict === "block") return reasons;
  const said = reasons.join(" ");
  const low = r.evidence.filter((e) => !e.skipped && e.score < 0.5).sort(worstFirst);
  const lowFit = low.some((e) => e.family === "fit");
  const out = reasons.map((x) => (lowFit && x.includes("(strong fit)")
    ? x.replace("(strong fit)", "(strong overall, with one weak fit check below)") : x));
  for (const e of low.slice(0, 3)) {
    const name = e.dimension || e.analyzer;
    if (!said.includes(name)) out.push(`Worth a look — ${name} scores ${e.score.toFixed(2)}: ${e.summary}`);
  }
  return out;
}

function Score({ report, onPick }) {
  const [hot, setHot] = useState(null);
  const [drawn, setDrawn] = useState(false);
  useEffect(() => { const t = requestAnimationFrame(() => setDrawn(true)); return () => cancelAnimationFrame(t); }, []);
  const sec = report.evidence.filter((e) => e.family === "security");
  const fit = report.evidence.filter((e) => e.family === "fit");
  const ran = (list) => list.filter((e) => !e.skipped).length;
  const weak = (list) => list.filter((e) => !e.skipped && e.score < 0.5).length;
  return html`<section class=${"score " + (drawn ? "drawn" : "")} aria-labelledby="score-h">
    <h2 id="score-h" class="sr">The score: every check, safety on the left and fit on the right. Long marks are problems.</h2>
    <div class="score-head">
      <${ScoreHead} name="Safety" comp=${report.safety} t=${tone(report.safety.score)} side="l" ran=${ran(sec)} of=${sec.length} weak=${weak(sec)} />
      <span class="gap"></span>
      <${ScoreHead} name="Fit" comp=${report.fit} t=${fitTone(report.fit.score)} side="r" ran=${ran(fit)} of=${fit.length} weak=${weak(fit)} />
    </div>
    <div class="staff">
      <div class="lane l">${sec.map((e, i) => html`<${ScoreMark} key=${e.analyzer} e=${e} i=${i} side="l"
        count=${sec.length} hot=${hot} setHot=${setHot} onPick=${onPick} />`)}</div>
      <span class="gap"></span>
      <div class="lane r">${fit.map((e, i) => html`<${ScoreMark} key=${e.analyzer} e=${e} i=${i} side="r"
        count=${fit.length} hot=${hot} setHot=${setHot} onPick=${onPick} />`)}</div>
    </div>
    <div class="score-legend" aria-label="How to read the score">
      <span>Mark length is how far a check falls short of 1.00 — long marks are problems</span>
      <span><i class="lvl-solid"></i>confident (≥ 0.8)</span>
      <span><i class="lvl-hatch"></i>0.6 – 0.8</span>
      <span><i class="lvl-dot"></i>tentative (under 0.6)</span>
    </div>
    <p class="legend-note">Confidence is how much evidence stands behind a score: how many rows a check could read and how reliable its method is.</p>
    <${GainLine} report=${report} />
  </section>`;
}

function EvidenceTable({ evidence, open, setOpen }) {
  const rows = [...evidence].sort(worstFirst);
  return html`<div class="tablewrap"><table>
    <thead><tr><th scope="col">Check</th><th scope="col">Side</th><th scope="col" class="num">Score</th>
      <th scope="col" class="num">Conf.</th><th scope="col">Finding</th></tr></thead>
    <tbody>${rows.map((e) => {
      const has = Object.keys(e.metrics || {}).length > 0;
      const t = tone(e.score);
      const toggle = () => has && setOpen(open === e.analyzer ? null : e.analyzer);
      const out = [html`<tr key=${e.analyzer} id=${"ev-" + e.analyzer}
          class=${(e.skipped ? "skipped " : "") + (has ? "clickable " : "") + (open === e.analyzer ? "on" : "")}
          onClick=${toggle} tabindex=${has ? 0 : null} aria-expanded=${has ? String(open === e.analyzer) : null}
          onKeyDown=${(k) => { if (k.key === "Enter" || k.key === " ") { k.preventDefault(); toggle(); } }}>
        <td><b>${e.dimension || e.analyzer}</b></td>
        <td class="muted">${e.family === "security" ? "Safety" : "Fit"}</td>
        <td class="num">${e.skipped ? "—" : html`<b style=${{ color: toneColor(t) }}>${e.score.toFixed(2)}</b>`}</td>
        <td class="num">${e.skipped ? "—" : e.confidence.toFixed(2)}</td>
        <td class="wrap">${e.summary}</td></tr>`];
      if (open === e.analyzer) {
        out.push(html`<tr key=${e.analyzer + "-m"} class="expand"><td colspan="5">
          <pre class="metrics">${JSON.stringify(e.metrics, null, 2)}</pre></td></tr>`);
      }
      return out;
    })}</tbody>
  </table></div>`;
}

function Quarantine({ reportId, report, canPreview }) {
  const [reason, setReason] = useState(null);
  const [limit, setLimit] = useState(25);
  const [previews, setPreviews] = useState({});
  const list = report.quarantine_list.filter(
    (f) => !reason || f.reason.split("; ").some((r) => r.split(":")[0] === reason));

  async function toggle(index) {
    if (!canPreview) return;
    if (previews[index]) { const next = { ...previews }; delete next[index]; setPreviews(next); return; }
    try {
      const res = await api(`/reports/${reportId}/rows?indices=${index}`);
      setPreviews({ ...previews, [index]: res.rows[0] || { note: "row not available" } });
    } catch (err) { setPreviews({ ...previews, [index]: { error: err.message } }); }
  }

  if (!report.quarantine_total) return html`<p class="meta">No rows were pulled aside.</p>`;
  return html`<div>
    <div class="segs" role="group" aria-label="Filter by reason">
      <button aria-pressed=${String(!reason)} onClick=${() => setReason(null)}>All <b>${report.quarantine_total.toLocaleString()}</b></button>
      ${Object.entries(report.quarantine_by_reason).map(([k, v]) => html`
        <button key=${k} aria-pressed=${String(reason === k)} onClick=${() => { setReason(k); setLimit(25); }}>
          ${k.replace(/_/g, " ")} <b>${v.toLocaleString()}</b></button>`)}
    </div>
    <div class="tablewrap"><table>
      <thead><tr><th scope="col" class="num">Row</th><th scope="col">Severity</th><th scope="col">Reason</th><th scope="col">Detail</th></tr></thead>
      <tbody>${list.slice(0, limit).map((f) => {
        const isOpen = !!previews[f.row_index];
        const rows = [html`<tr key=${f.row_index} class=${(canPreview ? "clickable " : "") + (isOpen ? "on" : "")}
            tabindex=${canPreview ? 0 : null} aria-expanded=${canPreview ? String(isOpen) : null}
            onClick=${() => toggle(f.row_index)}
            onKeyDown=${(k) => { if (k.key === "Enter" || k.key === " ") { k.preventDefault(); toggle(f.row_index); } }}>
          <td class="num">${f.row_index}</td>
          <td><span class=${"tag " + f.severity}>${f.severity}</span></td>
          <td class="mono">${f.reason}</td><td>${f.detail || ""}</td></tr>`];
        const p = previews[f.row_index];
        if (p) {
          rows.push(html`<tr key=${f.row_index + "-p"} class="expand"><td colspan="4"><dl class="rowdl">
            ${Object.entries(p).filter(([k]) => k !== "row_index").map(([k, v]) => html`
              <dt key=${k + "t"}>${k}</dt><dd key=${k + "d"}>${v === null ? "∅" : String(v)}</dd>`)}
          </dl></td></tr>`);
        }
        return rows;
      })}</tbody></table></div>
    <div class="actions" style=${{ marginTop: "14px" }}>
      ${list.length > limit ? html`<button class="btn small" onClick=${() => setLimit(limit + 50)}>Show 50 more</button>` : null}
      <span class="muted small">${Math.min(limit, list.length).toLocaleString()} of ${list.length.toLocaleString()} shown${
        report.quarantine_total > report.quarantine_list.length ? ` · the report embeds the worst ${report.quarantine_list.length.toLocaleString()}; the export has all of them` : ""}${
        canPreview ? " · select a row to see its contents" : ""}</span>
    </div>
  </div>`;
}

/* In a blind report the buyer sees how much is wrong, never which rows. */
function BlindCounts({ report }) {
  const entries = Object.entries(report.quarantine_by_reason);
  if (!report.quarantine_total) return html`<p class="meta">No rows were pulled aside.</p>`;
  return html`<div>
    <dl class="counts">${entries.map(([k, v]) => html`<div key=${k}><dt>${k.replace(/_/g, " ")}</dt><dd>${v.toLocaleString()}</dd></div>`)}</dl>
    <p class="muted small" style=${{ marginTop: "12px" }}>Counts only. Which rows these are stays with the seller until the deal is accepted.</p>
  </div>`;
}

function DealDecision({ data, onDone }) {
  const [confirm, setConfirm] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const verdict = data.report.decision.verdict;
  if (data.deal_state !== "evaluated") {
    return html`<p class="meta">This deal is <b>${data.deal_state || "closed"}</b>. <a href="#/market">Back to the marketplace</a></p>`;
  }
  async function decide(decision) {
    setBusy(true); setError(null);
    try { await postJSON(`/deals/${data.deal_id}/decision`, { decision }); onDone(); }
    catch (err) { setError(err.message); setBusy(false); }
  }
  return html`<div>
    ${error ? html`<div class="notice error" role="alert"><b>Could not record the decision.</b> ${error}</div>` : null}
    ${confirm ? html`<${Confirm} busy=${busy}
        prompt=${confirm === "accept"
          ? (verdict === "pass" ? "Accept this dataset? The seller is told you accepted." : `Accept a dataset rated ${verdict.toUpperCase()}? The seller is told you accepted.`)
          : "Decline this dataset? The seller can list it again."}
        confirmLabel=${confirm === "accept" ? "Yes, accept" : "Yes, decline"} danger=${confirm === "decline" || verdict !== "pass"}
        onConfirm=${() => decide(confirm)} onCancel=${() => setConfirm(null)} />`
      : html`<div class="actions">
          <button class="btn primary" onClick=${() => setConfirm("accept")}>Accept dataset</button>
          <button class="btn danger" onClick=${() => setConfirm("decline")}>Decline</button>
          <a class="linkbtn" href="#/market">Back to the marketplace</a>
        </div>`}
  </div>`;
}

function ReportView({ reportId }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [verified, setVerified] = useState(null);
  const [open, setOpen] = useState(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    setData(null); setError(null); setVerified(null); setOpen(null);
    api(`/reports/${reportId}`).then(setData).catch((e) => setError(e.message));
  }, [reportId, tick]);

  if (error) return html`<div class="notice error" role="alert"><b>Could not load the report.</b> ${error}</div>`;
  if (!data) return html`<${Loading} text="Loading the report" />`;

  const r = data.report;
  if (!r.decision) {
    return html`<div class="empty"><h1 class="title">${data.dataset_filename}</h1>
      <p class="lede">This report came from an earlier engine (${data.engine_version}) and has no gate decision. Run the check again to get one.</p>
      <div class="actions" style=${{ marginTop: "18px" }}><button class="btn primary" onClick=${() => go("#/")}>Check a dataset${I.next}</button></div></div>`;
  }
  const s = r.spec;
  const facts = [s.task_type, s.target_column && `target ${s.target_column}`, s.domain, s.deployment_context,
    s.success_metric && `metric ${s.success_metric}`].filter(Boolean);
  const exports = data.exports_available;
  const blind = data.blind;
  const d = r.decision;

  function pick(analyzer) {
    setOpen(analyzer);
    requestAnimationFrame(() => {
      const row = document.getElementById("ev-" + analyzer);
      if (row) { row.scrollIntoView({ behavior: "smooth", block: "center" }); row.focus({ preventScroll: true }); }
    });
  }
  function downloadJSON() {
    const blob = new Blob([JSON.stringify(r, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${data.dataset_filename}.datagate-report.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }
  async function verify() {
    try { setVerified(await postJSON("/reports/verify", r)); }
    catch (err) { setVerified({ ok: false, detail: err.message }); }
  }
  const verifyLine = verified ? html`<p class=${"verified " + (verified.ok ? "ok" : "bad")} role="status">
    ${verified.ok ? "Signature checks out: " : "Signature problem: "}${verified.detail}</p>` : null;

  return html`<article>
    <a class="skip" href="#evidence-h" onClick=${(e) => { e.preventDefault(); document.getElementById("evidence-h").scrollIntoView(); document.getElementById("evidence-h").focus(); }}>Skip to the evidence</a>
    <header class="reporthead">
      <div class="reporthead-id">
        <h1 class="title" style=${{ fontSize: "clamp(26px, 2.6vw, 34px)" }}>${data.dataset_filename}</h1>
        <p class="meta factline">${[`${r.dataset_rows.toLocaleString()} rows × ${r.dataset_columns} columns`, ...facts,
          r.reference_name && `against ${r.reference_name}`, when(data.created_at)].filter(Boolean)
          .map((f, i) => html`<span key=${i}>${f}</span>`)}</p>
      </div>
      ${exports ? html`
        <div class="actions downloads-wide">
          <a class="btn small" href=${`/reports/${reportId}/clean`} download>${I.down}Cleaned data</a>
          <a class="btn small" href=${`/reports/${reportId}/clean?redact_pii=true`} download>${I.down}Cleaned, PII masked</a>
          <a class="btn small" href=${`/reports/${reportId}/quarantine`} download>${I.down}Quarantined rows</a>
        </div>` : null}
    </header>

    <section class=${"decision " + d.verdict} aria-labelledby="verdict-h">
      <div class=${"verdict " + d.verdict} aria-hidden="true">${d.verdict}</div>
      <div>
        <h2 id="verdict-h" class="headline"><span class="sr">${cap(d.verdict)}. </span>${d.headline}</h2>
        <ul class="reasons">${decisionReasons(r).map((x) => html`<li key=${x}>${x}</li>`)}</ul>
        ${blind ? html`<div class="blindbox">
          <p>Blind evaluation: you see the verdict, the scores and how many rows each problem affects — not the rows themselves.</p>
          <div class="actions"><button class="btn small" onClick=${verify}>Verify the signature</button></div>
          ${verifyLine}
        </div>` : null}
        ${exports ? html`<details class="downloads-narrow">
          <summary class="btn small">${I.down}Downloads</summary>
          <div class="actions">
            <a class="btn small" href=${`/reports/${reportId}/clean`} download>Cleaned data</a>
            <a class="btn small" href=${`/reports/${reportId}/clean?redact_pii=true`} download>Cleaned, PII masked</a>
            <a class="btn small" href=${`/reports/${reportId}/quarantine`} download>Quarantined rows</a>
          </div>
        </details>` : null}
      </div>
    </section>

    <${Score} report=${r} onPick=${pick} />

    ${blind ? html`<div class="section-head"><h2>Your decision</h2><p>Accepting tells the seller; nothing is paid in this preview</p></div>
      <${DealDecision} data=${data} onDone=${() => setTick(tick + 1)} />`
      : r.recommendations.length ? html`<div class="section-head"><h2>What to do next</h2><p>Most serious first</p></div>
      <ol class="steps-next">${r.recommendations.map((x) => html`<li key=${x}>${x}</li>`)}</ol>` : null}

    <div class="section-head"><h2 id="evidence-h" tabindex="-1">Evidence</h2><p>Worst first · select a check for its raw measurements</p></div>
    <${EvidenceTable} evidence=${r.evidence} open=${open} setOpen=${setOpen} />

    <div class="section-head"><h2>Quarantine · ${r.quarantine_total.toLocaleString()} of ${r.dataset_rows.toLocaleString()} rows</h2>
      <p>${blind ? "Counts only in a blind evaluation" : exports ? "The cleaned dataset is everything else" : "The original data has been deleted; exports are unavailable"}</p></div>
    ${blind ? html`<${BlindCounts} report=${r} />` : html`<${Quarantine} reportId=${reportId} report=${r} canPreview=${exports} />`}

    <div class="section-head"><h2>Integrity</h2><p>Same inputs, seed and policy always give the same digest</p></div>
    <div class="integrity">
      <div class="digest"><b>Digest</b>${r.digest}${r.signature ? html`<br /><b>Signed</b>Ed25519 · ${r.signature.public_key}` : null}</div>
      <div class="actions">
        ${blind ? null : html`<button class="btn small" onClick=${verify}>Verify signature</button>`}
        <button class="btn small" onClick=${downloadJSON}>${I.down}JSON</button>
        <a class="btn small" href=${`/reports/${reportId}/markdown`} target="_blank" rel="noopener">${I.file}Markdown</a>
      </div>
    </div>
    ${blind ? null : verifyLine}
    <p class="disclaimer">Scores are calibrated estimates with confidence, not guarantees. A clean result lowers risk; it does not prove a dataset is safe. Engine ${r.engine_version} · seed ${r.seed}.</p>
  </article>`;
}

/* ---------- history ---------- */
function HistoryPage({ hosted }) {
  const [jobs, setJobs] = useState(null);
  const [error, setError] = useState(null);
  const [q, setQ] = useState("");
  const [only, setOnly] = useState("all");
  useEffect(() => { api("/analyses?limit=100").then(setJobs).catch((e) => setError(e.message)); }, []);
  const shown = (jobs || []).filter((j) =>
    (!q || (j.dataset_filename || "").toLowerCase().includes(q.toLowerCase())) &&
    (only === "all" || j.verdict === only));
  const counts = { all: (jobs || []).length };
  for (const v of ["block", "review", "pass"]) counts[v] = (jobs || []).filter((j) => j.verdict === v).length;
  const num = (v, t) => v == null ? html`<td class="num muted">—</td>`
    : html`<td class="num"><b style=${{ color: toneColor(t(v)) }}>${v.toFixed(2)}</b></td>`;
  return html`<div>
    <h1 class="title">History</h1>
    <p class="lede">${hosted ? "Every check you have run, newest first." : "Every check run on this machine, newest first."}</p>
    ${error ? html`<div class="notice error" role="alert" style=${{ marginTop: "24px" }}><b>Could not load history.</b> ${error}</div>` : null}
    ${!jobs && !error ? html`<${Loading} />` : null}
    ${jobs && jobs.length === 0 ? html`<div class="empty" style=${{ marginTop: "28px" }}>
      <p class="lede">Nothing checked yet. Reports land here once a check finishes.</p>
      <div class="actions" style=${{ marginTop: "18px" }}><button class="btn primary" onClick=${() => go("#/")}>Check a dataset${I.next}</button></div></div>` : null}
    ${jobs && jobs.length ? html`<div class="filters">
        <label class="search"><span class="sr">Search by file name</span>
          <input type="text" placeholder="Search by file name" value=${q} onInput=${(e) => setQ(e.target.value)} /></label>
        <div class="segs" role="group" aria-label="Filter by decision">
          ${["all", "block", "review", "pass"].map((v) => html`<button key=${v} aria-pressed=${String(only === v)}
            onClick=${() => setOnly(v)}>${v === "all" ? "All" : cap(v)} <b>${counts[v]}</b></button>`)}
        </div>
      </div>
      <div class="tablewrap"><table>
      <thead><tr><th scope="col">Dataset</th><th scope="col">Checked</th><th scope="col">Decision</th>
        <th scope="col" class="num">Safety</th><th scope="col" class="num">Fit</th></tr></thead>
      <tbody>${shown.map((j) => html`<tr key=${j.id} class=${j.report_id ? "clickable" : ""}
          onClick=${() => j.report_id && go(`#/report/${j.report_id}`)}>
        <td>${j.report_id ? html`<a class="rowlink" href=${`#/report/${j.report_id}`} onClick=${(e) => e.stopPropagation()}>${j.dataset_filename || "—"}</a>`
          : html`<b>${j.dataset_filename || "—"}</b>`}
          ${j.error ? html`<div class="t-block small">${j.error}</div>` : null}</td>
        <td class="muted">${when(j.created_at)}</td>
        <td><span class=${"tag " + (j.verdict || (j.status === "succeeded" ? "neutral" : j.status))}>${j.verdict || (j.status === "succeeded" ? "no decision" : j.status)}</span></td>
        ${num(j.safety_score, tone)}${num(j.fit_score, fitTone)}
      </tr>`)}</tbody></table></div>
      ${shown.length === 0 ? html`<p class="meta" style=${{ marginTop: "16px" }}>Nothing matches. <button class="linkbtn" onClick=${() => { setQ(""); setOnly("all"); }}>Clear the filters</button></p>` : null}` : null}
  </div>`;
}

/* ---------- marketplace preview ---------- */
function DealRow({ deal, onChanged, setError }) {
  const [open, setOpen] = useState(false);
  const [confirm, setConfirm] = useState(null);
  const [objective, setObjective] = useState("");
  const [reference, setReference] = useState(null);
  const [busy, setBusy] = useState(false);
  const canEvaluate = ["listed", "evaluated", "declined"].includes(deal.state);

  async function act(fn) {
    setBusy(true); setError(null);
    try { await fn(); await onChanged(); } catch (err) { setError(err.message); }
    setBusy(false); setConfirm(null);
  }
  const evaluate = () => act(async () => {
    const ref = reference ? await uploadFile(reference) : null;
    const job = await postJSON(`/deals/${deal.id}/evaluations`, {
      description: objective, reference_dataset_id: ref ? ref.id : null });
    await onChanged();
    const final = await waitForJob(job.id);
    if (final.status === "failed") throw new Error(final.error || "The evaluation failed.");
    setOpen(false);
  });
  const decide = (decision) => act(() => postJSON(`/deals/${deal.id}/decision`, { decision }));
  const withdraw = () => act(() => api(`/deals/${deal.id}/withdraw`, { method: "POST" }));
  const v = deal.verdict;

  const prompts = {
    accept: v && v !== "pass" ? `Accept a dataset rated ${v.toUpperCase()}? The seller is told you accepted.` : "Accept this dataset? The seller is told you accepted.",
    decline: "Decline this dataset? The seller can list it again.",
    withdraw: "Take this listing down? Buyers can no longer evaluate it.",
  };

  return html`<div class="deal">
    <div class="deal-top">
      <h3>${deal.title}</h3>
      <span class=${"tag " + deal.state}>${deal.state}</span>
    </div>
    <p class="desc">${deal.description || "No description."}</p>
    <div class="facts">
      <span><b>${deal.n_rows.toLocaleString()}</b> rows</span><span><b>${deal.format}</b></span>
      ${deal.asking_price ? html`<span>asking <b>${deal.asking_price}</b></span>` : null}
      ${Object.entries(deal.columns).map(([name, kind]) => html`<span key=${name}><b>${name}</b> ${kind}</span>`)}
    </div>
    ${v ? html`<div class="dealresult">
      <span class=${"tag " + v}>${v}</span>
      <span>Safety <b style=${{ color: toneColor(tone(deal.safety_score)) }}>${deal.safety_score.toFixed(2)}</b></span>
      <span>Fit <b style=${{ color: toneColor(fitTone(deal.fit_score)) }}>${deal.fit_score.toFixed(2)}</b></span>
      <span class="muted small">latest blind evaluation</span>
    </div>` : null}

    ${confirm ? html`<${Confirm} busy=${busy} prompt=${prompts[confirm]}
        confirmLabel=${{ accept: "Yes, accept", decline: "Yes, decline", withdraw: "Yes, take it down" }[confirm]}
        danger=${confirm !== "accept" || (v && v !== "pass")}
        onConfirm=${() => (confirm === "withdraw" ? withdraw() : decide(confirm))} onCancel=${() => setConfirm(null)} />`
      : html`<div class="dealactions">
        <div class="actions">
          ${deal.state === "evaluating" ? html`<span class="meta"><span class="spin"></span> Evaluating</span>` : null}
          ${deal.report_id ? html`<button class="btn small primary" onClick=${() => go(`#/report/${deal.report_id}/market`)}>Read the score-only report</button>` : null}
          ${canEvaluate ? html`<button class="btn small" disabled=${busy} onClick=${() => setOpen(!open)} aria-expanded=${String(open)}>
            ${deal.report_id ? "Evaluate again" : "Evaluate blind"}</button>` : null}
          ${deal.state === "evaluated" && deal.is_evaluator ? html`
            <button class="btn small" disabled=${busy} onClick=${() => setConfirm("accept")}>Accept</button>
            <button class="btn small danger" disabled=${busy} onClick=${() => setConfirm("decline")}>Decline</button>` : null}
        </div>
        ${canEvaluate && deal.is_seller ? html`<button class="linkbtn seller" disabled=${busy} onClick=${() => setConfirm("withdraw")}>Withdraw your listing</button>` : null}
      </div>`}

    ${open && canEvaluate ? html`<div class="evalform">
      <label class="field"><${FieldLabel} text="What would you train with it?" />
        <textarea style=${{ minHeight: "60px" }} value=${objective} placeholder="e.g. classify product reviews by sentiment"
          onInput=${(e) => setObjective(e.target.value)}></textarea></label>
      <${FieldLabel} text="Your own task data" opt="optional · stays private" />
      <${FileDrop} file=${reference} onFile=${setReference} label="Your own task data" title="Add a sample of your data" hint="Sharpens the fit estimate" />
      <div class="actions" style=${{ marginTop: "16px" }}>
        <button class="btn primary" disabled=${busy || !objective.trim()} onClick=${evaluate}>
          ${busy ? html`<span class="spin"></span>Evaluating` : html`${Staff}Run blind evaluation`}</button>
        <button class="btn" disabled=${busy} onClick=${() => setOpen(false)}>Cancel</button>
      </div>
    </div>` : null}
  </div>`;
}

function MarketPage({ hosted }) {
  const [deals, setDeals] = useState(null);
  const [error, setError] = useState(null);
  const [file, setFile] = useState(null);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [price, setPrice] = useState("");
  const [busy, setBusy] = useState(false);

  const load = () => api("/deals").then(setDeals).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  async function list(e) {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      const dataset = await uploadFile(file);
      await postJSON("/deals", { dataset_id: dataset.id, title, description, asking_price: price });
      setFile(null); setTitle(""); setDescription(""); setPrice("");
      await load();
    } catch (err) { setError(err.message); }
    setBusy(false);
  }

  return html`<div>
    <h1 class="title">Marketplace</h1>
    <p class="lede">A buyer vets a seller's dataset and gets a signed, score-only report — never the rows.</p>
    <div class="notice warn" style=${{ marginTop: "22px" }}><b>Preview.</b> ${hosted
      ? "There are no payments, identity checks or escrow. Buyers see the shape of your data and a score, never the rows."
      : "There are no payments, identity checks or escrow, and anyone using this installation can act as either party."}</div>
    ${error ? html`<div class="notice error" role="alert"><b>Something went wrong.</b> ${error}</div>` : null}

    <div class="section-head"><h2>Listed datasets</h2><p>Buyers see the shape of the data only</p></div>
    ${!deals ? html`<${Loading} />` : deals.length === 0
      ? html`<p class="meta">Nothing listed yet. List one below to try a blind evaluation.</p>`
      : deals.map((d) => html`<${DealRow} key=${d.id} deal=${d} onChanged=${load} setError=${setError} />`)}

    <div class="section-head"><h2>List a dataset</h2><p>${hosted ? "As the seller · buyers never receive the file" : "As the seller · the file stays on this machine"}</p></div>
    <form class="form" onSubmit=${list}>
      <${FileDrop} file=${file} onFile=${setFile} label="Dataset to list" title="Drop the dataset to list" hint="Any supported format" />
      <div style=${{ height: "20px" }}></div>
      <div class="grid2">
        <label class="field"><${FieldLabel} text="Title" />
          <input type="text" value=${title} onInput=${(e) => setTitle(e.target.value)} placeholder="e.g. Retail reviews, 2024" /></label>
        <label class="field"><${FieldLabel} text="Asking price" opt="display only" />
          <input type="text" value=${price} onInput=${(e) => setPrice(e.target.value)} placeholder="e.g. $400" /></label>
      </div>
      <label class="field"><${FieldLabel} text="Description" />
        <textarea style=${{ minHeight: "64px" }} value=${description} onInput=${(e) => setDescription(e.target.value)}></textarea></label>
      <button class="btn primary" type="submit" disabled=${busy || !file || !title.trim()}>
        ${busy ? html`<span class="spin"></span>Listing` : "List dataset"}</button>
    </form>
  </div>`;
}

/* ---------- accounts ---------- */
function AuthScreen({ onAuthed }) {
  const [mode, setMode] = useState("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const signup = mode === "signup";

  async function submit(e) {
    e.preventDefault();
    setBusy(true); setError(null);
    try {
      onAuthed(await postJSON(signup ? "/auth/signup" : "/auth/login", { email, password }));
      go("#/");
    } catch (err) { setError(err.message); }
    setBusy(false);
  }
  const switchMode = () => { setMode(signup ? "signin" : "signup"); setError(null); };

  return html`<div class="authwrap"><main class="authcard">
    <div class="brand">${Mark}<b>DATAGATE</b></div>
    <h1 class="title">${signup ? "Create your account" : "Sign in"}</h1>
    <p class="lede">${signup
      ? "Check any dataset before you train on it. Your uploads, reports and history stay on your account."
      : "Pick up where you left off: your datasets, reports and history."}</p>
    ${error ? html`<div class="notice error" role="alert" style=${{ marginTop: "22px" }}>${error}</div>` : null}
    <form onSubmit=${submit} style=${{ marginTop: "26px" }}>
      <label class="field"><${FieldLabel} text="Email" />
        <input type="email" name="email" autocomplete="email" required autofocus
          value=${email} onInput=${(e) => setEmail(e.target.value)} /></label>
      <label class="field"><${FieldLabel} text="Password" opt=${signup ? "8 characters or more" : null} />
        <input type="password" name="password" required minlength=${signup ? 8 : null} maxlength="128"
          autocomplete=${signup ? "new-password" : "current-password"}
          value=${password} onInput=${(e) => setPassword(e.target.value)} /></label>
      <button class="btn primary" type="submit" disabled=${busy || !email || !password}>
        ${busy ? html`<span class="spin"></span>${signup ? "Creating" : "Signing in"}` : html`${Staff}${signup ? "Create account" : "Sign in"}`}</button>
    </form>
    <p class="meta authswitch">${signup ? "Already have an account? " : "New to DataGate? "}
      <button class="linkbtn" type="button" onClick=${switchMode}>${signup ? "Sign in" : "Create an account"}</button></p>
  </main></div>`;
}

/* ---------- shell ---------- */
function App() {
  const [theme, toggleTheme] = useTheme();
  const [page, arg, from] = useRoute();
  const [me, setMe] = useState(undefined);   // undefined = still asking the server
  useEffect(() => {
    api("/auth/me").then(setMe).catch(() => setMe({ login_required: false, user: null }));
    const out = () => setMe((m) => (m && m.login_required ? { login_required: true, user: null } : m));
    window.addEventListener("dg-signed-out", out);
    return () => window.removeEventListener("dg-signed-out", out);
  }, []);
  async function signOut() {
    try { await api("/auth/logout", { method: "POST" }); } catch (e) { /* already signed out */ }
    setMe({ login_required: true, user: null });
    go("#/");
  }
  if (me === undefined) return html`<div class="authwrap"><${Loading} /></div>`;
  if (me.login_required && !me.user) return html`<${AuthScreen} onAuthed=${setMe} />`;
  const hosted = me.login_required;
  const tab = page === "market" || (page === "report" && from === "market") ? "market"
    : page === "history" || page === "report" ? "history" : "check";
  const nav = [["check", "#/", "Check", I.check], ["history", "#/history", "History", I.history],
    ["market", "#/market", "Marketplace", I.market]];

  return html`<div class="shell">
    <header class="rail">
      <div class="brand">${Mark}<b>DATAGATE</b></div>
      <nav class="navlist" aria-label="Main">
        ${nav.map(([key, hash, label, ic]) => html`<a key=${key} href=${hash} aria-current=${tab === key ? "page" : null}>
          ${ic}${label}${key === "market" ? html`<span class="navtag">preview</span>` : null}</a>`)}
      </nav>
      <div class="rail-foot">
        <button class="themebtn" onClick=${toggleTheme}
          aria-label=${theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}>
          ${theme === "dark" ? I.sun : I.moon}<span>${theme === "dark" ? "Light" : "Dark"}</span></button>
        ${hosted ? html`<div class="acct">
          <span class="acct-email" title=${me.user.email}>${me.user.email}</span>
          <button class="linkbtn acct-out" onClick=${signOut}>Sign out</button></div>`
          : html`<span class="ver">engine 0.2 · runs locally</span>`}
      </div>
    </header>
    <main class="main"><div class="page">
      ${page === "report" ? html`<${ReportView} reportId=${arg} />` : null}
      ${page !== "report" && tab === "check" ? html`<${CheckFlow} />` : null}
      ${page !== "report" && tab === "history" ? html`<${HistoryPage} hosted=${hosted} />` : null}
      ${page !== "report" && tab === "market" ? html`<${MarketPage} hosted=${hosted} />` : null}
    </div></main>
  </div>`;
}

render(html`<${App} />`, document.getElementById("root"));
