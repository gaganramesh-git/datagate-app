---
name: DataGate
description: A pre-ingestion gate for training data, drawn as a notated score on ruled paper.
colors:
  paper: "#fbfbf8"
  sheet: "#ffffff"
  ink: "#111111"
  ink-2: "#4a4a47"
  ink-3: "#6b6b67"
  rule: "#d4d4cf"
  rule-strong: "#111111"
  band: "#e8e8e3"
  rail: "#111111"
  rail-ink: "#f4f4f0"
  rail-ink-2: "#b9b9b3"
  rail-band: "#2a2a28"
  block: "#b42318"
  review: "#9a5b00"
  pass: "#1d6b3f"
  block-wash: "#fbeceb"
  review-wash: "#fbf1e0"
  pass-wash: "#e8f3ec"
  paper-dark: "#0f0f0e"
  sheet-dark: "#161615"
  ink-dark: "#f2f2ee"
  ink-2-dark: "#c2c2bc"
  ink-3-dark: "#9a9a94"
  rule-dark: "#33332f"
  band-dark: "#262624"
  rail-dark: "#1d1d1b"
  rail-ink-2-dark: "#b4b4ae"
  rail-band-dark: "#2c2c29"
  block-dark: "#f0766b"
  review-dark: "#e9b04a"
  pass-dark: "#63c487"
  block-wash-dark: "#2a1513"
  review-wash-dark: "#2a2010"
  pass-wash-dark: "#12261a"
typography:
  verdict:
    fontFamily: "Big Shoulders Display, Arial Narrow, sans-serif"
    fontSize: "clamp(76px, 10vw, 132px)"
    fontWeight: 800
    lineHeight: 0.82
    letterSpacing: "0.005em"
  display:
    fontFamily: "Big Shoulders Display, Arial Narrow, sans-serif"
    fontSize: "64px"
    fontWeight: 700
    lineHeight: 0.85
  headline:
    fontFamily: "Big Shoulders Display, Arial Narrow, sans-serif"
    fontSize: "clamp(30px, 3.4vw, 44px)"
    fontWeight: 700
    lineHeight: 1.02
    letterSpacing: "0.01em"
  title:
    fontFamily: "Hanken Grotesk, -apple-system, Segoe UI, sans-serif"
    fontSize: "clamp(22px, 2.2vw, 28px)"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.01em"
  numeral:
    fontFamily: "Big Shoulders Display, Arial Narrow, sans-serif"
    fontSize: "17px"
    fontWeight: 600
    lineHeight: 1
    fontFeature: "\"tnum\" 1"
  body:
    fontFamily: "Hanken Grotesk, -apple-system, Segoe UI, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.55
    fontFeature: "\"tnum\" 1, \"lnum\" 1"
  section-label:
    fontFamily: "Hanken Grotesk, -apple-system, Segoe UI, sans-serif"
    fontSize: "12px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.2em"
  label:
    fontFamily: "Hanken Grotesk, -apple-system, Segoe UI, sans-serif"
    fontSize: "11px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "0.18em"
  mono:
    fontFamily: "JetBrains Mono, ui-monospace, Menlo, monospace"
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: 1.6
rounded:
  none: "0px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "18px"
  lg: "26px"
  xl: "46px"
  page-x: "44px"
  page-x-narrow: "16px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "13px 20px 13px 14px"
  button-primary-hover:
    backgroundColor: "{colors.ink-2}"
    textColor: "{colors.paper}"
  button-secondary:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "10px 16px"
  button-secondary-hover:
    backgroundColor: "{colors.band}"
    textColor: "{colors.ink}"
  button-danger:
    backgroundColor: "transparent"
    textColor: "{colors.block}"
    rounded: "{rounded.none}"
    padding: "10px 16px"
  button-danger-hover:
    backgroundColor: "{colors.block-wash}"
    textColor: "{colors.block}"
  button-danger-solid:
    backgroundColor: "{colors.block}"
    textColor: "{colors.paper}"
    rounded: "{rounded.none}"
    padding: "8px 12px"
  tag:
    backgroundColor: "transparent"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "5px 7px"
  input:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.none}"
    padding: "11px 12px"
  nav-item:
    backgroundColor: "transparent"
    textColor: "{colors.rail-ink-2}"
    padding: "0 10px"
    height: "44px"
  nav-item-hover:
    backgroundColor: "{colors.rail-band}"
    textColor: "{colors.rail-ink}"
  nav-item-active:
    backgroundColor: "{colors.rail-ink}"
    textColor: "{colors.rail}"
  score-mark:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    height: "36px"
  score-mark-hover:
    backgroundColor: "{colors.band}"
  confirm:
    backgroundColor: "{colors.sheet}"
    rounded: "{rounded.none}"
    padding: "14px 16px"
---

# Design System: DataGate

## Overview

**Creative North Star: "The Score Paper"**

DataGate's report is a notated score of an inspection, not a dashboard. The world is Labanotation score paper: an off-white ground ruled in ink, an ink-black rail down the left edge, and a single staff on which every check is one measured mark. Security checks grow left from a central double rule, fit checks grow right, and each mark's length is the shortfall from a perfect 1.00, so problems are the long marks and a clean dataset reads as a quiet staff. Confidence is written the way the notation writes level: solid, hatched or dotted fills.

The world lends type, palette, density and that one move. Layout, navigation and controls stay the ones a working tool should have: a sticky rail, ruled tables, plain forms, a stepper. Density is high and deliberate; the report is read closely, so rows are tight (36px marks, 11-12px cell padding) and numbers sit in tabular figures everywhere. Colour is ink by default; red, amber and green appear only where a state carries meaning.

The system rejects the category default it was built against: stat cards, score rings, grids of same-size cards, and rounded pills. Depth comes from rules and tone, never from lifted shadows.

**Key Characteristics:**
- Ink on off-white paper, inverting to paper on ink in dark mode, with the rail lifted so it still reads.
- Tall condensed numerals and verdicts against a quiet grotesque for running text.
- Wide-tracked uppercase labels for every heading, tag, button and column head.
- Square corners and 1px rules throughout; a 3px ink rule marks the decision.
- One signature component, the score staff, reused as the progress screen.
- Confidence drawn as fill texture, state drawn as colour, magnitude drawn as length.

## Colors

A monochrome ink-and-paper palette with three state colours held in reserve.

### Primary
- **Score Ink** (ink): All text, rules that carry structure, the primary button, the selected segment, the verdict rule and every healthy mark. It is the voice of the system; nearly everything that is not a state is ink.

### Secondary
- **Block Red** (block): BLOCK verdicts, failing checks (safety under 0.5, fit under 0.35), critical and failed tags, error notices and destructive actions.
- **Review Amber** (review): REVIEW verdicts, borderline checks (safety 0.5-0.8, fit 0.35-0.75), warning tags and warning notices.
- **Pass Green** (pass): PASS verdicts, accepted tags and verified-signature text. Healthy individual marks do not turn green; they stay ink.
- **Washes** (block-wash, review-wash, pass-wash): Pale grounds behind error and warning notices and under the danger button's hover. Never used as decoration.

### Neutral
- **Score Paper** (paper): The page ground, and the text colour on ink buttons.
- **Clean Sheet** (sheet): Raised-in-tone surfaces: inputs, drop zones, the confirm box, notices, expanded table rows, the evaluation form.
- **Secondary Ink** (ink-2): Secondary text, table heads, labels, ledes. 8.6:1 on paper.
- **Tertiary Ink** (ink-3): Metadata, count numerals in the score margins, placeholders, skipped rows, the clear-check tick. 5.2:1 on paper.
- **Hairline** (rule): Row dividers, the hairline track under every mark, input side borders.
- **Strong Rule** (rule-strong): Section-head underlines, table tops, the confirm and notice frames.
- **Beat Band** (band): Hover and selection fill: the hovered mark, the current table row, secondary button hover.
- **Ink Rail** (rail, rail-ink, rail-ink-2, rail-band): The shell's left rail, its text, its quieter text (9:1) and its hover band.

Dark theme (`[data-theme="dark"]`, toggled from the rail and remembered per browser) inverts paper and ink, brightens the three state colours so they hold contrast on ink, and lifts the rail to its own tone (rail-dark) so the shell still separates from the page.

### Named Rules
**The Ink-First Rule.** A healthy check is ink. State colour appears only when a score crosses a threshold or a verdict, tag or notice needs it; a screen with nothing wrong is black and white.

**The Wash-Behind Rule.** A wash only ever sits behind its own state's ink (an error notice, a danger hover). It never tints a section for atmosphere.

## Typography

**Display Font:** Big Shoulders Display (with Arial Narrow fallback), self-hosted, variable 300-800
**Body Font:** Hanken Grotesk (with -apple-system, Segoe UI fallback), self-hosted, 400-700
**Label/Mono Font:** JetBrains Mono (with ui-monospace, Menlo fallback), self-hosted, 400-500

**Character:** A tall condensed poster face for the words and numbers that decide things, set against a calm humanist grotesque that does all the reading work. Mono is a utility voice for digests, IDs and raw metrics, never a costume.

### Hierarchy
- **Verdict** (800, clamp 76-132px, 0.82, uppercase): The single PASS / REVIEW / BLOCK word at the head of a report, in its state colour.
- **Display** (700, 64px, 0.85; 48px under 860px): The safety and fit composites at the head of the score. The running count uses the same face at clamp 64-112px.
- **Headline** (700, clamp 30-44px, 1.02, uppercase): Page titles. Marketplace deal names use the same face at 24px.
- **Title** (Hanken 700, clamp 22-28px, 1.2, -0.01em, balanced): The decision's one-sentence headline beside the verdict.
- **Numeral** (600, 17px): Every mark's score value; counts in the counts grid at 26px; next-step and stepper ordinals at 30px in ink-3.
- **Body** (400, 15px, 1.55, tabular lining figures): All running text. Reasons cap at 78ch, ledes at 68ch, next steps at 92ch.
- **Section label** (Hanken 700, 12px, 0.2em, uppercase): Section heads, score-side names, the brand wordmark (13px, 0.24em).
- **Label** (Hanken 700, 11px, 0.14-0.18em, uppercase): Field labels, table heads, tags, buttons (12px), rail nav (12px, 600).
- **Mono** (400, 12.5-13px): Digests, row metrics, definition-list keys, the engine version.

### Named Rules
**The Condensed-Is-A-Decision Rule.** Big Shoulders Display sets verdicts, composites, counts, ordinals, score values and page titles. It never sets a sentence.

**The Tracked-Caps Rule.** Every structural label (section heads, tags, buttons, column heads, nav) is Hanken 600-700, uppercase, tracked 0.1-0.24em. Lowercase sentences carry content; tracked caps carry structure.

**The 11px Floor Rule.** No text renders below 11px. Labels, tags and the nav tag sit exactly on the floor; nothing is shrunk under it to fit.

## Layout

A two-column shell: a 228px sticky ink rail and a main column whose page is capped at 1240px with 34px top, 44px side and 96px bottom padding. Sections are separated by a section head (a label left, a muted note right, a strong rule beneath) with 46px above and 18px below. Vertical rhythm otherwise steps through roughly 8, 12, 18, 26 and 46px; it is an observed cadence, not a formal scale.

The report reads top to bottom: identity and downloads, the decision (verdict column at least 200px beside headline and reasons, 38px gutter), the score staff across the full content width, next steps, evidence, quarantine, integrity. The staff is a three-column grid: two equal lanes around a 40px gutter that holds the double rule.

Responsive behaviour, in order of width:
- **1100px:** The report head stacks; downloads wrap under the dataset name.
- **900px:** The rail becomes a sticky top bar: mark, horizontal nav without icons, a 44px icon-only theme toggle. Page padding drops to 22px / 16px / 72px.
- **860px:** The staff stacks into one column; both lanes share one grid (22px count, 148px name, flexible bar, 46px value) so they read as one score, and the fit lane is set off by an ink rule.
- **760px:** Downloads collapse into one Downloads disclosure; the decision and integrity grids go single-column.
- **720px:** Two- and three-column form grids and the drop zones go single-column.
- **640px:** The stepper keeps its numerals and drops its step names.

## Elevation & Depth

Flat. There are no drop shadows anywhere. Depth is conveyed by three means: the paper/sheet tonal step (inputs, notices, confirm boxes and expanded rows sit on the whiter sheet), ruled frames (1px hairline or strong ink rule), and weight of rule (a 3px ink rule over the decision, a 3px rule over the current stepper step, dashed for completed steps). Inset `box-shadow` appears only as a stroke: the 1.5px outline of a level fill, the 1px track of a progress mark, and the input focus underline. It never lifts.

### Named Rules
**The Ruled-Not-Raised Rule.** If a surface needs to stand apart, rule it or step its tone; never float it.

## Shapes

Square everything: zero radius on buttons, inputs, tags, notices, the confirm box, drop zones and table cells. Tags are ruled squares with a 7px square swatch, never pills. Selects draw their chevron from two gradient triangles rather than an icon. The one circle in the system is the 12px in-button spinner. The recurring silhouette is the staff: a pair of parallel 1.5px ink rules (the central double rule, the beat loader, the staff glyph in the Run button).

### Named Rules
**The Square Corner Rule.** Radius is 0. The busy spinner is the only round form, and it is not a precedent.

## Components

### Buttons
Tactile and literal: ruled rectangles with tracked caps.
- **Shape:** Square (0px), 1px ink border.
- **Primary:** Ink fill, paper text, 13px 20px 13px 14px padding, 12px Hanken 700 caps at 0.14em. Carries an inline 15px stroke icon or the staff glyph.
- **Secondary:** Transparent with ink border and text, 10px 16px; small variant 8px 12px at 11px.
- **Danger:** Block-red text and border; hover lays the block wash. The solid variant (red fill) is reserved for the confirming click of a destructive or non-pass decision.
- **Hover / Focus:** Background crossfades over 160ms on the system ease; secondary hover lays the beat band, primary hover lightens to ink-2. Focus is a 2px ink outline offset 2px. Disabled drops to 40% opacity.
- **Link button:** Underlined 13px 600 text; hover thickens the underline to 2px.

### Tags
- **Style:** Transparent, 1px border in currentColor, 11px Hanken 700 caps at 0.14em, 5px 7px padding, a 7px square swatch before the word.
- **State:** Colour comes from state (block, review, pass, or ink-2 for neutral and lifecycle states). The swatch encodes severity as fill: solid for critical, dotted for warning, outline for info.

### Cards / Containers
There are no cards. Grouped content is ruled, not boxed.
- **Confirm:** An inline sheet box with a strong rule frame, 14px 16px padding, max 640px, a 600-weight question and the action row. It replaces a modal; destructive actions always pass through it.
- **Notice:** Strong-rule frame on sheet, 13px 16px; error and warn variants swap to state rule and wash.
- **Drop zone:** Sheet ground, 1px dashed ink-3 frame, 112px min height; on drag it goes solid ink with a 12% dot texture; filled goes solid.
- **Blind box / evaluation form:** Strong-rule sheet frames, 14-20px padding.

### Inputs / Fields
- **Style:** Sheet ground, 1px hairline sides with an ink bottom border (a ruled line to write on), 0 radius, 11px 12px padding, 15px body text.
- **Focus:** All sides go ink and an inset 1px ink underline doubles the bottom rule; no glow.
- **Labels:** Tracked-caps label above with 8px gap; optional hints in lowercase ink-3.

### Navigation
- **Rail:** Ink ground, 26px shield mark and tracked wordmark, then nav items: 44px tall, 12px Hanken 600 caps at 0.16em, 16px stroke icon, rail-ink-2 text. Hover lays the rail band; the current page inverts to a paper block with ink text. A ruled theme toggle and the mono engine version sit at the foot.
- **Under 900px:** The rail becomes a horizontal top bar; icons and the version drop, the toggle becomes a 44px square.
- **Segmented filters:** A strong-ruled strip of 12px buttons divided by hairlines; the pressed segment inverts to ink, counts in tabular figures.
- **Stepper:** Three columns under a strong rule; 30px condensed ordinals; the current step carries a 3px ink bar, completed steps a 3px dashed bar.

### Tables
Strong rule on top, hairline between rows, 11px tracked-caps heads in ink-2, 11px 12px cells, numerals right-aligned in tabular figures. Clickable rows and the open row lay the beat band; skipped rows fall to ink-3. Expanded rows sit on sheet with mono metrics. Inline bars are a 64px hairline-framed 8px track.

### The Score (signature)
The staff: two lanes of 8 marks around a central double rule (two 1.5px ink rules, 14px apart, overshooting the lanes by 8px). Each mark is a 36px button row: count numeral in ink-3, check name (13.5px, ellipsised), a bar over a hairline track, and the score value in condensed numerals. Security lanes read right-to-left toward the rule; fit lanes read left-to-right from it.
- **Length:** The bar's width is the shortfall from 1.00 (minimum 3px). A check that is fine leaves only a 7px ink-3 tick on the hairline. A check that did not run shows an italic ink-3 note ("not run", "needs your data") and a dash for its value.
- **Colour:** Ink unless the score crosses a state threshold; value text matches.
- **Level fill:** Confidence ≥ 0.8 is solid; 0.6-0.8 is hatched (1.6px stripes at 135°, 5px period, 1.5px outline); under 0.6 is dotted (5px dot grid, 1.5px outline). A legend under the staff always names the three levels.
- **Head:** Each side's composite in 64px display numerals with its name, a summary and its confidence. A composite built from fewer than half its checks is drawn as an ink-3 outline (1.5px text stroke) and labelled provisional.
- **Interaction:** Hover or focus lays the beat band on that one mark and bolds its name; activating it opens and focuses that check's evidence row. Each mark carries a full spoken label.
- **Motion:** On first render the bars draw out from the staff (scaleX, 900ms, system ease, 45ms stagger per mark), only when reduced motion is not requested.
- **Progress variant:** The running screen reuses the staff with the 16 named checks: unfinished marks are 44px hairline outlines on paper, and each inks solid (500ms) the moment its check finishes, its name darkening from ink-3 to ink. A large condensed count sits at the head; screen readers hear quarter-way announcements only.

### Named Rules
**The Shortfall Rule.** Mark length is always distance from perfect, never the score itself. Long marks are problems; a clean staff is quiet.

**The Level Fill Rule.** Confidence is fill texture (solid, hatched, dotted) and nothing else. Hatching and dots belong to the level encoding and the tag swatches that echo it; they are not surface decoration.

**The One-Mark Band Rule.** The beat band covers only the mark under the pointer or focus. Never band across lanes: security check N and fit check N are unrelated.

### Motion
One ease for everything, `cubic-bezier(0.16, 1, 0.3, 1)`. State changes run 160ms (background, border, colour). The score draws at 900ms with a 45ms stagger; progress marks ink at 500ms; names darken over 400ms. The loading state is a beat loader: a block rising through a 44px double rule every 1.4s. Under `prefers-reduced-motion: reduce` every animation and transition collapses to 1ms.

## Do's and Don'ts

### Do:
- **Do** keep healthy checks, composites and running text in ink; reach for block, review or pass only at the thresholds the score defines.
- **Do** draw every new measured quantity on the staff as a shortfall mark with a level fill, and keep the three-level legend beside it.
- **Do** separate surfaces with 1px rules (hairline #d4d4cf or strong ink) and the paper-to-sheet tone step.
- **Do** set structural labels in Hanken Grotesk 700 uppercase, tracked 0.14-0.2em, at 11-12px.
- **Do** set verdicts, composites and numerals in Big Shoulders Display with tabular figures.
- **Do** route destructive or non-pass decisions through the inline confirm box with a solid danger button.
- **Do** give every focusable element the 2px ink focus outline at 2px offset, and keep rail targets at 44px.
- **Do** hold all text at or above 11px and honour reduced motion with the 1ms collapse.

### Don't:
- **Don't** build stat cards, score rings, gauges or grids of same-size cards; the score is a staff.
- **Don't** round corners or turn tags into pills.
- **Don't** add drop shadows or floating elevation; inset strokes only.
- **Don't** scale a mark by the score itself, or colour a healthy mark green.
- **Don't** use hatched or dotted textures as backgrounds or ornament outside the level encoding.
- **Don't** lay the beat band across both lanes.
- **Don't** set sentences in the condensed display face, or set the interface in monospace.
- **Don't** load fonts or assets from third parties; all type is self-hosted under the strict CSP.
