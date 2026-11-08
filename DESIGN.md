---
name: snakePPO dashboard
description: A localhost, read-only research dashboard for watching PPO runs and judging switches across seeds.
colors:
  bg: "#f4f5f7"
  surface: "#ffffff"
  surface-2: "#eceef2"
  surface-3: "#e1e4ea"
  ink: "#14171d"
  ink-2: "#4c5361"
  line: "#d8dce3"
  line-strong: "#b9c0cb"
  accent: "#2456c4"
  accent-hover: "#1c47a6"
  accent-ink: "#ffffff"
  accent-soft: "#e5ecfa"
  focus: "#2456c4"
  ok: "#19703f"
  warn: "#8f5200"
  bad: "#b3261e"
  info: "#2456c4"
  neutral: "#59606d"
  skeleton: "#e6e9ee"
  skeleton-shine: "#f1f3f6"
  chart-paper: "#ffffff"
  chart-plot: "#ffffff"
  chart-grid: "#eceef2"
  chart-axis: "#b9c0cb"
  chart-font: "#4c5361"
  chart-title: "#14171d"
  series-1: "#2f6fde"
  series-2: "#d9612b"
  series-3: "#16926b"
  series-4: "#9a46cf"
  series-5: "#b9860f"
  series-6: "#cf3f74"
  series-7: "#2f93b8"
  series-8: "#6b717d"
  dark-bg: "#0e1014"
  dark-surface: "#161920"
  dark-surface-2: "#1b1f27"
  dark-surface-3: "#262b35"
  dark-ink: "#e6e8ed"
  dark-ink-2: "#a4acba"
  dark-line: "#2a2f39"
  dark-line-strong: "#3d4450"
  dark-accent: "#7ea4ff"
  dark-accent-hover: "#9cb9ff"
  dark-accent-ink: "#0e1014"
  dark-accent-soft: "#1c2945"
  dark-ok: "#55c98f"
  dark-warn: "#e8a64a"
  dark-bad: "#f47c72"
  dark-neutral: "#a4acba"
  dark-skeleton: "#20242d"
  dark-skeleton-shine: "#282d37"
  dark-chart-grid: "#232832"
  dark-series-1: "#6f9bff"
  dark-series-2: "#ff8a57"
  dark-series-3: "#3fc49a"
  dark-series-4: "#c07cf0"
  dark-series-5: "#e6b84a"
  dark-series-6: "#f06d9b"
  dark-series-7: "#62bfe0"
  dark-series-8: "#a3a9b5"
typography:
  headline:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.3125rem"
    fontWeight: 650
    lineHeight: 1.25
    letterSpacing: "-0.01em"
  title:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "1.125rem"
    fontWeight: 650
    lineHeight: 1.25
    letterSpacing: "-0.01em"
  title-sm:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "1rem"
    fontWeight: 650
    lineHeight: 1.25
  body:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    lineHeight: 1.5
  body-sm:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 400
    lineHeight: 1.5
    fontFeature: "tnum"
  label:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 600
    lineHeight: 1.5
  code:
    fontFamily: "ui-monospace, Cascadia Mono, Segoe UI Mono, Consolas, monospace"
    fontSize: "0.92em"
rounded:
  sm: "4px"
  md: "6px"
  lg: "10px"
  pill: "999px"
spacing:
  xs: "0.25rem"
  sm: "0.5rem"
  md: "0.75rem"
  inset: "0.85rem"
  lg: "1rem"
  xl: "1.25rem"
  section: "1.75rem"
components:
  button:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.md}"
    padding: "0.3rem 0.75rem"
    height: "2rem"
  button-hover:
    backgroundColor: "{colors.surface-3}"
  button-active:
    backgroundColor: "{colors.line}"
  button-quiet:
    backgroundColor: "transparent"
    textColor: "{colors.accent}"
    padding: "0.3rem 0.4rem"
  button-quiet-hover:
    backgroundColor: "{colors.accent-soft}"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.md}"
    padding: "0.3rem 0.55rem"
    height: "2rem"
  field-label:
    textColor: "{colors.ink-2}"
    typography: "{typography.label}"
  segmented-option:
    textColor: "{colors.ink-2}"
    rounded: "{rounded.sm}"
    padding: "0.2rem 0.6rem"
  segmented-option-selected:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.accent-ink}"
  tab:
    backgroundColor: "transparent"
    textColor: "{colors.ink-2}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: "0.3rem 0.85rem"
    height: "2rem"
  tab-hover:
    backgroundColor: "{colors.surface-3}"
    textColor: "{colors.ink}"
  tab-selected:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
  run-row:
    rounded: "{rounded.md}"
    padding: "0.45rem 0.45rem 0.5rem"
  run-row-hover:
    backgroundColor: "{colors.surface-3}"
  run-row-selected:
    backgroundColor: "{colors.accent-soft}"
  chip:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "0.05rem 0.5rem"
  chip-shared:
    backgroundColor: "transparent"
    textColor: "{colors.ink-2}"
  badge:
    typography: "{typography.label}"
    rounded: "{rounded.pill}"
    padding: "0.05rem 0.45rem"
  card:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.lg}"
    padding: "1rem 1.1rem 0.75rem"
  table-header:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.ink-2}"
    typography: "{typography.label}"
    padding: "0.45rem 0.7rem"
  progress:
    backgroundColor: "{colors.line}"
    rounded: "{rounded.pill}"
    height: "0.3rem"
---

# Design System: snakePPO dashboard

## Overview

**Creative North Star: "The Calm Readout"**

The dashboard is a working instrument for one researcher who keeps it open beside the terminal while a run trains. Everything about the system steps back so the curves can speak: cool grey neutrals in three surface tiers, one blue accent for selection and action, a separate eight-hue palette reserved for data series, and a small set of state hues (green, amber, red, grey) that only appear where they carry a verdict or a run status. It is dense in the way good operator tools are dense: 13px controls, 2rem hit targets, a fixed sidebar of runs and a scrolling workspace of chart cards.

The form is the standard internal-tool shell, on purpose: sticky header, a sidebar list with filters, tabs over the workspace, cards of Plotly charts, plain tables. The character lives in restraint and in the honesty of the states: a dashed empty box that names the exact CLI command to run, a skeleton where a chart is still loading, an amber callout when a run looks stalled, a red banner when the server cannot be reached. Light and dark are both first-class and follow the system unless the researcher pins one; every chart colour is read from the same tokens, so the charts change theme with the page.

This record is derived from the shipped build (commit `4dfe3c6`). The North Star name is the documenter's provisional label and has not been confirmed by the user.

**Key Characteristics:**
- Three cool-grey surface tiers plus white cards; depth comes mostly from tone and hairlines.
- One accent blue for action, selection and focus; status hues only for status.
- A dedicated eight-colour series palette, tuned separately for light and dark.
- Platform UI sans throughout; monospace only for code and commands.
- Every view has a designed loading, empty, error and stale state.

## Colors

A cool, low-chroma grey frame with one saturated blue and a separate set of data and state hues; each colour has a light and a dark value, and the dark theme is a full re-tuning, not an inversion.

### Primary
- **Bench Blue** (`accent`; dark `dark-accent`): the one action and selection colour. Quiet buttons, links, the checked segment of a segmented control, the fill of a progress bar while a run is running, the caret, native checkbox accent, and the focus ring (`focus` is the same value). Its hover deepens to `accent-hover` in light and brightens to `dark-accent-hover` in dark.
- **Blue Wash** (`accent-soft`; dark `dark-accent-soft`): the selected run row, quiet-button hover, and text selection. Never used as a text colour.

### Neutral
- **Desk Grey** (`bg`): the page ground behind the workspace and the sticky workspace bar.
- **Card White** (`surface`): cards, inputs, buttons, table bodies, the selected tab, chart paper.
- **Rail Grey** (`surface-2`): the header, the sidebar, table header rows, chips and inline code.
- **Hover Grey** (`surface-3`): hover fill for buttons, tabs, segmented options and run rows.
- **Ink** (`ink`) and **Secondary Ink** (`ink-2`): primary text and everything subordinate (hints, captions, field labels, run ids, placeholder text). `ink-2` holds AA contrast on every surface tier in both themes.
- **Hairline** (`line`) and **Strong Hairline** (`line-strong`): dividers and card borders, then control borders, the axis line and the scrollbar thumb.

### State
- **Finished Green** (`ok`), **Stalled Amber** (`warn`), **Failed Red** (`bad`), **Idle Grey** (`neutral`), and `info` (same as the accent) for running. They colour status badges, verdict badges, progress fills, the stalled callout, the error banner and invalid field borders, and nothing else.

### Data series
- `series-1` to `series-8` (blue, orange, green, violet, ochre, rose, teal, grey), with `dark-series-*` counterparts. Runs and groups take them in order and wrap after eight.

### Named Rules
**The Tokens-All-The-Way-Down Rule.** Every colour on screen, including Plotly's paper, grid, axis, font, hover label and series, is read from a CSS custom property. A theme change re-reads the tokens and redraws; no chart carries a literal hex.

**The Status-Is-Not-Decoration Rule.** `ok`, `warn`, `bad` and `neutral` appear only where they report a run status, a verdict, or a problem. Series colours identify data; they never signal state.

**The Mix-Don't-Pick Rule.** Tinted state surfaces are mixed from the state hue, not picked as new swatches: badges at 12% hue over the surface with a 35% hue border and text at 70% hue over ink; callouts and banners at 6-9% hue over the surface with a 35-45% hue border.

## Typography

**Body Font:** the platform UI sans (`system-ui`, then -apple-system, Segoe UI, Roboto, Helvetica Neue, Arial)
**Label/Mono Font:** ui-monospace, Cascadia Mono, Segoe UI Mono, Consolas (code and commands only)

**Character:** One familiar sans in tight steps, with weight doing most of the hierarchy work (400 body, 550 controls, 600 labels, 650 headings). The researcher should read it as their operating system, not as a brand.

### Hierarchy
- **Headline** (650, 1.3125rem, 1.25): the run name at the top of the Live tab. The largest text in the product; there is no display tier.
- **Title** (650, 1.125rem, 1.25): card headings, section titles in Results, empty-state headings.
- **Title Small** (650, 1rem): the brand in the header, the guide callout heading, sidebar empty-state heading.
- **Body** (400, 0.875rem, 1.5): the default page text and tab labels (tabs at 600).
- **Body Small** (400-550, 0.8125rem): controls, inputs, tables, chips, card hints and captions. Hints and callouts cap at 75ch.
- **Label** (600-650, 0.75rem): field labels, table header cells, sidebar group headings, badges.
- **Code** (0.92em mono on a `surface-2` pill with a hairline): config keys and CLI commands in prose.
- **Chart text** (Plotly, same family): 12px for the chart font, small-chart titles and empty annotations; 11px for ticks and hero axis titles.

### Named Rules
**The Tabular Numbers Rule.** Every table, tab count and inline number uses tabular numerals so columns and live-updating values do not jitter.

**The Weight-Before-Size Rule.** Headings tighten to -0.01em and balance their wrap; the step between tiers is weight first, then a modest size step (about 1.07-1.17x).

## Layout

A fixed application shell: a sticky header (3.25rem) on Rail Grey, then a two-column grid with a 20rem sidebar and a fluid workspace, each scrolling independently inside the viewport height. The sidebar stacks a two-column filter form, a meta line with quiet actions, an optional notice, and the scrolling run list with sticky group headings. The workspace opens with a sticky bar holding the tabs on the left and view controls (x-axis segmented control, Group seeds checkbox) on the right.

The Live and Compare boards are a column of cards: one hero card holding the eval-score chart (23rem tall, with a range slider), then three metric cards each holding a grid of small charts (13.75rem tall), three across. At 1100px and below the grid drops to two across. At 760px and below the shell collapses: the header wraps, the folder summary hides, a Runs toggle shows the sidebar as a block above the workspace, the workspace bar stops being sticky, charts go one across and the hero drops to 18rem.

Spacing runs on a rem rhythm: 0.25rem inside tight groups, 0.5-0.75rem between controls and chart tiles, 0.85rem as the sidebar's inner inset, 1rem between cards, 1.25rem workspace gutters and panel top, 1.75rem above a results section title. Prose blocks cap at 75ch, empty states at 62ch.

## Elevation & Depth

Mostly flat and tonal: the frame separates by surface tier (Desk Grey, Rail Grey, Card White) and 1px hairlines. A single soft two-layer shadow lifts only two things, the chart cards and the selected tab, so the eye knows what is content and what is the current view. Dark theme deepens the same shadow rather than dropping it. Focus is an outline, not a glow.

### Shadow Vocabulary
- **Card lift** (`box-shadow: 0 1px 2px rgb(20 23 29 / 6%), 0 4px 12px rgb(20 23 29 / 6%)`; dark `0 1px 2px rgb(0 0 0 / 30%), 0 6px 16px rgb(0 0 0 / 28%)`): chart cards and the selected tab.

### Named Rules
**The One Shadow Rule.** There is one shadow token. Anything else that needs separation gets a surface tier or a hairline.

## Shapes

Gently rounded and consistent: 6px on every control, row and table frame, 10px on cards and empty states, 4px on the small things inside them (chips, code, segmented options, skeleton lines), and full pills for badges, progress bars and scrollbar thumbs. All borders are 1px. Dashed borders carry one meaning: something absent or stepping back (the empty-state box, a chip for a setting every visible run shares). Icons are authored inline SVG, 1rem, 1.75 stroke, round caps and joins, drawn in `currentcolor`.

## Components

### Buttons
Plain and compact, with the platform's expectations intact.
- **Shape:** gently rounded (6px), 2rem minimum height, 1px Strong Hairline border.
- **Default:** Card White fill, Ink text, 13px at 550, optional leading icon with a 0.4rem gap.
- **Hover / Active:** fill steps to Hover Grey, then Hairline on press; 180ms ease-out on background and border.
- **Quiet:** no border or fill, accent text, tighter inline padding; hover fills with Blue Wash. Used for secondary list actions (Clear filters, Clear selection).
- **Disabled:** 55% opacity with a not-allowed cursor (also via `aria-disabled`).

### Segmented control
Built from real radio inputs in a fieldset: a 2px-padded track on Card White with a Strong Hairline border, options as 4px-rounded spans in Secondary Ink. Hover raises the option to Ink on Hover Grey; the checked option fills with the accent and Accent Ink text. Keyboard focus draws the focus ring on the option span (1px offset). Used for colour theme and x axis.

### Tabs
ARIA tabs in the workspace bar. Transparent with Secondary Ink at 600 by default; hover takes Hover Grey and Ink. The selected tab becomes a small card: Card White, Strong Hairline border and the card lift shadow. A tab may carry a tabular count in Secondary Ink at 500.

### Inputs / Fields
- **Style:** Card White, 1px Strong Hairline, 6px radius, 2rem tall, 13px text; a field is a 0.25rem-gap stack of a 12px/600 Secondary Ink label over the control. Placeholders use Secondary Ink at full opacity.
- **Hover / Focus:** border darkens to Secondary Ink on hover; focus is the global 2px focus outline at 2px offset.
- **Error / Disabled:** `aria-invalid` turns the border Failed Red; disabled drops to 55% opacity.

### Run list rows
The sidebar's signature list. A checkbox column and a stacked body: a bold, ellipsised title with its status badge, a 12px run id, and a stats line pairing a progress bar with tabular numbers. Rows are 6px-rounded, take Hover Grey on hover and Blue Wash when ticked. Group headings stick to the top of the scroller on Rail Grey and can carry the switches under test as a full-width Ink line.

### Badges
Pill-shaped status and verdict tags with a 0.45rem dot in the hue, drawn from one `--hue` variable through the Mix-Don't-Pick ratios. Running uses info blue, stalled amber, finished and better green, failed and worse red, interrupted, unknown and inconclusive grey. The reference verdict is accent-hued but outline only (transparent fill): it is what others are measured against, not an outcome.

### Chips
Config differences shown as 4px-rounded tags on Rail Grey with a hairline. A setting every visible run shares steps back: dashed Strong Hairline border, no fill, Secondary Ink text.

### Cards / Containers
- **Corner Style:** 10px.
- **Background:** Card White with a Hairline border.
- **Shadow Strategy:** the card lift (see Elevation & Depth).
- **Internal Padding:** 1rem top, 1.1rem sides, 0.75rem bottom (0.85rem / 0.75rem / 0.6rem on narrow screens). A card head pairs a Title with a Body Small hint on one baseline, wrapping as needed. Cards are never nested.

### Tables
Inside a 6px-rounded, hairline-bordered, horizontally scrollable frame on Card White. Header row on Rail Grey with 12px/650 Secondary Ink labels, cells at 0.45rem by 0.7rem with hairline row dividers and tabular numerals. Notes for a row sit on a following row in 12px Secondary Ink with the divider moved below them. In Results the reference row is pinned first and the comparison column is headed "Δ mean [95% CI]".

### Notices, callouts and banners
State messages are tinted boxes with a full 1px border and an inline icon, never a thick side stripe. The error banner (`role="alert"`) mixes Failed Red; the stalled callout and the sidebar's skipped-runs notice mix Stalled Amber; the guide callout that leads the Results tab mixes the accent and carries a Title Small heading and a row of action buttons. Inline errors are Failed Red text at 550.

### Empty and loading states
- **Empty:** a 10px-rounded box with a dashed Strong Hairline border, left-aligned, 62ch max, a Title heading and prose that names the exact command or action that fills it.
- **Loading:** chart frames show a skeleton panel inset 0.5rem; the run list shows skeleton lines. Skeletons pulse between `skeleton` and `skeleton-shine` over 1.6s.
- **Freshness:** a late update turns the freshness text Stalled Amber at 600.

### Charts (signature)
Plotly, themed entirely from tokens. Chart paper and plot match the card surface, gridlines use `chart-grid`, axis lines `chart-axis`, text `chart-font` in the page family; no zero line, no Plotly logo, and the lasso, select and autoscale buttons are removed. Hover is "x unified" with a Card White label and a Strong Hairline border; the mode bar is transparent with Secondary Ink icons.
- **Hero (eval score):** lines plus 6px markers at 2.5px width, y from zero, axis titles in 11px, a slim range slider, a horizontal legend above the plot when more than one run or group is drawn.
- **Small multiples:** 1.75px lines, the metric key as a 12px left-aligned title in `chart-title`.
- **Seed groups:** a group draws its mean line over a min-max band filled with the group colour at 16% alpha; the hero shows the mean only.
- **Shared x-range:** zooming one chart on a board pans the others; zoom survives new data (one `uirevision` per chart and x mode).
- **Empty chart:** a centered 12px annotation ("No values yet", "No evaluation recorded yet").

### Motion
State transitions on background, border and colour run 180ms on an exponential ease-out (`cubic-bezier(0.16, 1, 0.3, 1)`); progress bars ease their width over 240ms. Reduced-motion turns off every animation and transition, skeleton pulse included.

## Do's and Don'ts

### Do:
- **Do** read every chart colour through the tokens (`--chart-*`, `--series-*`, `--ink`, `--surface`, `--line-strong`, `--font`) and re-read them on a theme change.
- **Do** give every new token a light value and a dark value, and set both dark blocks (system preference and the pinned `data-theme="dark"`).
- **Do** derive tinted state surfaces with `color-mix` from the state hue (12% / 35% / 70% for badges, 6-9% fill with a 35-45% border for callouts and banners).
- **Do** use tabular numerals for every number that sits in a column or updates live.
- **Do** ship loading, empty, error and stale states for every new view, and make empty states name the exact command or action that fills them.
- **Do** draw icons as inline SVG at 1rem with a 1.75 stroke, round caps and joins, in `currentcolor`.
- **Do** keep controls at a 2rem minimum height, 6px radius and 1px Strong Hairline border.

### Don't:
- **Don't** write a literal hex into a component rule or a Plotly layout; add or reuse a token.
- **Don't** use `ok`, `warn`, `bad` or `neutral` for decoration or to tell series apart; don't use series colours to signal state.
- **Don't** give the reference an outcome colour; it stays outline-only in the accent.
- **Don't** add a second shadow or a coloured side stripe; separate with a surface tier, a hairline or a full 1px tinted border.
- **Don't** set prose, labels or headings in monospace; it is for code and commands.
- **Don't** nest cards, or wrap a single chart tile in its own card inside a card.
