// The dashboard page: URL-hash state, the run list, and the Live, Compare and Results tabs.
// Everything shown comes from the read-only GET API in api.js.

import { MetricsStore, fetchDiff, fetchResults, fetchRuns } from "./api.js";
import { ChartBoard, DEFAULT_X, SCORE_MEANING, X_MODES, readTheme, seriesColor } from "./charts.js";

const RUNS_EVERY_MS = 5000;
const METRICS_EVERY_MS = 2000;
const TABS = ["live", "compare", "results"];
const STATUSES = ["running", "stalled", "finished", "interrupted", "failed", "unknown"];
const THEME_KEY = "snakeppo.dashboard.theme";
const SVG_NS = "http:" + "//www.w3.org/2000/svg";
const SMOKE_COMMAND = "uv run snakeppo train --config configs/smoke.yaml";
// Switches that change what training curves measure, with the value a run without the key has.
const CURVE_SWITCHES = { "reward.strategy": "classic", "curriculum.grow": "off" };

const state = {
  tab: "live",
  selected: [], // run ids in the order they were ticked
  x: DEFAULT_X,
  group: false,
  baseline: "",
  runs: null, // null until the first answer from /api/runs
  runsDir: null, // the runs folder as the server names it, for copy-ready commands
  runsSeenAt: 0, // Date.now() of the last /api/runs answer; run quiet_seconds were measured then
  skipped: [],
  filters: { text: "", agent: "", group: "", status: "", switch: "" },
  problem: null, // {kind: "offline" | "server", message}
  results: { key: null, payload: null, error: null },
  diff: { key: null, payload: null, error: null },
  metricErrors: new Map(),
};

const store = new MetricsStore();
const rendered = new Map(); // element id -> signature of what it shows, to skip identical redraws
const $ = (id) => document.getElementById(id);
let liveBoard;
let compareBoard;

// ------------------------------------------------------------------ DOM helpers
function h(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "checked" || key === "value" || key === "selected") node[key] = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  append(node, children);
  return node;
}

function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

const ICONS = {
  alert: [
    "M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z",
    "M12 9v4",
    "M12 17h.01",
  ],
  info: ["M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z", "M12 16v-4", "M12 8h.01"],
};

function icon(name) {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("class", "icon");
  svg.setAttribute("aria-hidden", "true");
  for (const d of ICONS[name]) {
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("d", d);
    svg.append(path);
  }
  return svg;
}

/** Replace `node`'s content unless `signature` matches what it already shows. */
function show(node, signature, build) {
  if (rendered.get(node.id) === signature) return false;
  rendered.set(node.id, signature);
  node.replaceChildren(...[build()].flat().filter(Boolean));
  return true;
}

const code = (text) => h("code", null, text);

// ------------------------------------------------------------------ formatting
const integer = new Intl.NumberFormat("en-US");
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

function fixed(value, digits = 2) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function signed(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  const text = value.toFixed(2);
  return value >= 0 && !text.startsWith("-") ? `+${text}` : text;
}

function settingValue(value) {
  return typeof value === "string" ? value : JSON.stringify(value);
}

function duration(seconds) {
  if (seconds < 60) return "under a minute";
  const minutes = Math.round(seconds / 60);
  if (minutes < 90) return `${minutes} min`;
  const hours = Math.floor(minutes / 60);
  return `${hours} h ${minutes % 60} min`;
}

function scoreText(run) {
  if (typeof run.score !== "number") return "no eval yet";
  const evals = run.score_evals < 3 ? ` (${run.score_evals} eval${run.score_evals === 1 ? "" : "s"})` : "";
  return `score ${fixed(run.score)}${evals}`;
}

function runLabel(run) {
  if (run.group !== null && run.group !== undefined) return `${run.group} seed ${run.seed ?? "?"}`;
  return run.name || run.id;
}

function switchStrings(run) {
  return Object.entries(run.switches ?? {}).map(([key, value]) => `${key}=${settingValue(value)}`);
}

/** Switch settings every known run shares, such as a smoke config's sizes: noise when telling runs apart. */
function sharedSwitches(runs) {
  if (runs.length < 2) return new Set();
  const [first, ...rest] = runs.map((run) => new Set(switchStrings(run)));
  return new Set([...first].filter((text) => rest.every((set) => set.has(text))));
}

/** The switches that tell `run` apart from the other known runs. */
function distinctSwitches(run, shared) {
  return switchStrings(run).filter((text) => !shared.has(text));
}

function quietText(seconds) {
  if (seconds < 10) return "metrics just written";
  if (seconds < 90) return `last metrics ${Math.round(seconds)} s ago`;
  return `last metrics ${duration(seconds)} ago`;
}

function progressBar(run) {
  const total = run.total_updates;
  const share = total ? Math.min(1, (run.update ?? 0) / total) : 0;
  return h(
    "span",
    {
      class: "progress",
      role: "progressbar",
      "aria-label": "Training progress",
      "aria-valuemin": 0,
      "aria-valuemax": total ?? 0,
      "aria-valuenow": run.update ?? 0,
      dataset: { status: run.status },
    },
    h("span", { style: `width: ${(share * 100).toFixed(1)}%` }),
  );
}

function badge(status) {
  return h("span", { class: "badge", dataset: { status } }, status);
}

function progressText(run) {
  return `${run.update ?? 0} / ${run.total_updates ?? "?"}`;
}

const byCreatedDesc = (a, b) => String(b.created ?? "").localeCompare(String(a.created ?? ""));

// ------------------------------------------------------------------ URL hash
function readHash() {
  const params = {};
  for (const part of location.hash.replace(/^#/, "").split("&")) {
    if (!part) continue;
    const at = part.indexOf("=");
    params[at < 0 ? part : part.slice(0, at)] = at < 0 ? "" : part.slice(at + 1);
  }
  const decode = (text) => {
    try {
      return decodeURIComponent(text);
    } catch {
      return text;
    }
  };
  const tab = decode(params.tab ?? "");
  state.tab = TABS.includes(tab) ? tab : "live";
  const runs = params.runs ? params.runs.split(",").map(decode).filter(Boolean) : [];
  state.selected = [...new Set(runs)];
  const x = decode(params.x ?? "");
  state.x = X_MODES.some((mode) => mode.id === x) ? x : DEFAULT_X;
  state.group = params.group === "1";
  state.baseline = params.baseline ? decode(params.baseline) : "";
}

function writeHash() {
  const parts = [`tab=${state.tab}`];
  if (state.selected.length) parts.push(`runs=${state.selected.map(encodeURIComponent).join(",")}`);
  parts.push(`x=${state.x}`);
  if (state.group) parts.push("group=1");
  if (state.baseline) parts.push(`baseline=${encodeURIComponent(state.baseline)}`);
  const hash = `#${parts.join("&")}`;
  if (hash !== location.hash) history.replaceState(null, "", hash);
}

// ------------------------------------------------------------------ derived state
function knownRuns() {
  return new Map((state.runs ?? []).map((run) => [run.id, run]));
}

/** The Live tab's run: the run ticked last, else the newest running run, else the newest run. */
function liveRun() {
  const runs = knownRuns();
  for (let i = state.selected.length - 1; i >= 0; i -= 1) {
    if (runs.has(state.selected[i])) return runs.get(state.selected[i]);
  }
  const newest = [...(state.runs ?? [])].sort(byCreatedDesc);
  return newest.find((run) => run.status === "running") ?? newest[0] ?? null;
}

function compareRuns() {
  const runs = knownRuns();
  return state.selected.filter((id) => runs.has(id)).map((id) => runs.get(id));
}

function displayedRunIds() {
  if (state.tab === "live") {
    const run = liveRun();
    return run ? [run.id] : [];
  }
  if (state.tab === "compare") {
    const runs = compareRuns();
    return runs.length >= 2 ? runs.map((run) => run.id) : [];
  }
  return [];
}

function matchesFilters(run) {
  const { text, agent, group, status } = state.filters;
  if (text) {
    const needle = text.toLowerCase();
    const haystack = [run.id, run.name, run.group].filter(Boolean).join(" ").toLowerCase();
    if (!haystack.includes(needle)) return false;
  }
  if (agent && run.agent !== agent) return false;
  if (group && (run.group ?? "") !== group) return false;
  if (status && run.status !== status) return false;
  if (state.filters.switch && !switchStrings(run).includes(state.filters.switch)) return false;
  return true;
}

// ------------------------------------------------------------------ problems (offline, server)
function noteProblem(error) {
  state.problem = error?.unreachable
    ? { kind: "offline", message: error.message }
    : { kind: "server", message: error?.message ?? String(error) };
  renderBanner();
}

function clearProblem() {
  if (!state.problem) return;
  state.problem = null;
  renderBanner();
}

function renderBanner() {
  const banner = $("banner");
  const problem = state.problem;
  banner.hidden = !problem;
  show(banner, JSON.stringify(problem), () => {
    if (!problem) return [];
    const offline = problem.kind === "offline";
    return h(
      "div",
      { class: "banner" },
      icon("alert"),
      h(
        "div",
        null,
        h("strong", null, offline ? "The dashboard server cannot be reached." : "The server answered with an error."),
        offline
          ? h(
              "span",
              { class: "muted" },
              "Check that the server process is still running. This page keeps retrying and recovers on its own.",
            )
          : h("span", { class: "muted" }, `${problem.message} Retrying every few seconds.`),
      ),
    );
  });
}

// ------------------------------------------------------------------ sidebar
function fillSelect(select, firstLabel, values) {
  const signature = JSON.stringify(values);
  if (select.dataset.options === signature) return;
  select.dataset.options = signature;
  const current = select.value;
  select.replaceChildren(
    h("option", { value: "" }, firstLabel),
    ...values.map((value) => h("option", { value }, value)),
  );
  select.value = values.includes(current) ? current : "";
  if (select.value !== current) {
    state.filters[select.name] = select.value;
  }
}

function renderFilters() {
  const runs = state.runs ?? [];
  const form = $("filters");
  const distinct = (values) => [...new Set(values.filter(Boolean))].sort();
  fillSelect(form.elements.agent, "All agents", distinct(runs.map((run) => run.agent)));
  fillSelect(form.elements.group, "All groups", distinct(runs.map((run) => run.group)));
  const present = new Set(runs.map((run) => run.status));
  fillSelect(
    form.elements.status,
    "Any status",
    STATUSES.filter((status) => present.has(status)),
  );
  const shared = sharedSwitches(runs);
  fillSelect(form.elements.switch, "Any switch", distinct(runs.flatMap((run) => distinctSwitches(run, shared))));
  const disabled = state.runs === null || runs.length === 0;
  for (const control of form.elements) control.disabled = disabled;
}

function runItem(run, selected, shared) {
  // A group's seeds share their switches, so the section header names them once instead.
  const switches = run.group ? [] : distinctSwitches(run, shared);
  const all = switchStrings(run);
  return h(
    "li",
    null,
    h(
      "label",
      { class: selected ? "run is-selected" : "run", title: all.length ? `${run.id}\n${all.join("\n")}` : run.id },
      h("input", { type: "checkbox", checked: selected, dataset: { runId: run.id } }),
      h(
        "span",
        { class: "run-head" },
        h("span", { class: "run-title" }, run.group ? `seed ${run.seed ?? "?"}` : run.name || run.id),
        badge(run.status),
      ),
      h("span", { class: "run-id" }, switches.length ? `${run.id} · ${switches.join(", ")}` : run.id),
      h(
        "span",
        { class: "run-stats" },
        h("span", { class: "run-progress" }, progressBar(run), h("span", { class: "num" }, progressText(run))),
        h("span", { class: "num" }, scoreText(run)),
      ),
    ),
  );
}

function runSections(runs) {
  const groups = new Map();
  const ungrouped = [];
  for (const run of runs) {
    if (run.group === null || run.group === undefined) ungrouped.push(run);
    else groups.set(run.group, [...(groups.get(run.group) ?? []), run]);
  }
  const newest = (list) => list.map((run) => String(run.created ?? "")).sort().at(-1) ?? "";
  const sections = [...groups.entries()]
    .sort(([, a], [, b]) => newest(b).localeCompare(newest(a)))
    .map(([group, list]) => ({
      title: group,
      runs: list.sort((a, b) => (a.seed ?? 0) - (b.seed ?? 0)),
    }));
  if (ungrouped.length) sections.push({ title: "Ungrouped runs", runs: ungrouped.sort(byCreatedDesc) });
  return sections;
}

function renderSidebar() {
  renderFilters();
  const runs = state.runs;
  const list = $("run-list");
  const scroller = $("run-scroll");
  const filtered = (runs ?? []).filter(matchesFilters);
  const selected = new Set(state.selected);
  const filtering = Object.values(state.filters).some(Boolean);

  $("list-count").textContent =
    runs === null
      ? "Loading runs"
      : filtering
        ? `${filtered.length} of ${runs.length} runs`
        : `${runs.length} run${runs.length === 1 ? "" : "s"}`;
  $("clear-filters").hidden = !filtering;
  $("clear-selection").hidden = state.selected.length === 0;
  const toggleLabel = $("sidebar-toggle-label");
  toggleLabel.textContent = state.selected.length ? `Runs (${state.selected.length})` : "Runs";

  const signature = JSON.stringify([runs, state.filters, state.selected]);
  const focused = document.activeElement?.dataset?.runId;
  const scrollTop = scroller.scrollTop;
  const changed = show(list, signature, () => {
    if (runs === null) {
      return Array.from({ length: 6 }, () =>
        h(
          "div",
          { class: "skeleton-run", "aria-hidden": "true" },
          h("span", { class: "skeleton skeleton-line", style: "width: 55%" }),
          h("span", { class: "skeleton skeleton-line", style: "width: 85%" }),
        ),
      );
    }
    if (runs.length === 0) {
      return h(
        "div",
        { class: "empty" },
        h("h3", null, "No runs yet"),
        h("p", null, "Start a short training run from the repo:"),
        h("p", null, code(SMOKE_COMMAND)),
        h("p", { class: "muted" }, "It appears here within a few seconds."),
      );
    }
    if (filtered.length === 0) {
      return h(
        "div",
        { class: "empty" },
        h("h3", null, "No runs match these filters"),
        h("p", null, h("button", { type: "button", class: "button", onclick: clearFilters }, "Clear filters")),
      );
    }
    const shared = sharedSwitches(runs);
    return runSections(filtered).map((section, index) => {
      const grouped = section.runs[0].group;
      const switches = grouped ? distinctSwitches(section.runs[0], shared) : [];
      return h(
        "section",
        { class: "run-section", "aria-labelledby": `run-section-${index}` },
        h(
          "h3",
          { id: `run-section-${index}` },
          h("span", { class: "run-section-title" }, section.title),
          h("span", { class: "num" }, String(section.runs.length)),
          switches.length ? h("span", { class: "run-section-switches" }, switches.join(", ")) : null,
        ),
        h("ul", { class: "run-list" }, section.runs.map((run) => runItem(run, selected.has(run.id), shared))),
      );
    });
  });
  if (changed) {
    scroller.scrollTop = scrollTop;
    if (focused) list.querySelector(`input[data-run-id="${CSS.escape(focused)}"]`)?.focus();
  }

  const skipped = $("skipped");
  show(skipped, JSON.stringify(state.skipped), () =>
    state.skipped.length
      ? h(
          "details",
          { class: "notice" },
          h(
            "summary",
            null,
            `${state.skipped.length} run folder${state.skipped.length === 1 ? "" : "s"} could not be read`,
          ),
          h(
            "ul",
            null,
            state.skipped.map((item) => h("li", null, h("strong", null, item.id), `: ${item.error}`)),
          ),
        )
      : [],
  );
}

function renderSummary() {
  const runs = state.runs;
  const running = (runs ?? []).filter((run) => run.status === "running").length;
  $("run-summary").textContent =
    runs === null ? "" : `${runs.length} run${runs.length === 1 ? "" : "s"}, ${running} running`;
}

function toggleRun(id, on) {
  state.selected = state.selected.filter((other) => other !== id);
  if (on) state.selected.push(id);
  state.diff = { key: null, payload: null, error: null };
  update();
}

function clearFilters() {
  state.filters = { text: "", agent: "", group: "", status: "", switch: "" };
  const form = $("filters");
  form.reset();
  renderSidebar();
}

// ------------------------------------------------------------------ tabs and view controls
function renderTabs() {
  for (const tab of TABS) {
    const button = $(`tab-${tab}`);
    const active = tab === state.tab;
    button.setAttribute("aria-selected", String(active));
    button.tabIndex = active ? 0 : -1;
    $(`panel-${tab}`).hidden = !active;
  }
  const count = compareRuns().length;
  $("compare-count").textContent = count ? String(count) : "";
  $("view-controls").hidden = state.tab === "results";
  $("group-control").hidden = state.tab !== "compare";
  for (const input of $("x-control").querySelectorAll("input")) input.checked = input.value === state.x;
  $("group-toggle").checked = state.group;
}

function buildXControl() {
  const fieldset = $("x-control");
  for (const mode of X_MODES) {
    fieldset.append(
      h("label", null, h("input", { type: "radio", name: "x", value: mode.id }), h("span", null, mode.label)),
    );
  }
}

function selectTab(tab, focus = false) {
  if (state.tab === tab) return;
  state.tab = tab;
  update();
  if (focus) $(`tab-${tab}`).focus();
  if (tab === "results") refreshResults();
}

// ------------------------------------------------------------------ Live
function renderLive() {
  const header = $("live-header");
  const run = liveRun();
  const boardNode = $("live-board");
  if (state.runs === null) {
    show(header, "loading", () =>
      h(
        "div",
        { class: "run-header", "aria-hidden": "true" },
        h("span", { class: "skeleton", style: "width: 18rem; max-width: 100%; height: 1.6rem" }),
        h("span", { class: "skeleton skeleton-line", style: "width: 26rem; max-width: 100%" }),
      ),
    );
    boardNode.hidden = false;
    liveBoard.setLoading(true);
    return;
  }
  if (!run) {
    boardNode.hidden = true;
    show(header, "empty", () =>
      h(
        "div",
        { class: "empty" },
        h("h2", null, "No runs to watch yet"),
        h("p", null, "Training writes a run folder that this page picks up within a few seconds. Start a short run with"),
        h("p", null, code(SMOKE_COMMAND)),
        h("p", { class: "muted" }, "Its eval score, losses and PPO statistics then update here while it trains."),
      ),
    );
    return;
  }
  boardNode.hidden = false;
  const ticked = compareRuns().length;
  show(header, JSON.stringify([run, ticked, state.runsSeenAt]), () => liveHeader(run, ticked));
  renderFreshness();
  renderLiveCharts();
}

function liveHeader(run, ticked) {
  const facts = [
    h("span", { class: "num" }, `update ${run.update ?? 0} of ${run.total_updates ?? "?"}`),
    progressBar(run),
  ];
  const remaining = (run.total_updates ?? 0) - (run.update ?? 0);
  if (run.status === "running" && typeof run.seconds_per_update === "number" && remaining > 0) {
    facts.push(h("span", { class: "num" }, `about ${duration(remaining * run.seconds_per_update)} left`));
  }
  if (run.status === "running" && typeof run.quiet_seconds === "number") {
    facts.push(
      h("span", {
        class: "num freshness",
        dataset: { quiet: run.quiet_seconds, stall: run.stall_seconds, seenAt: state.runsSeenAt },
      }),
    );
  }
  if (typeof run.env_steps === "number") {
    facts.push(h("span", { class: "num" }, `${compact.format(run.env_steps)} env steps`));
  }
  facts.push(h("span", { class: "num" }, scoreText(run)));
  if (run.agent) facts.push(h("span", null, `agent ${run.agent}`));
  if (run.device) facts.push(h("span", null, `on ${run.device}`));
  const shared = sharedSwitches(state.runs ?? []);
  const switches = [...distinctSwitches(run, shared), ...switchStrings(run).filter((text) => shared.has(text))];
  const title = run.group ? runLabel(run) : run.name || run.id;
  const pieces = [
    h("div", { class: "run-header-top" }, h("h2", null, title), badge(run.status)),
    title !== run.id ? h("p", { class: "muted" }, run.id) : null,
    h("div", { class: "run-facts" }, facts),
    h(
      "div",
      { class: "chips", "aria-label": "Switches" },
      h("span", { class: "muted" }, "Switches:"),
      switches.length
        ? switches.map((text) =>
            h(
              "span",
              shared.has(text)
                ? { class: "chip is-shared", title: "Every run here has this setting" }
                : { class: "chip" },
              text,
            ),
          )
        : h("span", null, "none — baseline"),
    ),
  ];
  if (run.status === "stalled") {
    pieces.push(
      h(
        "div",
        { class: "callout", role: "status" },
        icon("alert"),
        h(
          "p",
          null,
          "No metrics for longer than the stall window; the process may be gone. Resume it with ",
          code(`uv run snakeppo train --resume ${state.runsDir ?? "<runs folder>"}/${run.id}`),
          ".",
        ),
      ),
    );
  }
  if (ticked > 1) {
    pieces.push(
      h(
        "p",
        { class: "muted" },
        `Showing the run ticked last. The Compare tab overlays all ${ticked} ticked runs.`,
      ),
    );
  }
  const error = state.metricErrors.get(run.id);
  if (error) pieces.push(h("p", { class: "inline-error" }, `Metrics could not be read: ${error}`));
  return h("div", { class: "run-header" }, pieces);
}

/** Ages each freshness note by the time since the server measured it, between run-list polls. */
function renderFreshness() {
  for (const node of document.querySelectorAll(".freshness")) {
    const quiet = Number(node.dataset.quiet) + (Date.now() - Number(node.dataset.seenAt)) / 1000;
    node.textContent = quietText(quiet);
    // Flag it at half the stall window: still running, but quieter than this run usually is.
    node.dataset.late = String(quiet > Number(node.dataset.stall) / 2);
  }
}

function renderLiveCharts() {
  const run = liveRun();
  if (state.tab !== "live" || !run) return;
  if (!store.loaded(run.id)) {
    liveBoard.setLoading(true);
    return;
  }
  const theme = readTheme();
  liveBoard.render({
    runs: [{ id: run.id, label: runLabel(run), color: seriesColor(theme, 0), lines: store.lines(run.id) }],
    xMode: state.x,
    theme,
  });
}

// ------------------------------------------------------------------ Compare
/** The CURVE_SWITCHES keys whose values differ between `runs`. */
function incomparableSwitches(runs) {
  return Object.entries(CURVE_SWITCHES)
    .filter(([key, fallback]) => new Set(runs.map((run) => settingValue(run.switches?.[key] ?? fallback))).size > 1)
    .map(([key]) => key);
}

function renderCompareNotice(runs) {
  const keys = state.runs === null || runs.length < 2 ? [] : incomparableSwitches(runs);
  show($("compare-notice"), JSON.stringify(keys), () =>
    keys.length
      ? h(
          "div",
          { class: "callout", role: "note" },
          icon("alert"),
          h(
            "p",
            null,
            "These runs differ in ",
            keys.map((key, index) => [index ? " and " : "", code(key)]),
            ", so their training curves (train score, losses, value fit) are not comparable: shaped rewards and curriculum stages change what those measure. The eval score is comparable, since every run is evaluated with greedy play on the full board.",
          ),
        )
      : [],
  );
}

function renderCompare() {
  const runs = compareRuns();
  renderCompareNotice(runs);
  const empty = $("compare-empty");
  const boardNode = $("compare-board");
  const diffNode = $("compare-diff");
  if (state.runs === null) {
    show(empty, "loading", () => []);
    boardNode.hidden = false;
    diffNode.hidden = true;
    compareBoard.setLoading(true);
    return;
  }
  if (runs.length < 2) {
    boardNode.hidden = true;
    diffNode.hidden = true;
    show(empty, `empty:${runs.length}:${(state.runs ?? []).length}`, () =>
      h(
        "div",
        { class: "empty" },
        h("h2", null, "Tick two or more runs to compare them"),
        h(
          "p",
          null,
          runs.length === 1
            ? "One run is ticked. Tick at least one more in the run list; every ticked run is overlaid on the same charts."
            : "Use the checkboxes in the run list; every ticked run is overlaid on the same charts.",
        ),
        h(
          "p",
          { class: "muted" },
          "Turn on Group seeds to collapse the seeds of each group into a mean line with a min–max band. The table below the charts lists each run's config differences from the first ticked run.",
        ),
      ),
    );
    return;
  }
  show(empty, "none", () => []);
  boardNode.hidden = false;
  diffNode.hidden = false;
  renderCompareCharts();
  refreshDiff(runs);
  renderDiff(runs);
}

function compareGroups(runs, theme) {
  const groups = new Map();
  for (const run of runs) {
    const id = run.group ?? `run:${run.id}`;
    if (!groups.has(id)) groups.set(id, { id, label: run.group ?? runLabel(run), runs: [] });
    groups.get(id).runs.push({ id: run.id, lines: store.lines(run.id) });
  }
  return [...groups.values()].map((group, index) => ({
    ...group,
    label: group.runs.length > 1 ? `${group.label} (${group.runs.length} seeds)` : group.label,
    color: seriesColor(theme, index),
  }));
}

function renderCompareCharts() {
  const runs = compareRuns();
  if (state.tab !== "compare" || runs.length < 2) return;
  if (!runs.some((run) => store.loaded(run.id))) {
    compareBoard.setLoading(true);
    return;
  }
  const theme = readTheme();
  if (state.group) {
    compareBoard.render({ groups: compareGroups(runs, theme), xMode: state.x, theme });
  } else {
    compareBoard.render({
      runs: runs.map((run, index) => ({
        id: run.id,
        label: runLabel(run),
        color: seriesColor(theme, index),
        lines: store.lines(run.id),
      })),
      xMode: state.x,
      theme,
    });
  }
}

async function refreshDiff(runs) {
  const key = runs.map((run) => run.id).join(",");
  if (state.diff.key === key) return;
  state.diff = { key, payload: null, error: null };
  try {
    const payload = await fetchDiff(runs[0].id, runs.slice(1).map((run) => run.id));
    if (state.diff.key === key) state.diff.payload = payload;
  } catch (error) {
    if (error.unreachable) noteProblem(error);
    if (state.diff.key === key) state.diff.error = error.message;
  }
  renderDiff(compareRuns());
}

function renderDiff(runs) {
  const node = $("compare-diff");
  const { key, payload, error } = state.diff;
  if (runs.length < 2) return;
  show(node, JSON.stringify([key, payload, error]), () => {
    const base = runs[0];
    const heading = h("h2", { class: "section-title" }, `Config differences against ${runLabel(base)}`);
    if (error) return [heading, h("p", { class: "inline-error", role: "alert" }, `The differences could not be loaded: ${error}`)];
    if (!payload) {
      return [heading, h("div", { class: "skeleton", style: "height: 6rem", "aria-hidden": "true" })];
    }
    const labels = knownRuns();
    const rows = [];
    for (const [id, differences] of Object.entries(payload.differences)) {
      const label = labels.has(id) ? runLabel(labels.get(id)) : id;
      const entries = Object.entries(differences);
      if (!entries.length) {
        rows.push(h("tr", null, h("td", null, label), h("td", { colspan: 3, class: "muted" }, "same learning config")));
      }
      for (const [setting, [was, now]] of entries) {
        rows.push(
          h(
            "tr",
            null,
            h("td", null, label),
            h("td", null, setting),
            h("td", null, settingValue(was)),
            h("td", null, settingValue(now)),
          ),
        );
      }
    }
    return [
      heading,
      h(
        "div",
        { class: "table-wrap" },
        h(
          "table",
          { class: "table" },
          h(
            "thead",
            null,
            h("tr", null, h("th", null, "Run"), h("th", null, "Setting"), h("th", null, runLabel(base)), h("th", null, "This run")),
          ),
          h("tbody", null, rows),
        ),
      ),
    ];
  });
}

// ------------------------------------------------------------------ Results
async function refreshResults() {
  if (state.tab !== "results") return;
  const key = state.baseline;
  try {
    const payload = await fetchResults(key || null);
    if (state.baseline !== key) return;
    state.results = { key, payload, error: null };
  } catch (error) {
    if (state.baseline !== key) return;
    if (error.unreachable) {
      noteProblem(error);
      return;
    }
    state.results = { key, payload: null, error: error.message };
  }
  renderResults();
}

const protocolKey = (row) => JSON.stringify(row.protocol ?? null);

/** Groups that can serve as a reference: another group shares their budget and eval settings. */
function referenceCandidates(rows) {
  const counts = new Map();
  for (const row of rows) counts.set(protocolKey(row), (counts.get(protocolKey(row)) ?? 0) + 1);
  return rows.filter((row) => counts.get(protocolKey(row)) > 1).map((row) => row.group);
}

function chooseReference(group) {
  state.baseline = group;
  update();
  refreshResults();
  $("baseline-picker").focus();
}

function renderBaselinePicker() {
  const picker = $("baseline-picker");
  const rows = state.results.payload?.rows ?? [];
  const byGroup = new Map(rows.map((row) => [row.group, row]));
  for (const run of state.runs ?? []) if (run.group && !byGroup.has(run.group)) byGroup.set(run.group, null);
  if (state.baseline && !byGroup.has(state.baseline)) byGroup.set(state.baseline, null);
  const usable = new Set(referenceCandidates(rows));
  const options = [...byGroup.keys()].sort().map((group) => {
    const row = byGroup.get(group);
    const budget = typeof row?.budget === "number" ? ` · ${compact.format(row.budget)} steps` : "";
    const alone = Boolean(row) && !usable.has(group) && group !== state.baseline;
    return { group, label: `${group}${budget}${alone ? " (no other group at this budget)" : ""}`, alone };
  });
  const signature = JSON.stringify(options);
  if (picker.dataset.options !== signature) {
    picker.dataset.options = signature;
    picker.replaceChildren(
      h("option", { value: "" }, "Auto: newest no-switch group at the same budget"),
      ...options.map(({ group, label, alone }) => h("option", { value: group, disabled: alone }, label)),
    );
  }
  picker.value = state.baseline;
  const failed = state.results.key === state.baseline && Boolean(state.results.error);
  picker.setAttribute("aria-invalid", String(failed));
}

function verdictCell(row, waiting) {
  if (row.verdict === null || row.verdict === undefined) {
    return h("td", { class: "muted" }, waiting ? "needs a reference" : `withheld: ${row.reason ?? "no reason given"}`);
  }
  if (row.verdict === "reference") return h("td", null, h("span", { class: "badge", dataset: { verdict: "reference" } }, "reference"));
  return h(
    "td",
    { class: "nowrap" },
    h("span", { class: "badge", dataset: { verdict: row.verdict } }, row.verdict),
    `${signed(row.mean_difference)} [${signed(row.ci_low)}, ${signed(row.ci_high)}]`,
  );
}

function resultRows(row, { waiting, showReference }) {
  const score =
    typeof row.score_std === "number" ? `${fixed(row.score_mean)} ± ${fixed(row.score_std)}` : fixed(row.score_mean);
  const differences =
    Object.entries(row.differences ?? {})
      .map(([key, [, value]]) => `${key}=${settingValue(value)}`)
      .join(", ") || "-";
  const notes = [];
  const judged = row.verdict !== null && row.verdict !== undefined && row.verdict !== "reference";
  if (judged && row.unpaired_seeds?.length) notes.push(`unpaired seeds ${row.unpaired_seeds.join(", ")}`);
  if (judged && row.ci_low === row.ci_high) {
    notes.push("every paired seed differs by the same amount, so the interval has no width");
  }
  notes.push(...(row.notes ?? []));
  const cells = [
    h("td", null, h("strong", null, row.group)),
    h("td", { class: "nowrap" }, typeof row.budget === "number" ? integer.format(row.budget) : "-"),
    h("td", null, `${row.finished}/${row.seeds}`),
    h("td", { class: "nowrap" }, score),
    showReference ? h("td", null, row.reference ?? "-") : null,
    h("td", null, differences),
    verdictCell(row, waiting),
  ].filter(Boolean);
  const main = h("tr", { class: notes.length ? "has-notes" : null }, cells);
  if (!notes.length) return [main];
  return [main, h("tr", { class: "row-notes" }, h("td", { colspan: cells.length }, notes.join("; ")))];
}

function referenceGuide(rows) {
  const candidates = referenceCandidates(rows);
  return h(
    "div",
    { class: "callout guide", role: "status" },
    icon("info"),
    h(
      "div",
      null,
      h("h2", null, "Pick a reference group"),
      h(
        "p",
        null,
        "No group here runs the code defaults at a matching budget, so there is nothing to judge the others against automatically. Groups trained with ",
        code("configs/smoke.yaml"),
        " never count as the baseline.",
      ),
      candidates.length
        ? h(
            "div",
            { class: "guide-actions" },
            candidates.map((group) =>
              h("button", { type: "button", class: "button", onclick: () => chooseReference(group) }, `Judge against ${group}`),
            ),
          )
        : h("p", { class: "muted" }, "No two groups share a budget and eval settings yet, so none can be compared."),
    ),
  );
}

function renderResults() {
  renderBaselinePicker();
  const { key, payload, error } = state.results;
  const current = key === state.baseline;
  const errorNode = $("results-error");
  show(errorNode, JSON.stringify(current ? error : null), () =>
    current && error
      ? h("p", { class: "inline-error", role: "alert" }, `${error}. Pick another reference group or the automatic choice.`)
      : [],
  );
  const body = $("results-body");
  show(body, JSON.stringify([current, payload, error]), () => {
    if (current && error) return [];
    if (!current || !payload) {
      return h(
        "div",
        { class: "table-wrap", "aria-hidden": "true" },
        h(
          "table",
          { class: "table" },
          h(
            "tbody",
            null,
            Array.from({ length: 4 }, () =>
              h("tr", null, Array.from({ length: 6 }, () => h("td", null, h("span", { class: "skeleton skeleton-line" })))),
            ),
          ),
        ),
      );
    }
    if (!payload.rows.length) {
      return h(
        "div",
        { class: "empty" },
        h("h2", null, "No seed groups yet"),
        h(
          "p",
          null,
          "The results table compares groups of runs that share a config and differ only in seed. Create one with ",
          code("snakeppo sweep"),
          ", or give several runs the same ",
          code("snakeppo train --group <id>"),
          ".",
        ),
        payload.left_out
          ? h("p", { class: "muted" }, `${payload.left_out} ungrouped run${payload.left_out === 1 ? "" : "s"} left out.`)
          : null,
      );
    }
    const unjudged = (row) => row.verdict === null || row.verdict === undefined;
    const waiting = payload.baseline === null && payload.rows.every(unjudged);
    const references = new Set(payload.rows.map((row) => row.reference).filter(Boolean));
    const showReference = references.size > 1;
    const rows = [...payload.rows].sort((a, b) => (b.verdict === "reference") - (a.verdict === "reference"));
    const columns = ["Group", "Budget", "Seeds", "Score", showReference ? "Reference" : null, "Differences", "Δ mean [95% CI]"];
    return [
      waiting ? referenceGuide(payload.rows) : null,
      h(
        "p",
        { class: "table-caption" },
        `Score: ${SCORE_MEANING}.`,
        references.size === 1 ? [" Each group is judged against ", h("strong", null, [...references][0]), " on paired seeds."] : null,
      ),
      h(
        "div",
        { class: "table-wrap" },
        h(
          "table",
          { class: "table" },
          h("thead", null, h("tr", null, columns.filter(Boolean).map((name) => h("th", { scope: "col" }, name)))),
          h("tbody", null, rows.flatMap((row) => resultRows(row, { waiting, showReference }))),
        ),
      ),
      payload.left_out
        ? h(
            "p",
            { class: "table-foot" },
            `${payload.left_out} ungrouped run${payload.left_out === 1 ? "" : "s"} left out (sweeps and --group set one).`,
          )
        : null,
    ];
  });
}

// ------------------------------------------------------------------ rendering and loops
function update() {
  writeHash();
  renderTabs();
  renderSummary();
  renderSidebar();
  renderBanner();
  if (state.tab === "live") renderLive();
  if (state.tab === "compare") renderCompare();
  if (state.tab === "results") renderResults();
  metricsLoop.kick();
}

function redrawCharts() {
  renderLiveCharts();
  renderCompareCharts();
}

/** Run `task` every `ms` while the page is visible; `kick` runs it now (or right after). */
function every(ms, task) {
  let timer = null;
  let running = false;
  let again = false;
  async function tick() {
    clearTimeout(timer);
    timer = null;
    if (document.hidden) return;
    if (running) {
      again = true;
      return;
    }
    running = true;
    try {
      await task();
    } catch (error) {
      console.error(error);
    } finally {
      running = false;
      if (again) {
        again = false;
        queueMicrotask(tick);
      } else if (!document.hidden) {
        timer = setTimeout(tick, ms);
      }
    }
  }
  return {
    kick: tick,
    stop() {
      clearTimeout(timer);
      timer = null;
    },
  };
}

const runsLoop = every(RUNS_EVERY_MS, async () => {
  try {
    const payload = await fetchRuns();
    state.runs = payload.runs ?? [];
    state.skipped = payload.skipped ?? [];
    state.runsDir = payload.runs_dir ?? null;
    state.runsSeenAt = Date.now();
    clearProblem();
  } catch (error) {
    noteProblem(error);
    return;
  }
  update();
  if (state.tab === "results") await refreshResults();
});

const metricsLoop = every(METRICS_EVERY_MS, async () => {
  const ids = displayedRunIds();
  if (!ids.length) return;
  const before = new Set(ids.filter((id) => store.loaded(id)));
  const outcomes = await Promise.allSettled(ids.map((id) => store.poll(id)));
  let changed = false;
  let errorsChanged = false;
  outcomes.forEach((outcome, index) => {
    const id = ids[index];
    if (outcome.status === "fulfilled") {
      clearProblem();
      if (state.metricErrors.delete(id)) errorsChanged = true;
      if (outcome.value.appended || outcome.value.reset || !before.has(id)) changed = true;
    } else if (outcome.reason?.unreachable) {
      noteProblem(outcome.reason);
    } else {
      const message = outcome.reason?.message ?? String(outcome.reason);
      if (state.metricErrors.get(id) !== message) errorsChanged = true;
      state.metricErrors.set(id, message);
    }
  });
  if (errorsChanged && state.tab === "live") renderLive();
  if (changed) redrawCharts();
});

// ------------------------------------------------------------------ theme
function storedTheme() {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}

function applyTheme(choice) {
  if (choice === "light" || choice === "dark") document.documentElement.dataset.theme = choice;
  else delete document.documentElement.dataset.theme;
  try {
    if (choice === "system") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, choice);
  } catch {
    // Without storage the choice lasts until the page is reloaded.
  }
  for (const input of $("theme-control").querySelectorAll("input")) input.checked = input.value === choice;
  redrawCharts();
}

// ------------------------------------------------------------------ wiring
function wire() {
  $("theme-control").addEventListener("change", (event) => applyTheme(event.target.value));
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (storedTheme() === "system") redrawCharts();
  });

  const toggle = $("sidebar-toggle");
  toggle.addEventListener("click", () => {
    const open = toggle.getAttribute("aria-expanded") !== "true";
    toggle.setAttribute("aria-expanded", String(open));
    $("sidebar").classList.toggle("is-open", open);
  });

  const form = $("filters");
  form.addEventListener("submit", (event) => event.preventDefault());
  form.addEventListener("input", (event) => {
    const { name, value } = event.target;
    if (!(name in state.filters)) return;
    state.filters[name] = name === "text" ? value.trim() : value;
    renderSidebar();
  });
  $("clear-filters").addEventListener("click", clearFilters);
  $("clear-selection").addEventListener("click", () => {
    state.selected = [];
    update();
  });
  $("run-list").addEventListener("change", (event) => {
    const id = event.target.dataset?.runId;
    if (id) toggleRun(id, event.target.checked);
  });

  const tablist = document.querySelector('[role="tablist"]');
  tablist.addEventListener("click", (event) => {
    const tab = event.target.closest('[role="tab"]');
    if (tab) selectTab(tab.id.replace("tab-", ""));
  });
  tablist.addEventListener("keydown", (event) => {
    const index = TABS.indexOf(state.tab);
    const moves = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: TABS.length - 1 };
    if (!(event.key in moves)) return;
    event.preventDefault();
    selectTab(TABS[(moves[event.key] + TABS.length) % TABS.length], true);
  });

  $("x-control").addEventListener("change", (event) => {
    state.x = event.target.value;
    writeHash();
    redrawCharts();
  });
  $("group-toggle").addEventListener("change", (event) => {
    state.group = event.target.checked;
    writeHash();
    redrawCharts();
  });
  $("baseline-picker").addEventListener("change", (event) => {
    state.baseline = event.target.value;
    update();
    refreshResults();
  });

  window.addEventListener("hashchange", () => {
    readHash();
    state.diff = { key: null, payload: null, error: null };
    update();
    if (state.tab === "results") refreshResults();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      runsLoop.stop();
      metricsLoop.stop();
    } else {
      runsLoop.kick();
      metricsLoop.kick();
    }
  });

  // A board drawn while its tab was hidden has the wrong size; fit it once it is visible.
  const resize = new ResizeObserver((entries) => {
    for (const entry of entries) {
      if (entry.contentRect.width === 0) continue;
      if (entry.target.id === "panel-live") liveBoard.resize();
      if (entry.target.id === "panel-compare") compareBoard.resize();
    }
  });
  resize.observe($("panel-live"));
  resize.observe($("panel-compare"));
}

function start() {
  liveBoard = new ChartBoard($("live-board"), "live");
  compareBoard = new ChartBoard($("compare-board"), "compare");
  buildXControl();
  for (const input of $("theme-control").querySelectorAll("input")) input.checked = input.value === storedTheme();
  readHash();
  wire();
  update();
  runsLoop.kick();
  setInterval(renderFreshness, 1000);
}

start();
