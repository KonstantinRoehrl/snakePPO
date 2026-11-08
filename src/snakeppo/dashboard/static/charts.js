// Chart specification and Plotly rendering. Plotly is the global the classic script
// /vendor/plotly.min.js defines; colours come from the CSS tokens in style.css.

export const X_MODES = [
  { id: "update", label: "Update", title: "update" },
  { id: "env_steps", label: "Env steps", title: "env steps" },
  { id: "wall_time", label: "Wall time", title: "wall time (minutes)" },
];
export const DEFAULT_X = "env_steps";

export const HERO = {
  id: "eval",
  title: "Eval score",
  hint: "eval/score_mean: the mean score of greedy play at each evaluation.",
  key: "eval/score_mean",
};

export const CARDS = [
  {
    id: "training",
    title: "Training",
    hint: "What the agent achieves while it explores, and how well the value head predicts returns.",
    keys: ["train/score_mean", "ppo/explained_variance", "train/length_mean"],
  },
  {
    id: "losses",
    title: "Losses",
    hint: "Mean over each update's minibatches.",
    keys: ["loss/policy", "loss/value", "loss/entropy"],
  },
  {
    id: "ppo",
    title: "PPO statistics",
    hint: "How far each update moves the policy, and the optimiser settings it ran with.",
    keys: [
      "ppo/approx_kl",
      "ppo/clip_fraction",
      "ppo/grad_norm",
      "ppo/epochs",
      "ppo/lr",
      "ppo/entropy_coef",
    ],
  },
];

const PLOT_CONFIG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"],
};
const SERIES_TOKENS = 8;

/** Chart colours read from the CSS custom properties; call again after a theme change. */
export function readTheme(element = document.documentElement) {
  const style = getComputedStyle(element);
  const token = (name) => style.getPropertyValue(name).trim();
  return {
    paper: token("--chart-paper"),
    plot: token("--chart-plot"),
    grid: token("--chart-grid"),
    axis: token("--chart-axis"),
    font: token("--chart-font"),
    title: token("--chart-title"),
    ink: token("--ink"),
    hover: token("--surface"),
    hoverBorder: token("--line-strong"),
    fontFamily: token("--font"),
    series: Array.from({ length: SERIES_TOKENS }, (_, i) => token(`--series-${i + 1}`)),
  };
}

export function seriesColor(theme, index) {
  return theme.series[index % theme.series.length];
}

function translucent(color, alpha) {
  const match = /^#([0-9a-f]{6})$/i.exec(color);
  if (!match) return `rgba(127, 127, 127, ${alpha})`;
  const value = parseInt(match[1], 16);
  return `rgba(${value >> 16}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}

function numberOrNull(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function xOf(line, xMode) {
  const value = numberOrNull(line[xMode]);
  return xMode === "wall_time" && value !== null ? value / 60 : value;
}

// ------------------------------------------------------------------ trace builders
/** One run's line for `key`; `sparse` keeps only the lines that carry the key (evals). */
export function runTraces(run, key, xMode, { sparse = false, hero = false } = {}) {
  const lines = sparse ? run.lines.filter((line) => key in line) : run.lines;
  return [
    {
      type: "scatter",
      mode: hero ? "lines+markers" : "lines",
      name: run.label,
      x: lines.map((line) => xOf(line, xMode)),
      y: lines.map((line) => numberOrNull(line[key])),
      connectgaps: false,
      line: { color: run.color, width: hero ? 2.5 : 1.75 },
      marker: { color: run.color, size: 6 },
      yhoverformat: ".4~g",
    },
  ];
}

/**
 * Seeds of one group aligned by `update`: per update the mean, min and max over the runs that
 * have a value there. x is that update's env_steps (identical across seeds), the update itself,
 * or the mean wall time.
 */
export function aggregate(runsLines, key, xMode, { sparse = false } = {}) {
  const byUpdate = new Map();
  for (const lines of runsLines) {
    for (const line of lines) {
      if (sparse && !(key in line)) continue;
      let slot = byUpdate.get(line.update);
      if (!slot) {
        slot = { envSteps: numberOrNull(line.env_steps), walls: [], values: [] };
        byUpdate.set(line.update, slot);
      }
      const wall = numberOrNull(line.wall_time);
      if (wall !== null) slot.walls.push(wall);
      const value = numberOrNull(line[key]);
      if (value !== null) slot.values.push(value);
    }
  }
  const updates = [...byUpdate.keys()].sort((a, b) => a - b);
  const out = { x: [], mean: [], low: [], high: [] };
  for (const update of updates) {
    const slot = byUpdate.get(update);
    let x = update;
    if (xMode === "env_steps") x = slot.envSteps;
    if (xMode === "wall_time") x = slot.walls.length ? average(slot.walls) / 60 : null;
    out.x.push(x);
    out.mean.push(slot.values.length ? average(slot.values) : null);
    out.low.push(slot.values.length ? Math.min(...slot.values) : null);
    out.high.push(slot.values.length ? Math.max(...slot.values) : null);
  }
  return out;
}

function average(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/** A group's mean line, preceded by a min–max band (a `tonexty` pair) when `band` is set. */
export function groupTraces(group, key, xMode, { sparse = false, hero = false, band = true } = {}) {
  const stats = aggregate(
    group.runs.map((run) => run.lines),
    key,
    xMode,
    { sparse },
  );
  const traces = [];
  if (band && group.runs.length > 1) {
    const edge = {
      type: "scatter",
      mode: "lines",
      x: stats.x,
      line: { width: 0, color: group.color },
      connectgaps: false,
      hoverinfo: "skip",
      showlegend: false,
      legendgroup: group.id,
    };
    traces.push({ ...edge, y: stats.high, name: `${group.label} max` });
    traces.push({
      ...edge,
      y: stats.low,
      name: `${group.label} min`,
      fill: "tonexty",
      fillcolor: translucent(group.color, 0.16),
    });
  }
  traces.push({
    type: "scatter",
    mode: hero ? "lines+markers" : "lines",
    name: group.label,
    legendgroup: group.id,
    x: stats.x,
    y: stats.mean,
    connectgaps: false,
    line: { color: group.color, width: hero ? 2.5 : 1.75 },
    marker: { color: group.color, size: 6 },
    yhoverformat: ".4~g",
  });
  return traces;
}

// ------------------------------------------------------------------ layout
function xTitle(xMode) {
  return X_MODES.find((mode) => mode.id === xMode)?.title ?? xMode;
}

function chartLayout(theme, chart, xMode, { empty, legend }) {
  const axis = {
    gridcolor: theme.grid,
    linecolor: theme.axis,
    zeroline: false,
    automargin: true,
    tickfont: { size: 11 },
  };
  const layout = {
    // A fixed revision per chart and x mode keeps the user's zoom across new data.
    uirevision: `${chart.id}:${xMode}`,
    paper_bgcolor: theme.paper,
    plot_bgcolor: theme.plot,
    font: { family: theme.fontFamily, size: 12, color: theme.font },
    margin: chart.hero ? { l: 52, r: 16, t: legend ? 30 : 12, b: 40 } : { l: 52, r: 12, t: 32, b: 30 },
    xaxis: {
      ...axis,
      tickformat: xMode === "env_steps" ? "~s" : "",
      hoverformat: xMode === "wall_time" ? ".1f" : "",
    },
    yaxis: { ...axis, rangemode: chart.hero ? "tozero" : "normal" },
    showlegend: legend,
    legend: { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom" },
    hovermode: "x unified",
    hoverlabel: {
      bgcolor: theme.hover,
      bordercolor: theme.hoverBorder,
      font: { color: theme.ink, family: theme.fontFamily },
    },
    annotations: empty
      ? [
          {
            text: chart.hero ? "No evaluation recorded yet" : "No values yet",
            showarrow: false,
            xref: "paper",
            yref: "paper",
            x: 0.5,
            y: 0.5,
            font: { color: theme.font, size: 12 },
          },
        ]
      : [],
  };
  if (chart.hero) {
    layout.xaxis.title = { text: xTitle(xMode), font: { size: 11 } };
    layout.xaxis.rangeslider = { thickness: 0.08, bgcolor: theme.plot, bordercolor: theme.grid };
    layout.yaxis.title = { text: "score", font: { size: 11 } };
  } else {
    layout.title = {
      text: chart.key,
      font: { size: 12, color: theme.title },
      x: 0.01,
      xanchor: "left",
      y: 0.98,
      yanchor: "top",
    };
  }
  return layout;
}

function hasValues(traces) {
  return traces.some((trace) => trace.y.some((value) => value !== null));
}

/** The x-range part of a relayout event, as an update for the other charts; null otherwise. */
function xRangeUpdate(event) {
  if ("xaxis.range[0]" in event) {
    return { "xaxis.range": [event["xaxis.range[0]"], event["xaxis.range[1]"]] };
  }
  if (Array.isArray(event["xaxis.range"])) return { "xaxis.range": event["xaxis.range"] };
  if (event["xaxis.autorange"]) return { "xaxis.autorange": true };
  return null;
}

// ------------------------------------------------------------------ board
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function card(spec, headingId) {
  const section = element("section", "card");
  section.setAttribute("aria-labelledby", headingId);
  const head = element("div", "card-head");
  const title = element("h3", "", spec.title);
  title.id = headingId;
  head.append(title, element("p", "card-hint", spec.hint));
  section.append(head);
  return section;
}

/**
 * The eval-score card and the three metric cards for one tab. `render` draws runs (one line
 * each) or groups (mean line, plus a min–max band in the small charts); every chart on the
 * board shares its x-range with the others.
 */
export class ChartBoard {
  constructor(root, prefix) {
    this.root = root;
    this.charts = [];
    this.syncing = false;
    const heroCard = card(HERO, `${prefix}-eval-title`);
    heroCard.append(this.#frame({ id: `${prefix}-eval`, key: HERO.key, hero: true, sparse: true }));
    root.append(heroCard);
    for (const spec of CARDS) {
      const section = card(spec, `${prefix}-${spec.id}-title`);
      const grid = element("div", "chart-grid");
      for (const key of spec.keys) {
        const id = `${prefix}-${key.replace(/[^a-z0-9]+/gi, "-")}`;
        grid.append(this.#frame({ id, key, hero: false, sparse: key.startsWith("eval/") }));
      }
      section.append(grid);
      root.append(section);
    }
  }

  #frame(chart) {
    const frame = element("div", chart.hero ? "chart-frame is-hero" : "chart-frame");
    frame.dataset.loading = "";
    const plot = element("div", "chart-plot");
    plot.id = chart.id;
    plot.setAttribute("role", "img");
    plot.setAttribute("aria-label", chart.hero ? "Eval score chart" : `${chart.key} chart`);
    frame.append(plot);
    this.charts.push({ ...chart, frame, plot, bound: false });
    return frame;
  }

  setLoading(loading) {
    for (const chart of this.charts) chart.frame.toggleAttribute("data-loading", loading);
  }

  /** `runs`: [{id, label, color, lines}]; or `groups`: [{id, label, color, runs: [run]}]. */
  render({ runs = null, groups = null, xMode, theme }) {
    const legend = (groups ?? runs ?? []).length > 1;
    for (const chart of this.charts) {
      const options = { sparse: chart.sparse, hero: chart.hero };
      const traces = groups
        ? groups.flatMap((group) =>
            groupTraces(group, chart.key, xMode, { ...options, band: !chart.hero }),
          )
        : (runs ?? []).flatMap((run) => runTraces(run, chart.key, xMode, options));
      const layout = chartLayout(theme, chart, xMode, {
        empty: !hasValues(traces),
        legend: chart.hero && legend,
      });
      Plotly.react(chart.plot, traces, layout, PLOT_CONFIG);
      chart.frame.removeAttribute("data-loading");
      if (!chart.bound) this.#bindSync(chart);
    }
  }

  /** Plotly sizes a chart when drawn; redraw at the real size after the board becomes visible. */
  resize() {
    for (const chart of this.charts) {
      if (chart.bound) Plotly.Plots.resize(chart.plot);
    }
  }

  #bindSync(chart) {
    chart.bound = true;
    chart.plot.on("plotly_relayout", (event) => {
      if (this.syncing) return;
      const update = xRangeUpdate(event);
      if (!update) return;
      this.syncing = true;
      const others = this.charts.filter((other) => other !== chart && other.bound);
      Promise.all(others.map((other) => Plotly.relayout(other.plot, update))).finally(() => {
        this.syncing = false;
      });
    });
  }
}
