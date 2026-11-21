// The Launch tab: pick a preset from the configs folder, start one training run or a seed sweep,
// and follow, stop and resume what the dashboard started. The server runs the same
// `snakeppo train` / `snakeppo sweep` command the terminal would, so every launch writes an
// ordinary run folder.

import { fetchLaunches, fetchPresets, resumeLaunch, startLaunch, stopLaunch } from "./api.js";
import { runColor } from "./colors.js";

const LAUNCHES_EVERY_MS = 5000;
const DEFAULT_SEEDS = "1,2,3";
const DEFAULT_PRESET = "baseline.yaml";
const TOP_LEVEL = "(top level)";
// Each launch state wears the run-status badge with the same meaning.
const STATE_BADGES = {
  running: "running",
  stopping: "stalled",
  finished: "finished",
  stopped: "interrupted",
  failed: "failed",
};
// How a sweep's seed statuses read in its progress line, in this order.
const SEED_WORDS = {
  finished: "done",
  running: "training",
  waiting: "waiting",
  interrupted: "interrupted",
  stalled: "stalled",
  skipped: "skipped",
  failed: "failed",
};
const startedFormat = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

// ------------------------------------------------------------------ request building
/** The non-empty lines of the overrides box, trimmed: one `section.key=value` each. */
export function overrideLines(text) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

/** Seeds typed as `1,2,3`, as `snakeppo sweep --seeds` takes them; null unless every part is an integer. */
export function parseSeeds(text) {
  const parts = text
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  if (!parts.length || !parts.every((part) => /^-?\d+$/.test(part))) return null;
  return parts.map(Number);
}

/**
 * The description the panel suggests, worded as the CLI words it: the preset's text, then
 * `Changed: key=value, …` for each override in the order given, leaving out bookkeeping keys
 * (seed, name, runs folder, …) that change nothing a comparison looks at.
 */
export function suggestedDescription(presetText, overrides, bookkeepingKeys) {
  const bookkeeping = new Set(bookkeepingKeys);
  const changes = overrides.filter((line) => !bookkeeping.has(line.split("=")[0].trim()));
  return changes.length ? `${presetText} Changed: ${changes.join(", ")}`.trim() : presetText;
}

/**
 * The JSON body for `POST /api/launch` from what the form holds, or `{error}` naming the value
 * that cannot be sent. The server checks the config itself before it starts anything.
 */
export function launchRequest({ kind, preset, overridesText, description, seedsText, jobsText }) {
  const request = { kind, preset, overrides: overrideLines(overridesText), description: description.trim() };
  if (kind !== "sweep") return { request };
  const seeds = parseSeeds(seedsText);
  if (!seeds) {
    return { error: `Seeds must be whole numbers separated by commas, such as 1,2,3; got "${seedsText.trim()}".` };
  }
  const jobs = Number(jobsText.trim());
  if (!Number.isInteger(jobs) || jobs < 1) {
    return { error: `Seeds at once must be a whole number of at least 1; got "${jobsText.trim()}".` };
  }
  return { request: { ...request, seeds, jobs } };
}

// ------------------------------------------------------------------ launch summaries
/** Seeds as a range when they count up without gaps (`1–5`), else listed (`1, 4, 9`). */
export function seedText(seeds) {
  const counting = seeds.length > 2 && seeds.every((seed, index) => !index || seed === seeds[index - 1] + 1);
  return counting ? `${seeds[0]}–${seeds[seeds.length - 1]}` : seeds.join(", ");
}

/** How far a sweep got, such as `2 done, 2 training, 1 waiting`. */
export function seedProgress(seedRuns) {
  const counts = new Map();
  for (const { status } of seedRuns) counts.set(status, (counts.get(status) ?? 0) + 1);
  const order = Object.keys(SEED_WORDS);
  const rank = (status) => (order.includes(status) ? order.indexOf(status) : order.length);
  return [...counts]
    .sort(([a], [b]) => rank(a) - rank(b))
    .map(([status, count]) => `${count} ${SEED_WORDS[status] ?? status}`)
    .join(", ");
}

/** `{title, detail}`: what a launch started, and where it stands now. */
export function launchSummary(entry) {
  if (entry.kind === "sweep") {
    const detail = [`seeds ${seedText(entry.seeds ?? [])}`, `${entry.jobs ?? 1} at a time`];
    const progress = seedProgress(entry.seed_runs ?? []);
    if (progress) detail.push(progress);
    const verb = entry.resumes ? "Resumed sweep" : "Sweep";
    return { title: `${verb} of ${entry.preset}`, detail: detail.join(" · ") };
  }
  const title = entry.kind === "resume" ? "Resumed run" : `Run of ${entry.preset}`;
  if (entry.run) return { title, detail: entry.run };
  return { title, detail: entry.alive ? "starting, no run folder yet" : "exited before creating a run folder" };
}

/** The run whose colour marks a launch: its own run, else the first seed run that exists. */
export function launchRunId(entry) {
  return entry.run ?? (entry.seed_runs ?? []).find((seed) => seed.run)?.run ?? null;
}

function startedText(created) {
  const date = new Date(created ?? "");
  return Number.isNaN(date.getTime()) ? String(created ?? "-") : startedFormat.format(date);
}

// ------------------------------------------------------------------ DOM helpers
function h(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (["checked", "value", "disabled", "open"].includes(key)) node[key] = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const code = (text) => h("code", null, text);
const skeletonLine = (width) =>
  h("span", { class: "skeleton skeleton-line launch-skeleton", style: `width: ${width}rem`, "aria-hidden": "true" });

/** A labelled control with an optional hint that screen readers read as its description. */
function field(id, label, control, hint) {
  control.id = id;
  if (hint) control.setAttribute("aria-describedby", `${id}-hint`);
  return h(
    "div",
    { class: "field" },
    h("label", { class: "field-label", for: id }, label),
    control,
    hint ? h("span", { class: "launch-field-hint", id: `${id}-hint` }, hint) : null,
  );
}

function radio(name, value, label, checked = false) {
  return h("label", null, h("input", { type: "radio", name, value, checked }), h("span", null, label));
}

function dot(color) {
  return color
    ? h("span", { class: "launch-dot", style: `--dot: ${color}`, "aria-hidden": "true" })
    : h("span", { class: "launch-dot is-empty", "aria-hidden": "true" });
}

// ------------------------------------------------------------------ the view
/** Mounts the Launch tab into `root`; `ctx` is the page's shared context (see app.js). */
export function mountLaunch(root, ctx) {
  const view = {
    visible: false,
    timer: null,
    presets: null, // the /api/presets payload once read
    presetsError: null,
    selected: null, // the chosen preset's path
    descriptionEdited: false,
    starting: false,
    formError: null,
    formNotice: null, // nodes of the "started" note
    launches: null, // the LaunchEntry list once read
    launchesError: null,
    busy: new Set(), // launch ids with a stop or resume request in flight
    openLogs: new Set(), // launch ids whose log the researcher opened
    closedLogs: new Set(), // failed launches whose log the researcher closed
    rendered: "", // signature of what the launches table shows
  };

  const presetList = h("div", { class: "launch-preset-list" });
  const presetCard = h(
    "section",
    { class: "card launch-presets", "aria-labelledby": "launch-presets-title" },
    h(
      "div",
      { class: "card-head" },
      h("h3", { id: "launch-presets-title" }, "Presets"),
      h("span", { class: "card-hint" }, "from the configs folder"),
    ),
    presetList,
  );

  const formTitle = h("h3", { id: "launch-form-title" }, "Pick a preset");
  const formHint = h("span", { class: "card-hint" });
  const kindControl = h(
    "fieldset",
    { class: "segmented" },
    h("legend", null, "What to start"),
    radio("launch-kind", "train", "Single run", true),
    radio("launch-kind", "sweep", "Seed sweep"),
  );
  const seedsInput = h("input", {
    class: "input",
    type: "text",
    name: "seeds",
    value: DEFAULT_SEEDS,
    inputmode: "numeric",
    autocomplete: "off",
  });
  const jobsInput = h("input", { class: "input", type: "number", name: "jobs", value: "1", min: "1", step: "1" });
  const sweepFields = h(
    "div",
    { class: "launch-sweep-fields", hidden: true },
    field("launch-seeds", "Seeds", seedsInput, "comma-separated, one run per seed"),
    field("launch-jobs", "Seeds at once", jobsInput, "seeds trained in parallel"),
  );
  const overridesInput = h("textarea", {
    class: "input launch-textarea launch-overrides",
    name: "overrides",
    rows: "4",
    spellcheck: "false",
    placeholder: "ppo.lr=0.0005\ntrain.total_env_steps=5000000",
  });
  const descriptionInput = h("textarea", { class: "input launch-textarea", name: "description", rows: "3" });
  const resetDescription = h(
    "button",
    { type: "button", class: "button button-quiet", hidden: true },
    "Use the suggested text",
  );
  const startButton = h("button", { type: "submit", class: "button launch-start", disabled: true }, "Start run");
  const formError = h("p", { class: "inline-error", role: "alert", hidden: true });
  const formNotice = h("p", { class: "launch-notice", role: "status", hidden: true });
  const form = h(
    "form",
    { class: "launch-form", novalidate: true, "aria-labelledby": "launch-form-title" },
    h("div", { class: "card-head" }, formTitle, formHint),
    h(
      "div",
      { class: "launch-kind" },
      h("span", { class: "field-label", "aria-hidden": "true" }, "What to start"),
      kindControl,
    ),
    sweepFields,
    field("launch-overrides", "Overrides", overridesInput, ["one ", code("section.key=value"), " per line"]),
    h(
      "div",
      { class: "launch-description" },
      field(
        "launch-description",
        "Description",
        descriptionInput,
        "information only, never read back by training · prefilled, edit freely",
      ),
      resetDescription,
    ),
    h(
      "div",
      { class: "launch-submit" },
      startButton,
      h(
        "p",
        { class: "card-hint" },
        "The config is checked before anything starts. Runs keep training if the dashboard closes.",
      ),
    ),
    formError,
    formNotice,
  );
  const formCard = h("section", { class: "card launch-form-card" }, form);

  const launchesError = h("p", { class: "inline-error", role: "alert", hidden: true });
  const launchesBody = h("div", { class: "launch-table" });
  const launchesCard = h(
    "section",
    { class: "card launch-launches", "aria-labelledby": "launch-launches-title" },
    h(
      "div",
      { class: "card-head" },
      h("h3", { id: "launch-launches-title" }, "Launches"),
      h("span", { class: "card-hint" }, "started from the dashboard; stop a single run from the run list"),
    ),
    launchesError,
    launchesBody,
  );

  root.replaceChildren(h("div", { class: "launch-layout" }, presetCard, formCard), launchesCard);

  // ---------------------------------------------------------------- presets and form
  const kind = () => form.elements["launch-kind"].value;
  const preset = () => view.presets?.presets.find((entry) => entry.path === view.selected) ?? null;

  function suggestion() {
    return suggestedDescription(
      preset()?.description ?? "",
      overrideLines(overridesInput.value),
      view.presets?.bookkeeping_keys ?? [],
    );
  }

  function refreshDescription() {
    if (!view.descriptionEdited) descriptionInput.value = suggestion();
    resetDescription.hidden = !view.descriptionEdited;
  }

  function startLabel(sweep, seeds) {
    if (view.starting) return "Starting…";
    if (!sweep) return "Start run";
    const count = seeds?.length ?? 0;
    return count ? `Start sweep of ${count} seed${count === 1 ? "" : "s"}` : "Start sweep";
  }

  function renderForm() {
    const chosen = preset();
    formTitle.textContent = chosen ? chosen.path : "Pick a preset";
    formHint.replaceChildren(...(chosen ? [code(`train.name: ${chosen.name}`)] : []));
    const sweep = kind() === "sweep";
    sweepFields.hidden = !sweep;
    const seeds = parseSeeds(seedsInput.value);
    seedsInput.setAttribute("aria-invalid", String(sweep && !seeds));
    startButton.textContent = startLabel(sweep, seeds);
    startButton.disabled = !chosen || view.starting;
    formError.hidden = !view.formError;
    formError.textContent = view.formError ?? "";
    formNotice.hidden = !view.formNotice;
    formNotice.replaceChildren(...(view.formNotice ?? []));
  }

  function presetOption(entry) {
    return h(
      "label",
      { class: "launch-preset" },
      h("input", { type: "radio", name: "launch-preset", value: entry.path, checked: entry.path === view.selected }),
      h("span", { class: "launch-preset-name" }, entry.name),
      h("span", { class: "launch-preset-path" }, entry.path),
      entry.description ? h("span", { class: "launch-preset-description" }, entry.description) : null,
    );
  }

  function renderPresets() {
    if (view.presetsError && !view.presets) {
      presetList.replaceChildren(
        h("p", { class: "inline-error" }, `Presets could not be read: ${view.presetsError}`),
        h("button", { type: "button", class: "button", onclick: loadPresets }, "Try again"),
      );
      return;
    }
    if (!view.presets) {
      presetList.replaceChildren(skeletonLine(18), skeletonLine(14), skeletonLine(16));
      return;
    }
    const entries = view.presets.presets;
    if (!entries.length) {
      presetList.replaceChildren(
        h(
          "div",
          { class: "empty launch-empty" },
          h("h3", null, "No presets found"),
          h("p", null, "There is no YAML preset in ", code(view.presets.configs_dir), "."),
          h(
            "p",
            null,
            "Start the dashboard from the repository root, or point it at a folder with ",
            code("snakeppo serve --configs-dir PATH"),
            ".",
          ),
        ),
      );
      return;
    }
    const folders = new Map();
    for (const entry of entries) {
      const cut = entry.path.lastIndexOf("/");
      const folder = cut < 0 ? TOP_LEVEL : entry.path.slice(0, cut);
      if (!folders.has(folder)) folders.set(folder, []);
      folders.get(folder).push(entry);
    }
    const topFirst = ([a], [b]) => (a === TOP_LEVEL ? -1 : b === TOP_LEVEL ? 1 : a.localeCompare(b));
    presetList.replaceChildren(
      h(
        "fieldset",
        { class: "launch-preset-groups" },
        h("legend", { class: "visually-hidden" }, "Preset"),
        [...folders]
          .sort(topFirst)
          .map(([folder, group]) =>
            h("div", { class: "launch-preset-group" }, h("h4", null, folder), group.map(presetOption)),
          ),
      ),
    );
  }

  async function loadPresets() {
    try {
      view.presets = await fetchPresets();
      view.presetsError = null;
    } catch (error) {
      view.presetsError = error.message;
    }
    const paths = (view.presets?.presets ?? []).map((entry) => entry.path);
    if (!paths.includes(view.selected)) {
      view.selected = paths.includes(DEFAULT_PRESET) ? DEFAULT_PRESET : (paths[0] ?? null);
    }
    renderPresets();
    refreshDescription();
    renderForm();
  }

  async function submit(event) {
    event.preventDefault();
    if (view.starting || !preset()) return;
    const { request, error } = launchRequest({
      kind: kind(),
      preset: view.selected,
      overridesText: overridesInput.value,
      description: descriptionInput.value,
      seedsText: seedsInput.value,
      jobsText: jobsInput.value,
    });
    view.formNotice = null;
    view.formError = error ?? null;
    if (error) {
      renderForm();
      return;
    }
    view.starting = true;
    renderForm();
    try {
      const { launch } = await startLaunch(request);
      view.formNotice = [
        `Started ${launch.kind === "sweep" ? "the sweep" : "the run"} (launch `,
        code(launch.id),
        "). It is listed below, and its run folder shows up in the run list within a few seconds.",
      ];
      view.launches = [launch, ...(view.launches ?? []).filter((entry) => entry.id !== launch.id)];
      renderLaunches();
      ctx.refreshRuns().catch(console.error);
      refreshLaunches();
    } catch (failure) {
      view.formError = failure.message;
    } finally {
      view.starting = false;
      renderForm();
    }
  }

  presetList.addEventListener("change", (event) => {
    if (event.target.name !== "launch-preset") return;
    view.selected = event.target.value;
    refreshDescription();
    renderForm();
  });
  form.addEventListener("change", (event) => {
    if (event.target.name === "launch-kind") renderForm();
  });
  form.addEventListener("input", (event) => {
    if (event.target === overridesInput) refreshDescription();
    if (event.target === descriptionInput) {
      view.descriptionEdited = descriptionInput.value !== suggestion();
      resetDescription.hidden = !view.descriptionEdited;
    }
    view.formError = null;
    renderForm();
  });
  resetDescription.addEventListener("click", () => {
    view.descriptionEdited = false;
    refreshDescription();
    descriptionInput.focus();
  });
  form.addEventListener("submit", submit);

  // ---------------------------------------------------------------- launches
  function actionButton(action, entry, label, extraClass = "") {
    return h(
      "button",
      {
        type: "button",
        class: `button ${extraClass}`.trim(),
        dataset: { action, id: entry.id, focusKey: `${action}:${entry.id}` },
        disabled: view.busy.has(entry.id),
      },
      label,
    );
  }

  /** Only a sweep the dashboard started is stopped or resumed as a whole; a run is stopped from the run list. */
  function sweepAction(entry, resumed) {
    if (entry.kind !== "sweep") return null;
    if (entry.alive && entry.stop_requested) {
      return h("span", { class: "muted" }, "Stopping… each seed ends after its current update");
    }
    if (entry.alive) return actionButton("stop", entry, "Stop sweep", "launch-stop");
    if ((entry.state === "stopped" || entry.state === "failed") && !resumed.has(entry.id)) {
      return actionButton("resume", entry, "Resume sweep");
    }
    return null;
  }

  function logTail(entry) {
    const lines = entry.log_tail ?? [];
    if (!lines.length) return null;
    const open = view.openLogs.has(entry.id) || (entry.state === "failed" && !view.closedLogs.has(entry.id));
    return h(
      "details",
      { class: "launch-log", open, dataset: { id: entry.id } },
      h("summary", { dataset: { focusKey: `log:${entry.id}` } }, "Log"),
      h("pre", null, lines.join("\n")),
      h("p", { class: "muted" }, "Full log: ", code(`.launches/${entry.id}/output.log`), " in the runs folder."),
    );
  }

  function launchRow(entry, color, resumed) {
    const { title, detail } = launchSummary(entry);
    const status = STATE_BADGES[entry.state] ?? "unknown";
    return h(
      "tr",
      null,
      h("td", { class: "launch-dot-cell" }, dot(color)),
      h(
        "td",
        null,
        h("span", { class: "launch-what" }, title),
        h("span", { class: "launch-detail" }, detail),
        entry.overrides?.length ? h("span", { class: "launch-detail" }, entry.overrides.join(", ")) : null,
      ),
      h("td", { class: "nowrap num" }, startedText(entry.created)),
      h("td", null, h("span", { class: "badge", dataset: { status } }, entry.state)),
      h("td", { class: "launch-actions" }, sweepAction(entry, resumed), logTail(entry)),
    );
  }

  function renderLaunches() {
    launchesError.hidden = !view.launchesError;
    launchesError.textContent = view.launchesError ? `Launches could not be read: ${view.launchesError}` : "";
    if (!view.visible) return;
    const entries = view.launches;
    const theme = ctx.theme();
    const colors = (entries ?? []).map((entry) => {
      const runId = launchRunId(entry);
      return runId ? runColor(theme, runId) : null;
    });
    const signature = JSON.stringify([entries, colors, [...view.busy], [...view.openLogs], [...view.closedLogs]]);
    if (signature === view.rendered) return;
    view.rendered = signature;
    const focusKey = launchesBody.contains(document.activeElement) ? document.activeElement.dataset.focusKey : null;
    if (entries === null) {
      launchesBody.replaceChildren(skeletonLine(24));
      return;
    }
    if (!entries.length) {
      launchesBody.replaceChildren(
        h(
          "div",
          { class: "empty launch-empty" },
          h("h3", null, "Nothing started from the dashboard yet"),
          h(
            "p",
            null,
            "Pick a preset above and start a run or a seed sweep; it is listed here with its output. ",
            "Runs started in the terminal show up in the run list only.",
          ),
        ),
      );
      return;
    }
    const resumed = new Set(entries.map((entry) => entry.resumes).filter(Boolean));
    const head = h(
      "tr",
      null,
      h("th", { scope: "col" }, h("span", { class: "visually-hidden" }, "Colour")),
      h("th", { scope: "col" }, "What"),
      h("th", { scope: "col" }, "Started"),
      h("th", { scope: "col" }, "State"),
      h("th", { scope: "col" }, h("span", { class: "visually-hidden" }, "Actions")),
    );
    launchesBody.replaceChildren(
      h(
        "div",
        { class: "table-wrap" },
        h(
          "table",
          { class: "table" },
          h("thead", null, head),
          h(
            "tbody",
            null,
            entries.map((entry, index) => launchRow(entry, colors[index], resumed)),
          ),
        ),
      ),
    );
    if (focusKey) launchesBody.querySelector(`[data-focus-key="${CSS.escape(focusKey)}"]`)?.focus();
  }

  async function refreshLaunches() {
    try {
      view.launches = (await fetchLaunches()).launches ?? [];
      view.launchesError = null;
    } catch (error) {
      view.launchesError = error.message;
    }
    renderLaunches();
  }

  function schedule() {
    clearTimeout(view.timer);
    view.timer = view.visible ? setTimeout(poll, LAUNCHES_EVERY_MS) : null;
  }

  async function poll() {
    if (!document.hidden) await refreshLaunches();
    schedule();
  }

  async function act(action, id) {
    view.busy.add(id);
    view.launchesError = null;
    renderLaunches();
    try {
      await (action === "stop" ? stopLaunch(id) : resumeLaunch(id));
      ctx.refreshRuns().catch(console.error);
    } catch (error) {
      view.launchesError = error.message;
    } finally {
      view.busy.delete(id);
      await refreshLaunches();
    }
  }

  launchesBody.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-action]");
    if (button && !button.disabled) act(button.dataset.action, button.dataset.id);
  });
  // `toggle` does not bubble, so listen in the capture phase to remember what the researcher opened.
  launchesBody.addEventListener(
    "toggle",
    (event) => {
      const id = event.target.dataset?.id;
      if (!id || !event.target.matches("details")) return;
      if (event.target.open) {
        view.openLogs.add(id);
        view.closedLogs.delete(id);
      } else {
        view.openLogs.delete(id);
        view.closedLogs.add(id);
      }
    },
    true,
  );

  renderPresets();
  renderForm();

  return {
    show() {
      if (view.visible) return;
      view.visible = true;
      view.rendered = "";
      loadPresets();
      renderLaunches();
      poll();
    },
    hide() {
      view.visible = false;
      clearTimeout(view.timer);
      view.timer = null;
    },
    onRuns() {
      // The run list refreshed: repaint the dots in case the theme changed since the last poll.
      renderLaunches();
    },
  };
}
