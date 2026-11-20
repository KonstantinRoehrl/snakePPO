// The dashboard's JSON API (served by snakeppo.dashboard.server) and the incremental metrics
// store. Reads are GETs; actions (launch, stop, resume, replay) are JSON POSTs. Every request goes
// to the page's own origin, which is what the server's origin check expects.

export class ApiError extends Error {
  /** `status` is the HTTP status, or 0 when the server could not be reached at all. */
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }

  get unreachable() {
    return this.status === 0;
  }
}

async function getJson(path) {
  return request(path, { headers: { Accept: "application/json" }, cache: "no-store" });
}

/** POST `body` as JSON; resolves to the JSON reply, rejects with an `ApiError` on non-2xx. */
export async function postJson(path, body) {
  return request(path, {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
    cache: "no-store",
    credentials: "same-origin",
  });
}

async function request(path, init) {
  let response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiError("The dashboard server cannot be reached.", 0);
  }
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const message =
      body && typeof body.error === "string" ? body.error : response.statusText || `HTTP ${response.status}`;
    throw new ApiError(message, response.status);
  }
  return body;
}

const enc = encodeURIComponent;

/** `{runs: [summary], skipped: [{id, error}]}` */
export const fetchRuns = () => getJson("/api/runs");

/** `{run: summary, config, info}` */
export const fetchRun = (id) => getJson(`/api/runs/${enc(id)}`);

/** `{lines, offset, reset}`: the complete metrics lines after byte `since`. */
export const fetchMetrics = (id, since) => getJson(`/api/runs/${enc(id)}/metrics?since=${since}`);

/** `{rows, left_out, baseline}`; an unknown `baseline` group is a 400 `ApiError`. */
export const fetchResults = (baseline) =>
  getJson(baseline ? `/api/results?baseline=${enc(baseline)}` : "/api/results");

/** `{base, differences: {id: {key: [base, run]}}}` */
export const fetchDiff = (base, ids) =>
  getJson(`/api/diff?base=${enc(base)}&runs=${ids.map(enc).join(",")}`);

/** `{presets: [{path, name, description}], bookkeeping_keys: [key], configs_dir}` */
export const fetchPresets = () => getJson("/api/presets");

/** `{launches: [LaunchEntry]}`, newest first. */
export const fetchLaunches = () => getJson("/api/launches");

/** Start a train run or a seed sweep; `{launch: LaunchEntry}`. An invalid config is a 400. */
export const startLaunch = (request) => postJson("/api/launch", request);

/** Ask a running run to stop at its next update boundary; `{ok: true}`. */
export const stopRun = (id) => postJson(`/api/runs/${enc(id)}/stop`, {});

/** Continue an interrupted, stalled or failed run; `{launch: LaunchEntry}`. */
export const resumeRun = (id) => postJson(`/api/runs/${enc(id)}/resume`, {});

/** Stop a browser-launched sweep and every seed it is training; `{ok: true}`. */
export const stopLaunch = (id) => postJson(`/api/launches/${enc(id)}/stop`, {});

/** Continue a stopped sweep as the same seed group; `{launch: LaunchEntry}`. */
export const resumeLaunch = (id) => postJson(`/api/launches/${enc(id)}/resume`, {});

/** Replace the replay session with one playing `boards` ([{run, checkpoint, greedy, board_size}]); `{session}`. */
export const createReplay = (boards, speed) => postJson("/api/replay", { boards, speed });

/** `action`: pause, resume, step, speed (with `value`) or restart; `{paused, speed}`. */
export const controlReplay = (id, action, value) =>
  postJson(`/api/replay/${enc(id)}/control`, { action, value });

/** The `EventSource` URL of a replay session's frames. */
export const replayStreamUrl = (id) => `/api/replay/${enc(id)}/stream`;

/**
 * Each run's metrics lines and the byte offset to continue from. `poll(id)` fetches what was
 * appended since the last poll and returns `{appended, reset}`; on a reset (the server says the
 * file was rewritten, or a line's `update` does not move forward) the run's lines are dropped and
 * read again from the start, so callers rebuild that run's traces from `lines(id)`.
 */
export class MetricsStore {
  constructor(fetchChunk = fetchMetrics) {
    this.fetchChunk = fetchChunk;
    this.runs = new Map();
    this.pending = new Map();
  }

  lines(id) {
    return this.runs.get(id)?.lines ?? [];
  }

  /** Whether `id` has been polled successfully at least once. */
  loaded(id) {
    return this.runs.has(id);
  }

  forget(id) {
    this.runs.delete(id);
  }

  /** One poll per run at a time: a second call while one is in flight shares its result. */
  poll(id) {
    if (!this.pending.has(id)) {
      const request = this.#poll(id).finally(() => this.pending.delete(id));
      this.pending.set(id, request);
    }
    return this.pending.get(id);
  }

  async #poll(id) {
    const known = this.runs.get(id) ?? { offset: 0, lines: [] };
    let chunk = await this.fetchChunk(id, known.offset);
    let reset = Boolean(chunk.reset);
    if (!reset && !movesForward(known.lines, chunk.lines)) {
      reset = true;
      if (known.offset > 0) chunk = await this.fetchChunk(id, 0);
    }
    const lines = reset ? chunk.lines.slice() : known.lines.concat(chunk.lines);
    this.runs.set(id, { offset: chunk.offset, lines });
    return { appended: chunk.lines.length, reset };
  }
}

function movesForward(known, incoming) {
  let last = known.length ? known[known.length - 1].update : -Infinity;
  for (const line of incoming) {
    if (!(line.update > last)) return false;
    last = line.update;
  }
  return true;
}
