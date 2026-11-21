// The Replay tab: up to four boards of trained models playing side by side. The server steps the
// games (snakeppo.dashboard.replay) and streams one frame per step over SSE; this view picks the
// boards, sends the controls and draws each frame.

import { controlReplay, createReplay, replayStreamUrl } from "./api.js";
import { drawBoard, readBoardTheme } from "./board.js";
import { runColorIndex } from "./colors.js";

// The server's REPLAY_SPEEDS, DEFAULT_SPEED and MAX_BOARDS; 0 plays as fast as the CPU allows.
export const REPLAY_SPEEDS = [2, 5, 10, 25, 60, 0];
export const DEFAULT_SPEED = 10;
export const MAX_BOARDS = 4;
const CHECKPOINTS = ["best", "last"];
const MIN_BOARD_SIZE = 5;
// Offered to agents that play any board size, besides the size each run trained on.
const BOARD_SIZES = [8, 10, 12, 16, 20, 24, 32];
const BOARD_BOUND = "This agent only plays the board it trained on.";

/** One board of the replay hash: `encodeURIComponent("run|checkpoint|greedy|size")`. */
export function replayToken({ run, checkpoint, greedy, board_size }) {
  return encodeURIComponent(
    [run, checkpoint, greedy ? "greedy" : "stochastic", board_size ?? ""].join("|"),
  );
}

/** The board a hash token names, or null when it names no run. Bad fields fall back to defaults. */
export function parseReplayToken(token) {
  let text = token;
  try {
    text = decodeURIComponent(token);
  } catch {
    // An undecodable token is read as it stands.
  }
  const parts = text.split("|");
  if (parts.length < 4) return null;
  const size = parts.pop();
  const mode = parts.pop();
  const checkpoint = parts.pop();
  const run = parts.join("|");
  if (!run) return null;
  const boardSize = /^\d+$/.test(size) && Number(size) >= MIN_BOARD_SIZE ? Number(size) : null;
  return {
    run,
    checkpoint: CHECKPOINTS.includes(checkpoint) ? checkpoint : "best",
    greedy: mode !== "stochastic",
    board_size: boardSize,
  };
}

/** `{boards, speed}` from the raw hash params (`boards=<t1>,<t2>&speed=<n>`). */
export function parseReplayParams(params) {
  const boards = String(params?.boards ?? "")
    .split(",")
    .filter(Boolean)
    .map(parseReplayToken)
    .filter(Boolean)
    .slice(0, MAX_BOARDS);
  const speed = Number(params?.speed);
  return {
    boards,
    speed: params?.speed !== undefined && REPLAY_SPEEDS.includes(speed) ? speed : DEFAULT_SPEED,
  };
}

const speedLabel = (speed) => (speed === 0 ? "Max" : `${speed}/s`);

function h(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "checked" || key === "value" || key === "selected" || key === "disabled") {
      node[key] = value;
    } else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function segmented(legend, name, options, current, onChange) {
  return h(
    "fieldset",
    { class: "segmented" },
    h("legend", null, legend),
    options.map(([value, label]) =>
      h(
        "label",
        null,
        h("input", {
          type: "radio",
          name,
          value: String(value),
          checked: value === current,
          dataset: { focusKey: `${name}:${value}` },
          onchange: () => onChange(value),
        }),
        h("span", null, label),
      ),
    ),
  );
}

function stat(label, value) {
  return h(
    "div",
    { class: "replay-stat" },
    h("span", { class: "replay-stat-label" }, label),
    h("span", { class: "replay-stat-value num" }, value),
  );
}

const sameBoards = (a, b) => JSON.stringify(a) === JSON.stringify(b);

/**
 * Mount the Replay view in `root`. `ctx` is the app's view context: `runs()` (the run list,
 * null until loaded), `setHash(fields)`. Returns `{show(params), hide(), onRuns(runs)}`.
 */
export function mountReplay(root, ctx) {
  let boards = []; // [{run, checkpoint, greedy, board_size}]
  let speed = DEFAULT_SPEED;
  let paused = false;
  let visible = false;
  let session = null; // the server's session id
  let source = null; // the EventSource of the session's stream
  let generation = 0; // bumped on every new session or hide; stale replies are dropped
  let frame = null; // the latest frame, redrawn after a resize or a rebuild
  let drawPending = false;
  let canvases = [];
  let statsNodes = [];
  let runsSignature = "";

  const pauseButton = h("button", { type: "button", class: "button", onclick: togglePause }, "Pause");
  const stepButton = h(
    "button",
    { type: "button", class: "button", disabled: true, onclick: () => control("step") },
    "Step ›",
  );
  const restartButton = h(
    "button",
    { type: "button", class: "button", onclick: () => control("restart") },
    "↺ Restart",
  );
  const speedControl = h("div", { class: "replay-speed" });
  const status = h("span", { class: "replay-status muted", role: "status" });
  const error = h("p", { class: "inline-error", role: "alert", hidden: true });
  const notice = h("div", { class: "callout replay-notice", hidden: true });
  const grid = h("div", { class: "replay-boards" });
  root.replaceChildren(
    h(
      "div",
      { class: "card replay-toolbar" },
      pauseButton,
      stepButton,
      restartButton,
      speedControl,
      h("span", { class: "replay-hint muted" }, "Plays on the CPU · a new game starts after each death"),
      status,
    ),
    error,
    notice,
    grid,
  );

  const resize = new ResizeObserver(() => {
    fitCanvases();
    scheduleDraw();
  });
  resize.observe(grid);

  function renderSpeed() {
    speedControl.replaceChildren(
      h("span", { class: "field-label", "aria-hidden": "true" }, "Speed"),
      segmented(
        "Speed",
        "replay-speed",
        REPLAY_SPEEDS.map((value) => [value, speedLabel(value)]),
        speed,
        (value) => {
          speed = value;
          writeHash();
          control("speed", value);
        },
      ),
    );
  }

  function renderControls() {
    pauseButton.textContent = paused ? "Resume" : "Pause";
    pauseButton.disabled = !session;
    stepButton.disabled = !session || !paused;
    restartButton.disabled = !session;
    for (const input of speedControl.querySelectorAll("input")) {
      input.checked = Number(input.value) === speed;
    }
  }

  function setStatus(text) {
    if (status.textContent !== text) status.textContent = text;
  }

  function showError(message) {
    error.textContent = message;
    error.hidden = !message;
  }

  function showNotice(message) {
    notice.hidden = !message;
    if (!message) return notice.replaceChildren();
    notice.replaceChildren(
      h("p", null, message),
      h("button", { type: "button", class: "button", onclick: startSession }, "Start again"),
    );
  }

  function writeHash() {
    ctx.setHash({ boards: boards.map(replayToken).join(","), speed: String(speed) });
  }

  // ---------------------------------------------------------------- boards
  function runsList() {
    return ctx.runs() ?? [];
  }

  function runsKey() {
    return runsList()
      .map((run) => `${run.id}:${run.status}:${run.size_agnostic}:${run.board_size}`)
      .join(",");
  }

  function runById(id) {
    return runsList().find((run) => run.id === id) ?? null;
  }

  function changeBoard(index, fields) {
    boards = boards.map((board, at) => (at === index ? { ...board, ...fields } : board));
    boardsChanged();
  }

  function removeBoard(index) {
    boards = boards.filter((_, at) => at !== index);
    boardsChanged([`remove:${index - 1}`, "add"]);
  }

  function addBoard() {
    const last = boards[boards.length - 1];
    const run = last?.run ?? runsList()[0]?.id;
    if (!run || boards.length >= MAX_BOARDS) return;
    boards = [...boards, { run, checkpoint: "best", greedy: true, board_size: null }];
    boardsChanged();
  }

  function boardsChanged(focusFallbacks) {
    writeHash();
    renderBoards(focusFallbacks);
    startSession();
  }

  function runSelect(board, index) {
    const runs = runsList();
    const known = runs.some((run) => run.id === board.run);
    return h(
      "select",
      {
        class: "select replay-run",
        "aria-label": `Run on board ${index + 1}`,
        dataset: { focusKey: `run:${index}` },
        onchange: (event) => changeBoard(index, { run: event.target.value, board_size: null }),
      },
      known ? null : h("option", { value: board.run, selected: true }, `${board.run} (not in the list)`),
      runs.map((run) =>
        h("option", { value: run.id, selected: run.id === board.run }, `${run.id} · ${run.status}`),
      ),
    );
  }

  function sizeSelect(board, run, index) {
    const trained = run?.board_size ?? null;
    const bound = run?.size_agnostic === false;
    const sizes = [...new Set([...BOARD_SIZES, ...(board.board_size ? [board.board_size] : [])])]
      .filter((size) => size !== trained)
      .sort((a, b) => a - b);
    const select = h(
      "select",
      {
        class: "select replay-size",
        disabled: bound,
        title: bound ? BOARD_BOUND : null,
        dataset: { focusKey: `size:${index}` },
        onchange: (event) =>
          changeBoard(index, { board_size: event.target.value ? Number(event.target.value) : null }),
      },
      h(
        "option",
        { value: "", selected: board.board_size === null },
        trained ? `${trained} (trained)` : "trained size",
      ),
      bound
        ? null
        : sizes.map((size) =>
            h("option", { value: String(size), selected: size === board.board_size }, size),
          ),
    );
    return h(
      "label",
      { class: "replay-size-field" },
      h("span", { class: "field-label" }, "Board"),
      select,
    );
  }

  function boardCard(board, index) {
    const run = runById(board.run);
    const canvas = h("canvas", {
      class: "replay-canvas",
      role: "img",
      "aria-label": `Board ${index + 1}: ${board.run}`,
    });
    const stats = h("div", { class: "replay-stats" });
    canvases.push(canvas);
    statsNodes.push(stats);
    return h(
      "article",
      { class: "replay-board" },
      h(
        "div",
        { class: "replay-board-head" },
        h("span", {
          class: "replay-dot",
          style: `background: var(--run-${runColorIndex(board.run) + 1})`,
          "aria-hidden": "true",
        }),
        runSelect(board, index),
        h(
          "button",
          {
            type: "button",
            class: "button button-quiet replay-remove",
            "aria-label": `Remove board ${index + 1}`,
            title: "Remove this board",
            dataset: { focusKey: `remove:${index}` },
            onclick: () => removeBoard(index),
          },
          "✕",
        ),
      ),
      h(
        "div",
        { class: "replay-board-options" },
        segmented(
          "Checkpoint",
          `replay-checkpoint-${index}`,
          CHECKPOINTS.map((name) => [name, name]),
          board.checkpoint,
          (value) => changeBoard(index, { checkpoint: value }),
        ),
        segmented(
          "Policy",
          `replay-mode-${index}`,
          [
            [true, "Greedy"],
            [false, "Stochastic"],
          ],
          board.greedy,
          (value) => changeBoard(index, { greedy: value }),
        ),
        sizeSelect(board, run, index),
      ),
      run?.size_agnostic === false ? h("p", { class: "replay-note muted" }, BOARD_BOUND) : null,
      canvas,
      stats,
    );
  }

  function addCard() {
    const disabled = runsList().length === 0;
    return h(
      "button",
      {
        type: "button",
        class: "replay-add",
        disabled,
        dataset: { focusKey: "add" },
        onclick: addBoard,
      },
      h("span", { class: "replay-add-title" }, "＋ Add a board"),
      h(
        "span",
        { class: "replay-add-hint muted" },
        disabled
          ? "No runs yet: train one first"
          : `up to ${MAX_BOARDS} · pick any run, the same run may appear twice`,
      ),
    );
  }

  // focusFallbacks: keys to try, in order, when the focused control no longer exists.
  function renderBoards(focusFallbacks = []) {
    const focused = document.activeElement?.dataset?.focusKey;
    canvases = [];
    statsNodes = [];
    grid.replaceChildren(
      ...boards.map(boardCard),
      boards.length < MAX_BOARDS ? addCard() : null,
    );
    for (let index = 0; index < boards.length; index += 1) renderStats(index);
    runsSignature = runsKey();
    const focusTarget = [focused, ...focusFallbacks]
      .filter(Boolean)
      .map((key) => grid.querySelector(`[data-focus-key="${CSS.escape(key)}"]`))
      .find((node) => node && !node.disabled);
    focusTarget?.focus();
    fitCanvases();
    scheduleDraw();
  }

  function renderStats(index) {
    const node = statsNodes[index];
    const board = frame?.boards?.[index];
    if (!node) return;
    const games = board?.episodes ?? 0;
    const mean = typeof board?.mean_score === "number" ? board.mean_score.toFixed(1) : "-";
    node.replaceChildren(
      stat("Score", board?.score ?? "-"),
      stat("Length", board?.length ?? "-"),
      stat("Since apple", board?.since_apple ?? "-"),
      stat("Last end", board?.last_end ?? "-"),
      stat(`Mean · ${games} game${games === 1 ? "" : "s"}`, mean),
    );
  }

  function fitCanvases() {
    const ratio = window.devicePixelRatio || 1;
    for (const canvas of canvases) {
      const side = Math.round(canvas.clientWidth * ratio);
      if (side > 0 && canvas.width !== side) {
        canvas.width = side;
        canvas.height = side;
      }
    }
  }

  function scheduleDraw() {
    if (drawPending || !frame) return;
    drawPending = true;
    requestAnimationFrame(() => {
      drawPending = false;
      if (!frame || !visible) return;
      const theme = readBoardTheme(document.documentElement);
      frame.boards.forEach((board, index) => {
        if (canvases[index]) drawBoard(canvases[index], board, theme);
        renderStats(index);
      });
    });
  }

  // ---------------------------------------------------------------- session and stream
  function closeStream() {
    source?.close();
    source = null;
  }

  async function startSession() {
    const mine = ++generation;
    closeStream();
    session = null;
    frame = null;
    paused = false;
    showError("");
    showNotice("");
    renderControls();
    if (!boards.length) {
      setStatus("");
      return;
    }
    setStatus("Loading the models…");
    let reply;
    try {
      reply = await createReplay(boards, speed);
    } catch (problem) {
      if (mine !== generation) return;
      setStatus("");
      showError(problem.message);
      return;
    }
    if (mine !== generation) return;
    session = reply.session.id;
    paused = Boolean(reply.session.paused);
    speed = reply.session.speed;
    renderControls();
    openStream(mine);
  }

  function openStream(mine) {
    const stream = new EventSource(replayStreamUrl(session));
    source = stream;
    setStatus("Connecting…");
    stream.onmessage = (event) => {
      if (mine !== generation) return;
      frame = JSON.parse(event.data);
      if (frame.paused !== paused || frame.speed !== speed) {
        paused = frame.paused;
        speed = frame.speed;
        renderControls();
      }
      setStatus(paused ? "Paused" : "Playing");
      scheduleDraw();
    };
    stream.addEventListener("end", () => {
      if (mine !== generation) return;
      ended("This replay ended: another replay started or the server stopped.");
    });
    stream.onerror = () => {
      if (mine !== generation) return;
      if (stream.readyState === EventSource.CLOSED) {
        ended("The replay stream closed: the session is gone or the server cannot be reached.");
      } else {
        setStatus("Reconnecting…");
      }
    };
  }

  function ended(message) {
    closeStream();
    session = null;
    renderControls();
    setStatus("");
    showNotice(message);
  }

  async function control(action, value) {
    if (!session) return;
    const mine = generation;
    try {
      const reply = await controlReplay(session, action, value);
      if (mine !== generation) return;
      paused = reply.paused;
      speed = reply.speed;
      renderControls();
      setStatus(paused ? "Paused" : "Playing");
    } catch (problem) {
      if (mine !== generation) return;
      if (problem.status === 404) startSession();
      else showError(problem.message);
    }
  }

  function togglePause() {
    control(paused ? "resume" : "pause");
  }

  renderSpeed();
  renderControls();

  return {
    show(params) {
      const wanted = parseReplayParams(params);
      const wasVisible = visible;
      visible = true;
      if (wasVisible && sameBoards(wanted.boards, boards) && wanted.speed === speed && session) return;
      boards = wanted.boards;
      speed = wanted.speed;
      renderSpeed();
      renderBoards();
      startSession();
    },
    hide() {
      visible = false;
      generation += 1;
      closeStream();
      session = null;
      renderControls();
      setStatus("");
    },
    onRuns() {
      if (visible && runsKey() !== runsSignature) renderBoards();
    },
  };
}
