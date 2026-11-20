// The Replay view: up to four boards playing trained models side by side, streamed from the
// server's replay session. app.js mounts it once and shows it on the Replay tab.
//
// Hash: #tab=replay&boards=<token>,<token>&speed=<steps per second>, one token per board.

/**
 * One board as a hash token: `encodeURIComponent("run|checkpoint|greedy or stochastic|size")`,
 * with an empty size for the board the model trained on.
 */
export function replayToken({ run, checkpoint, greedy, board_size }) {
  const size = board_size === null || board_size === undefined ? "" : String(board_size);
  return encodeURIComponent([run, checkpoint, greedy ? "greedy" : "stochastic", size].join("|"));
}

/**
 * Mount the view into `root`. `ctx` is the app's context: `runs()`, `theme()`, `refreshRuns()`
 * and `setHash(fields)`. Returns `{show(params), hide(), onRuns(runs)}`; `params` holds the raw
 * hash values (`boards`, `speed`).
 */
export function mountReplay(root, ctx) {
  const note = document.createElement("p");
  note.className = "muted";
  note.textContent = "Watching trained models play in the browser is not part of this build yet.";
  root.replaceChildren(note);
  return {
    show() {},
    hide() {},
    onRuns() {},
  };
}
