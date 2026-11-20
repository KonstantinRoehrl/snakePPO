// The Launch view: start a training run or a seed sweep from a configs/ preset, and follow the
// launches started from the dashboard. app.js mounts it once and shows it on the Launch tab.

/**
 * Mount the view into `root`. `ctx` is the app's context: `runs()`, `theme()`, `refreshRuns()`
 * and `setHash(fields)`. Returns `{show(params), hide(), onRuns(runs)}`.
 */
export function mountLaunch(root, ctx) {
  const note = document.createElement("p");
  note.className = "muted";
  note.textContent = "Starting runs from the browser is not part of this build yet.";
  root.replaceChildren(note);
  return {
    show() {},
    hide() {},
    onRuns() {},
  };
}
