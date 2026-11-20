// Stable run colours: a run's colour comes from a hash of its id, so every view (run list, Live,
// Compare, Results, Launch, Replay) agrees and a run never changes colour when the selection does.
// Two runs may share a colour; the legend and the hover text still tell them apart.

/** Number of `--run-N` tokens in style.css (light and dark). */
export const RUN_PALETTE_SIZE = 7;

const FNV_OFFSET = 0x811c9dc5;
const FNV_PRIME = 0x01000193;

/** FNV-1a (32-bit) over the UTF-16 code units of `runId`, reduced to a palette slot. */
export function runColorIndex(runId) {
  let hash = FNV_OFFSET;
  const text = String(runId);
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, FNV_PRIME) >>> 0;
  }
  return hash % RUN_PALETTE_SIZE;
}

/** The run's `#rrggbb` colour from a theme that `charts.readTheme` read. */
export function runColor(theme, runId) {
  return theme.runColors[runColorIndex(runId)];
}

/** The run's colour as a CSS value that follows theme changes without a redraw. */
export function runColorVar(runId) {
  return `var(--run-${runColorIndex(runId) + 1})`;
}
