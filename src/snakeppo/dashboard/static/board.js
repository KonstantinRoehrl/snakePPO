// Drawing one replay board on a canvas. Colours come from the board tokens in replay.css.

// The tail's body order in snakeppo.observation (BODY_ORDER_TAIL); the head is 1.0.
const BODY_ORDER_TAIL = 0.1;
// Cell codes of a replay frame's `cells` string (snakeppo.dashboard.replay CELL_*).
const CELL_WALL = "1";
const CELL_FOOD = "2";

/** The board colours from the CSS custom properties; call again after a theme change. */
export function readBoardTheme(element = document.documentElement) {
  const style = getComputedStyle(element);
  const token = (name) => style.getPropertyValue(name).trim();
  return {
    background: token("--board-bg"),
    grid: token("--board-grid"),
    wall: token("--wall"),
    food: token("--food"),
    head: token("--head"),
    near: token("--snake-near"),
    tail: token("--snake-tail"),
  };
}

function channels(color) {
  const match = /^#([0-9a-f]{6})$/i.exec(color);
  if (!match) return [127, 127, 127];
  const value = parseInt(match[1], 16);
  return [value >> 16, (value >> 8) & 255, value & 255];
}

/**
 * The colour of body segment `index` (0 is the head) of a snake `length` cells long: its body
 * order 1 - 0.9 * index / (length - 1) mapped linearly from `theme.near` (order 1.0) to
 * `theme.tail` (order 0.1), the rule the observation's body-order channel uses.
 */
export function bodyShade(index, length, theme) {
  const order = length > 1 ? 1 - ((1 - BODY_ORDER_TAIL) * index) / (length - 1) : 1;
  const fraction = Math.min(Math.max((1 - order) / (1 - BODY_ORDER_TAIL), 0), 1);
  const near = channels(theme.near);
  const tail = channels(theme.tail);
  const [red, green, blue] = near.map((value, i) => Math.round(value + (tail[i] - value) * fraction));
  return `rgb(${red}, ${green}, ${blue})`;
}

function cellRect(context, x, y, side, inset, radius) {
  context.beginPath();
  if (typeof context.roundRect === "function") {
    context.roundRect(x + inset, y + inset, side - 2 * inset, side - 2 * inset, radius);
  } else {
    context.rect(x + inset, y + inset, side - 2 * inset, side - 2 * inset);
  }
  context.fill();
}

/**
 * Draw a replay frame's board (`{size, cells, body}`) centred in `canvas` at its current pixel
 * size: background, grid lines, the wall ring, the food dot, then the snake tail to head so the
 * head is drawn last and stays visible.
 */
export function drawBoard(canvas, board, theme) {
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  context.fillStyle = theme.background;
  context.fillRect(0, 0, width, height);
  const size = board.size;
  const cell = Math.floor(Math.min(width, height) / size);
  if (cell < 1) return;
  const left = Math.floor((width - cell * size) / 2);
  const top = Math.floor((height - cell * size) / 2);
  const x = (index) => left + (index % size) * cell;
  const y = (index) => top + Math.floor(index / size) * cell;

  context.fillStyle = theme.grid;
  for (let line = 0; line <= size; line += 1) {
    context.fillRect(left + line * cell, top, 1, cell * size);
    context.fillRect(left, top + line * cell, cell * size, 1);
  }

  context.fillStyle = theme.wall;
  for (let index = 0; index < board.cells.length; index += 1) {
    if (board.cells[index] === CELL_WALL) context.fillRect(x(index), y(index), cell, cell);
  }
  context.fillStyle = theme.food;
  for (let index = 0; index < board.cells.length; index += 1) {
    if (board.cells[index] !== CELL_FOOD) continue;
    context.beginPath();
    context.arc(x(index) + cell / 2, y(index) + cell / 2, cell * 0.35, 0, 2 * Math.PI);
    context.fill();
  }

  const inset = Math.max(1, Math.round(cell * 0.08));
  const radius = cell * 0.25;
  for (let segment = board.body.length - 1; segment >= 0; segment -= 1) {
    const index = board.body[segment];
    const head = segment === 0;
    context.fillStyle = head ? theme.head : bodyShade(segment, board.body.length, theme);
    cellRect(context, x(index), y(index), cell, head ? Math.max(0, inset - 1) : inset, radius);
  }
}
