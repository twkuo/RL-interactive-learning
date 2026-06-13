// Canvas 2D drawing helpers.
export type Pt = [number, number];

export function fillTriangle(
  ctx: CanvasRenderingContext2D,
  pts: [Pt, Pt, Pt],
  fill: string,
  stroke?: string,
): void {
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  ctx.lineTo(pts[1][0], pts[1][1]);
  ctx.lineTo(pts[2][0], pts[2][1]);
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

export function drawArrow(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  color: string,
  width = 3,
): void {
  const head = 8;
  const ang = Math.atan2(y2 - y1, x2 - x1);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(ang - Math.PI / 6), y2 - head * Math.sin(ang - Math.PI / 6));
  ctx.lineTo(x2 - head * Math.cos(ang + Math.PI / 6), y2 - head * Math.sin(ang + Math.PI / 6));
  ctx.closePath();
  ctx.fill();
}

export function strokeRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  color: string,
  width: number,
): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.strokeRect(x + width / 2, y + width / 2, w - width, h - width);
}

export function strokeTriangle(
  ctx: CanvasRenderingContext2D,
  pts: [Pt, Pt, Pt],
  color: string,
  width: number,
): void {
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  ctx.lineTo(pts[1][0], pts[1][1]);
  ctx.lineTo(pts[2][0], pts[2][1]);
  ctx.closePath();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineJoin = 'round';
  ctx.stroke();
}

interface TextOpts {
  color?: string;
  size?: number;
  align?: CanvasTextAlign;
  baseline?: CanvasTextBaseline;
  bold?: boolean;
}

// Text with a dark outline, ensuring it stays legible on any heatmap background color.
export function outlinedText(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  opts: TextOpts = {},
): void {
  const { color = '#fff', size = 10, align = 'center', baseline = 'middle', bold = false } = opts;
  ctx.font = `${bold ? '700 ' : ''}${size}px ui-monospace, monospace`;
  ctx.textAlign = align;
  ctx.textBaseline = baseline;
  ctx.lineJoin = 'round';
  ctx.lineWidth = 3;
  ctx.strokeStyle = 'rgba(0,0,0,0.85)';
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}
