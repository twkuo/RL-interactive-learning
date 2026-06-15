// LunarLander rendering: terrain with a green landing pad between two flags, and the lander
// (body + legs) drawn at its position/angle with engine flames for the last action.
import { useEffect, useRef } from 'react';
import { useStore } from '../../state/store';
import type { LunarRenderState } from '../../envs/continuous/LunarLander';

const W = 520;
const H = 300;
const X_RANGE = 1.5;
const Y_RANGE = 1.5;

export function LunarLanderRenderer() {
  const ref = useRef<HTMLCanvasElement>(null);
  const env = useStore((s) => s.env);
  const tick = useStore((s) => s.tick);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr;
    cv.height = H * dpr;
    cv.style.width = `${W}px`;
    cv.style.height = `${H}px`;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const rs = env.getRenderState() as LunarRenderState;
    const groundY = H - 28;
    const sx = (W - 40) / (2 * X_RANGE);
    const sy = (groundY - 26) / Y_RANGE;
    const X = (wx: number) => W / 2 + wx * sx;
    const Y = (wy: number) => groundY - wy * sy;

    // Ground
    ctx.strokeStyle = '#3a4150';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, groundY);
    ctx.lineTo(W, groundY);
    ctx.stroke();

    // Landing pad (between flags)
    const padL = X(-rs.padHalf);
    const padR = X(rs.padHalf);
    ctx.strokeStyle = '#2e9e6b';
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(padL, groundY);
    ctx.lineTo(padR, groundY);
    ctx.stroke();
    ctx.fillStyle = '#2e9e6b';
    for (const px of [padL, padR]) {
      ctx.fillRect(px - 1, groundY - 22, 2, 22);
      ctx.beginPath();
      ctx.moveTo(px, groundY - 22);
      ctx.lineTo(px + (px === padL ? 12 : -12), groundY - 17);
      ctx.lineTo(px, groundY - 12);
      ctx.fill();
    }

    // Lander
    ctx.save();
    ctx.translate(X(rs.x), Y(rs.y));
    ctx.rotate(-rs.angle);
    // Flames (drawn first, behind the body)
    ctx.fillStyle = '#ffb347';
    if (rs.lastAction === 2) {
      ctx.beginPath();
      ctx.moveTo(-6, 8);
      ctx.lineTo(6, 8);
      ctx.lineTo(0, 26);
      ctx.fill();
    }
    if (rs.lastAction === 1) {
      ctx.beginPath();
      ctx.moveTo(-12, -2);
      ctx.lineTo(-12, 8);
      ctx.lineTo(-26, 3);
      ctx.fill();
    }
    if (rs.lastAction === 3) {
      ctx.beginPath();
      ctx.moveTo(12, -2);
      ctx.lineTo(12, 8);
      ctx.lineTo(26, 3);
      ctx.fill();
    }
    // Legs
    ctx.strokeStyle = '#8a90a0';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(-11, 8);
    ctx.lineTo(-18, 18);
    ctx.moveTo(11, 8);
    ctx.lineTo(18, 18);
    ctx.stroke();
    // Body
    ctx.fillStyle = rs.landed ? '#7dffb0' : rs.crashed ? '#e06c75' : '#cbd5e1';
    ctx.fillRect(-12, -12, 24, 20);
    ctx.restore();

    // HUD
    ctx.fillStyle = rs.landed ? '#7dffb0' : rs.crashed ? '#e06c75' : '#cbd5e1';
    ctx.font = '13px ui-monospace, monospace';
    ctx.textAlign = 'center';
    const status = rs.landed ? 'LANDED' : rs.crashed ? 'CRASHED' : 'flying';
    ctx.fillText(`${status}   x = ${rs.x.toFixed(2)}   alt = ${rs.y.toFixed(2)}`, W / 2, 20);
  }, [env, tick]);

  return <canvas ref={ref} style={{ borderRadius: 8, display: 'block', maxWidth: '100%' }} />;
}
