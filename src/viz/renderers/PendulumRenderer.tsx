// Pendulum rendering: a rod hinged at the center. θ = 0 points straight up (the goal); the bob
// turns green near upright. A bar at the bottom shows the applied torque (the continuous action).
import { useEffect, useRef } from 'react';
import { useStore } from '../../state/store';
import type { PendulumRenderState } from '../../envs/continuous/Pendulum';

const W = 520;
const H = 300;
const L = 110;

export function PendulumRenderer() {
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

    const rs = env.getRenderState() as PendulumRenderState;
    const cx = W / 2;
    const cy = H / 2 - 10;
    const th = rs.theta;
    const ex = cx + L * Math.sin(th);
    const ey = cy - L * Math.cos(th);
    const norm = Math.atan2(Math.sin(th), Math.cos(th)); // [-π, π], 0 = up
    const upright = Math.abs(norm) < 0.25;

    // Target marker at the top (upright goal).
    ctx.fillStyle = 'rgba(46,158,107,0.5)';
    ctx.beginPath();
    ctx.arc(cx, cy - L, 6, 0, Math.PI * 2);
    ctx.fill();

    // Rod
    ctx.strokeStyle = upright ? '#7dffb0' : '#5b9bd5';
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    // Pivot + bob
    ctx.fillStyle = '#cbd5e1';
    ctx.beginPath();
    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = upright ? '#7dffb0' : '#ffd54a';
    ctx.beginPath();
    ctx.arc(ex, ey, 12, 0, Math.PI * 2);
    ctx.fill();

    // Torque bar (the continuous action), centered at the bottom.
    const barY = H - 26;
    const barHalf = 120;
    ctx.strokeStyle = '#3a4150';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - barHalf, barY);
    ctx.lineTo(cx + barHalf, barY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx, barY - 6);
    ctx.lineTo(cx, barY + 6);
    ctx.stroke();
    const frac = Math.max(-1, Math.min(1, rs.torque / rs.maxTorque));
    ctx.strokeStyle = '#e0a458';
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.moveTo(cx, barY);
    ctx.lineTo(cx + frac * barHalf, barY);
    ctx.stroke();

    // Text
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '13px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`angle = ${((norm * 180) / Math.PI).toFixed(0)}°   torque = ${rs.torque.toFixed(2)}`, W / 2, 22);
    ctx.fillStyle = '#8a90a0';
    ctx.font = '11px ui-monospace, monospace';
    ctx.fillText('torque (action)', cx, barY + 20);
  }, [env, tick]);

  return <canvas ref={ref} style={{ borderRadius: 8, display: 'block', maxWidth: '100%' }} />;
}
