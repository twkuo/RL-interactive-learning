// MountainCar rendering: hill y=sin(3x)·0.45+0.55 + car (position) + flag on the right (goal).
import { useEffect, useRef } from 'react';
import { useStore } from '../../state/store';
import type { MountainCarRenderState } from '../../envs/continuous/MountainCar';

const W = 520;
const H = 300;

const height = (x: number) => Math.sin(3 * x) * 0.45 + 0.55;

export function MountainCarRenderer() {
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

    const rs = env.getRenderState() as MountainCarRenderState;
    const margin = 30;
    const sx = (x: number) =>
      margin + ((x - rs.minPos) / (rs.maxPos - rs.minPos)) * (W - 2 * margin);
    const sy = (h: number) => H - 30 - h * (H - 90);

    // Hill
    ctx.strokeStyle = '#5fc587';
    ctx.lineWidth = 3;
    ctx.beginPath();
    for (let i = 0; i <= 100; i++) {
      const x = rs.minPos + ((rs.maxPos - rs.minPos) * i) / 100;
      const px = sx(x);
      const py = sy(height(x));
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // Flag (goal)
    const gx = sx(rs.goal);
    const gy = sy(height(rs.goal));
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(gx, gy);
    ctx.lineTo(gx, gy - 40);
    ctx.stroke();
    ctx.fillStyle = '#7dffb0';
    ctx.beginPath();
    ctx.moveTo(gx, gy - 40);
    ctx.lineTo(gx + 22, gy - 32);
    ctx.lineTo(gx, gy - 24);
    ctx.closePath();
    ctx.fill();

    // Car
    const cx = sx(rs.position);
    const cy = sy(height(rs.position));
    ctx.fillStyle = '#ffd54a';
    ctx.beginPath();
    ctx.arc(cx, cy - 8, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#7a5c00';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Text
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '13px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`x = ${rs.position.toFixed(3)}    v = ${rs.velocity.toFixed(4)}`, W / 2, 22);
  }, [env, tick]);

  return <canvas ref={ref} style={{ borderRadius: 8, display: 'block', maxWidth: '100%' }} />;
}
