// Acrobot rendering: two-link pendulum hanging from a fixed pivot. A dashed line marks the goal
// height (tip above it = success); the tip turns green when it reaches the goal.
import { useEffect, useRef } from 'react';
import { useStore } from '../../state/store';
import type { AcrobotRenderState } from '../../envs/continuous/Acrobot';

const W = 520;
const H = 300;
const SCALE = 64; // px per unit length
const PIVOT_Y = 150;

export function AcrobotRenderer() {
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

    const rs = env.getRenderState() as AcrobotRenderState;
    const cx = W / 2;
    // world (y up) -> canvas (y down)
    const X = (wx: number) => cx + wx * SCALE;
    const Y = (wy: number) => PIVOT_Y - wy * SCALE;

    // Goal line at tip height = 1 (above the pivot).
    ctx.strokeStyle = 'rgba(46,158,107,0.6)';
    ctx.setLineDash([6, 5]);
    ctx.beginPath();
    ctx.moveTo(0, Y(1));
    ctx.lineTo(W, Y(1));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#7dffb0';
    ctx.font = '11px ui-monospace, monospace';
    ctx.textAlign = 'left';
    ctx.fillText('goal: tip above this line', 8, Y(1) - 6);

    const th1 = rs.theta1;
    const th2 = rs.theta2;
    // Joint positions (y up): link hangs down at θ=0.
    const p1x = rs.l1 * Math.sin(th1);
    const p1y = -rs.l1 * Math.cos(th1);
    const p2x = p1x + rs.l2 * Math.sin(th1 + th2);
    const p2y = p1y - rs.l2 * Math.cos(th1 + th2);
    const tipHeight = -Math.cos(th1) - Math.cos(th1 + th2);
    const reached = tipHeight > 1;

    // Links
    ctx.lineCap = 'round';
    ctx.lineWidth = 7;
    ctx.strokeStyle = '#3b6ea5';
    ctx.beginPath();
    ctx.moveTo(X(0), Y(0));
    ctx.lineTo(X(p1x), Y(p1y));
    ctx.stroke();
    ctx.strokeStyle = '#5b9bd5';
    ctx.beginPath();
    ctx.moveTo(X(p1x), Y(p1y));
    ctx.lineTo(X(p2x), Y(p2y));
    ctx.stroke();

    // Joints + tip
    ctx.fillStyle = '#cbd5e1';
    ctx.beginPath();
    ctx.arc(X(0), Y(0), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(X(p1x), Y(p1y), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = reached ? '#7dffb0' : '#ffd54a';
    ctx.beginPath();
    ctx.arc(X(p2x), Y(p2y), 7, 0, Math.PI * 2);
    ctx.fill();

    // Text
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '13px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(
      `θ₁ = ${((th1 * 180) / Math.PI).toFixed(0)}°   θ₂ = ${((th2 * 180) / Math.PI).toFixed(0)}°   tip = ${tipHeight.toFixed(2)}`,
      W / 2,
      24,
    );
  }, [env, tick]);

  return <canvas ref={ref} style={{ borderRadius: 8, display: 'block', maxWidth: '100%' }} />;
}
