// CartPole rendering: track + cart (x) + pole (θ). The pole is yellow within ±12° and turns red beyond that (about to terminate).
import { useEffect, useRef } from 'react';
import { useStore } from '../../state/store';
import type { CartPoleRenderState } from '../../envs/continuous/CartPole';

const W = 520;
const H = 300;

export function CartPoleRenderer() {
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

    const rs = env.getRenderState() as CartPoleRenderState;
    const margin = 60;
    const trackY = H - 80;
    const worldHalf = rs.xThreshold * 1.08;
    const sx = (wx: number) => W / 2 + (wx / worldHalf) * (W / 2 - margin);

    // Track
    ctx.strokeStyle = '#3a4150';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx(-rs.xThreshold), trackY);
    ctx.lineTo(sx(rs.xThreshold), trackY);
    ctx.stroke();
    // Boundaries (terminates if exceeded)
    ctx.strokeStyle = 'rgba(224,108,117,0.6)';
    for (const bx of [-rs.xThreshold, rs.xThreshold]) {
      ctx.beginPath();
      ctx.moveTo(sx(bx), trackY - 16);
      ctx.lineTo(sx(bx), trackY + 16);
      ctx.stroke();
    }

    // Cart
    const cartX = sx(rs.x);
    const cartW = 64;
    const cartH = 30;
    ctx.fillStyle = '#3b6ea5';
    ctx.fillRect(cartX - cartW / 2, trackY - cartH, cartW, cartH);
    ctx.fillStyle = '#222';
    for (const wx of [-cartW / 4, cartW / 4]) {
      ctx.beginPath();
      ctx.arc(cartX + wx, trackY, 6, 0, Math.PI * 2);
      ctx.fill();
    }

    // Pole
    const poleLen = 130;
    const px = cartX;
    const py = trackY - cartH;
    const ex = px + poleLen * Math.sin(rs.theta);
    const ey = py - poleLen * Math.cos(rs.theta);
    const safe = Math.abs(rs.theta) < rs.thetaThreshold;
    ctx.strokeStyle = safe ? '#ffd54a' : '#e06c75';
    ctx.lineWidth = 9;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(ex, ey);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#cbd5e1';
    ctx.fill();

    // Text
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '13px ui-monospace, monospace';
    ctx.textAlign = 'center';
    ctx.fillText(
      `θ = ${((rs.theta * 180) / Math.PI).toFixed(1)}°    x = ${rs.x.toFixed(2)} m`,
      W / 2,
      24,
    );
  }, [env, tick]);

  return <canvas ref={ref} style={{ borderRadius: 8, display: 'block', maxWidth: '100%' }} />;
}
