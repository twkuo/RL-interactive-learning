// Multi-armed bandit renderer: one bar per arm showing the agent's value estimate Q(s0, a)
// (or π(a) for policy methods like REINFORCE), with the hidden true mean as a dashed reference
// tick, per-arm pull counts, and highlights for the greedy arm and the arm just pulled.
import { useEffect, useRef } from 'react';
import { useStore } from '../../state/store';
import { outlinedText, strokeRect } from '../../canvas/draw';
import { argmax } from '../../core/utils';
import type { Bandit, BanditRenderState } from '../../envs/discrete/Bandit';

const PAD = 30;
const COLW = 86;
const H = 340;

export function BanditRenderer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // BanditRenderer only mounts for the bandit env, so the cast is safe.
  const env = useStore((s) => s.env) as unknown as Bandit;
  const agent = useStore((s) => s.agent);
  const tick = useStore((s) => s.tick);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rs = env.getRenderState() as BanditRenderState;
    const k = rs.arms;
    const W = PAD * 2 + k * COLW;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const Q = agent.usesQ && agent.getQ ? agent.getQ(1, k) : null;
    const policy = !Q && agent.getPolicy ? agent.getPolicy(1, k) : null;
    const isProb = !Q && !!policy;

    const vals: number[] = [];
    for (let a = 0; a < k; a++) vals.push(Q ? Q[a] : policy ? policy[a] : 0);
    const greedy = argmax(vals);

    // Value range for the y-axis (probabilities are fixed to [0,1]).
    let lo = 0;
    let hi = isProb ? 1 : 0;
    if (!isProb) {
      for (const v of vals) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
      for (const v of rs.trueMeans) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
      const pad = Math.max(0.5, (hi - lo) * 0.15);
      lo -= pad;
      hi += pad;
      if (hi - lo < 1e-6) {
        lo -= 1;
        hi += 1;
      }
    }
    const top = 26;
    const bottom = H - 62;
    const yOf = (v: number) => bottom - ((v - lo) / (hi - lo)) * (bottom - top);
    const yBase = yOf(0);

    // Zero baseline
    ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD - 8, yBase);
    ctx.lineTo(W - PAD + 8, yBase);
    ctx.stroke();
    outlinedText(ctx, '0', PAD - 14, yBase, { size: 10, align: 'right', color: 'rgba(255,255,255,0.45)' });

    for (let a = 0; a < k; a++) {
      const x = PAD + a * COLW;
      const bw = COLW - 26;
      const cx = x + 4 + bw / 2;
      const y = yOf(vals[a]);
      const isG = a === greedy;
      const isLast = a === rs.lastArm;

      const barTop = Math.min(y, yBase);
      const barH = Math.max(1, Math.abs(y - yBase));
      ctx.fillStyle = isG ? '#3b7dd8' : '#33405c';
      ctx.fillRect(x + 4, barTop, bw, barH);
      if (isLast) strokeRect(ctx, x + 4, barTop, bw, barH, '#ffd54a', 3);

      outlinedText(ctx, vals[a].toFixed(2), cx, vals[a] >= 0 ? barTop - 8 : barTop + barH + 12, {
        size: 11,
        color: isG ? '#ffe27a' : '#eef2f7',
        bold: isG,
      });

      // Hidden true mean (value mode only)
      if (!isProb) {
        const ty = yOf(rs.trueMeans[a]);
        ctx.strokeStyle = 'rgba(157,255,196,0.7)';
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x + 2, ty);
        ctx.lineTo(x + bw + 6, ty);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      outlinedText(ctx, `Arm ${a + 1}`, cx, H - 38, { size: 11, color: '#cdd6e6', bold: isG });
      outlinedText(ctx, `${rs.pulls[a]} pulls`, cx, H - 22, { size: 10, color: 'rgba(205,214,230,0.7)' });
    }

    if (rs.lastArm >= 0) {
      outlinedText(ctx, `Last: Arm ${rs.lastArm + 1} → reward ${rs.lastReward.toFixed(2)}`, PAD, 13, {
        size: 11,
        align: 'left',
        color: '#ffd54a',
      });
    }
    outlinedText(ctx, isProb ? 'bars = π(a)' : '– – true mean (hidden)', W - PAD, 13, {
      size: 10,
      align: 'right',
      color: isProb ? 'rgba(205,214,230,0.7)' : 'rgba(157,255,196,0.85)',
    });
  }, [env, agent, tick]);

  return <canvas ref={canvasRef} style={{ borderRadius: 8, display: 'block' }} />;
}
