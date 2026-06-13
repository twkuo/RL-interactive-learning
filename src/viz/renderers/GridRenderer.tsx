// GridWorld renderer: value-based shows V/Q; policy-based (REINFORCE) shows π. Greedy arrow, current/inspected highlight.
import { useEffect, useRef } from 'react';
import { useStore } from '../../state/store';
import { valueColor, normalize } from '../../canvas/colormap';
import { drawArrow, strokeRect, outlinedText } from '../../canvas/draw';
import { argmax } from '../../core/utils';

const CELL = 96;

export function GridRenderer() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const env = useStore((s) => s.env);
  const agent = useStore((s) => s.agent);
  const currentState = useStore((s) => s.currentState);
  const inspectedState = useStore((s) => s.inspectedState);
  const setInspectedState = useStore((s) => s.setInspectedState);
  const tick = useStore((s) => s.tick);

  // Mouse moves over a cell → set the inspected state
  useEffect(() => {
    const canvas = canvasRef.current;
    const grid = env.grid;
    if (!canvas || !grid) return;
    const move = (ev: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const col = Math.floor((ev.clientX - rect.left) / CELL);
      const row = Math.floor((ev.clientY - rect.top) / CELL);
      if (col < 0 || col >= grid.cols || row < 0 || row >= grid.rows) {
        setInspectedState(null);
        return;
      }
      setInspectedState(row * grid.cols + col);
    };
    const leave = () => setInspectedState(null);
    canvas.addEventListener('mousemove', move);
    canvas.addEventListener('mouseleave', leave);
    return () => {
      canvas.removeEventListener('mousemove', move);
      canvas.removeEventListener('mouseleave', leave);
    };
  }, [env, setInspectedState]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const grid = env.grid;
    if (!canvas || !grid) return;
    const { rows, cols } = grid;
    const W = cols * CELL;
    const H = rows * CELL;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const nStates = env.stateCount();
    const nActions = env.actionSpace.n;
    const Q = agent.usesQ && agent.getQ ? agent.getQ(nStates, nActions) : null;
    const policy = !Q && agent.getPolicy ? agent.getPolicy(nStates, nActions) : null;
    const V = agent.getV ? agent.getV(nStates) : null;

    // V color range
    let vLo = Infinity;
    let vHi = -Infinity;
    if (V) {
      for (let i = 0; i < V.length; i++) {
        if (V[i] < vLo) vLo = V[i];
        if (V[i] > vHi) vHi = V[i];
      }
    }

    const cellColor = (s: number): string => {
      if (V) return valueColor(normalize(V[s], vLo, vHi));
      if (policy) {
        let mp = 0;
        for (let a = 0; a < nActions; a++) mp = Math.max(mp, policy[s * nActions + a]);
        return valueColor(normalize(mp, 1 / nActions, 1)); // policy confidence
      }
      return '#262e42';
    };

    for (let s = 0; s < nStates; s++) {
      const r = Math.floor(s / cols);
      const c = s % cols;
      const x = c * CELL;
      const y = r * CELL;
      const cx = x + CELL / 2;
      const cy = y + CELL / 2;
      const isWall = grid.walls.includes(s);
      const isGoal = grid.goals.includes(s);
      const isHole = grid.holes.includes(s);

      ctx.fillStyle = isWall
        ? '#3a3f4b'
        : isGoal
          ? '#1f6f43'
          : isHole
            ? '#5b1f25'
            : cellColor(s);
      ctx.fillRect(x, y, CELL, CELL);
      ctx.strokeStyle = 'rgba(255,255,255,0.10)';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, CELL, CELL);

      if (isWall) continue;

      const pos: Array<[number, number, CanvasTextAlign]> = [
        [x + 11, cy, 'left'], // 0 left
        [cx, y + CELL - 12, 'center'], // 1 down
        [x + CELL - 11, cy, 'right'], // 2 right
        [cx, y + 13, 'center'], // 3 up
      ];

      if (isGoal) {
        outlinedText(ctx, 'Goal', cx, cy, { size: 14, color: '#9dffc4' });
      } else if (isHole) {
        outlinedText(ctx, 'Trap', cx, cy, { size: 14, color: '#ff9aa2' });
      } else if (Q) {
        const qs: number[] = [];
        for (let a = 0; a < nActions; a++) qs.push(Q[s * nActions + a]);
        const greedy = argmax(qs);
        const [dc, dr] = grid.actionDeltas[greedy];
        drawArrow(ctx, cx - dc * 13, cy - dr * 13, cx + dc * 16, cy + dr * 16, 'rgba(255,255,255,0.42)', 2);
        for (let a = 0; a < 4 && a < nActions; a++) {
          const isG = a === greedy;
          outlinedText(ctx, qs[a].toFixed(2), pos[a][0], pos[a][1], {
            size: 11,
            align: pos[a][2],
            color: isG ? '#ffe27a' : '#eef2f7',
            bold: isG,
          });
        }
        if (V) {
          outlinedText(ctx, `V ${V[s].toFixed(2)}`, x + 5, y + 9, {
            size: 10,
            align: 'left',
            color: 'rgba(190,225,255,0.95)',
          });
        }
      } else if (policy) {
        const ps: number[] = [];
        for (let a = 0; a < nActions; a++) ps.push(policy[s * nActions + a]);
        const greedy = argmax(ps);
        const [dc, dr] = grid.actionDeltas[greedy];
        drawArrow(ctx, cx - dc * 13, cy - dr * 13, cx + dc * 16, cy + dr * 16, 'rgba(255,255,255,0.42)', 2);
        for (let a = 0; a < 4 && a < nActions; a++) {
          const isG = a === greedy;
          outlinedText(ctx, ps[a].toFixed(2), pos[a][0], pos[a][1], {
            size: 11,
            align: pos[a][2],
            color: isG ? '#ffe27a' : '#eef2f7',
            bold: isG,
          });
        }
        outlinedText(ctx, `π* ${ps[greedy].toFixed(2)}`, x + 5, y + 9, {
          size: 10,
          align: 'left',
          color: 'rgba(190,225,255,0.95)',
        });
      } else if (V) {
        outlinedText(ctx, V[s].toFixed(2), cx, cy, { size: 14, color: '#ffffff', bold: true });
      }

      if (s === grid.start && !isGoal && !isHole) {
        outlinedText(ctx, 'S', x + 6, y + CELL - 11, {
          size: 10,
          align: 'left',
          color: 'rgba(255,255,255,0.6)',
        });
      }
    }

    // Current state: yellow border + agent dot
    {
      const c = currentState % cols;
      const r = Math.floor(currentState / cols);
      const x = c * CELL;
      const y = r * CELL;
      strokeRect(ctx, x, y, CELL, CELL, '#ffd54a', 4);
      ctx.beginPath();
      ctx.arc(x + CELL / 2, y + CELL / 2, 7, 0, Math.PI * 2);
      ctx.fillStyle = '#ffd54a';
      ctx.fill();
      ctx.strokeStyle = '#7a5c00';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Inspected (hovered) state: cyan border
    if (inspectedState !== null && inspectedState !== currentState) {
      const c = inspectedState % cols;
      const r = Math.floor(inspectedState / cols);
      strokeRect(ctx, c * CELL, r * CELL, CELL, CELL, '#5fd0ff', 3);
    }
  }, [env, agent, currentState, inspectedState, tick]);

  return <canvas ref={canvasRef} style={{ borderRadius: 8, display: 'block' }} />;
}
