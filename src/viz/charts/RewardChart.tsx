// Learning curve comparison: overlay the moving averages of multiple comparison runs plus the current interactive run on one chart.
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useStore } from '../../state/store';

// Color for the "current interactive agent" curve (also used by the ComparisonControls legend).
export const CURRENT_RUN_COLOR = '#cbd5e1';

function movingAverage(data: number[], window: number): number[] {
  const out: number[] = [];
  let sum = 0;
  for (let i = 0; i < data.length; i++) {
    sum += data[i];
    if (i >= window) sum -= data[i - window];
    out.push(sum / Math.min(i + 1, window));
  }
  return out;
}

interface ChartRun {
  id: string;
  label: string;
  color: string;
  returns: number[];
}

export function RewardChart() {
  const comparisonRuns = useStore((s) => s.comparisonRuns);
  const episodeReturns = useStore((s) => s.episodeReturns);

  const runs: ChartRun[] = [...comparisonRuns];
  if (episodeReturns.length > 0) {
    runs.push({
      id: 'current',
      label: 'Current interactive agent',
      color: CURRENT_RUN_COLOR,
      returns: episodeReturns.map((r) => r.return),
    });
  }

  if (runs.length === 0) {
    return (
      <div className="panel chart-panel">
        <div className="panel-title">Learning Curve</div>
        <div className="hint">
          Use 'Run current config & add to comparison' above, or 'Quick train' on the left, to show the learning curve here.
        </div>
      </div>
    );
  }

  const maxLen = Math.max(...runs.map((r) => r.returns.length));
  const maPerRun = runs.map((r) =>
    movingAverage(r.returns, Math.max(5, Math.floor(r.returns.length / 20))),
  );
  const stride = Math.max(1, Math.floor(maxLen / 300));
  const data: Record<string, number>[] = [];
  for (let e = 0; e < maxLen; e += stride) {
    const point: Record<string, number> = { episode: e + 1 };
    runs.forEach((r, ri) => {
      if (e < r.returns.length) point[r.id] = maPerRun[ri][e];
    });
    data.push(point);
  }

  return (
    <div className="panel chart-panel">
      <div className="panel-title">Learning Curve Comparison (moving-average return per episode, {runs.length} lines)</div>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2e38" />
          <XAxis dataKey="episode" stroke="#8a90a0" fontSize={11} />
          <YAxis stroke="#8a90a0" fontSize={11} />
          <Tooltip
            contentStyle={{ background: '#1b1e26', border: '1px solid #333', borderRadius: 6 }}
            labelStyle={{ color: '#ccc' }}
          />
          {runs.map((r) => (
            <Line
              key={r.id}
              type="monotone"
              dataKey={r.id}
              name={r.label}
              stroke={r.color}
              strokeWidth={r.id === 'current' ? 1.5 : 2.5}
              strokeDasharray={r.id === 'current' ? '4 3' : undefined}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
