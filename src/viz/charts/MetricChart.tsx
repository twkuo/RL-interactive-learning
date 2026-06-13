// Generic time-series line chart for training metrics (loss, return, …). Reuses the Recharts
// pattern from RewardChart; downsamples to ~300 points.
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export interface MetricSeries {
  label: string;
  color: string;
  values: number[];
}

interface Props {
  title: string;
  series: MetricSeries[];
  height?: number;
}

export function MetricChart({ title, series, height = 140 }: Props) {
  const maxLen = Math.max(0, ...series.map((s) => s.values.length));
  if (maxLen === 0) {
    return (
      <>
        <div className="panel-subtitle">{title}</div>
        <div className="hint">No data yet — start training to populate.</div>
      </>
    );
  }
  const stride = Math.max(1, Math.floor(maxLen / 300));
  const data: Record<string, number>[] = [];
  for (let i = 0; i < maxLen; i += stride) {
    const point: Record<string, number> = { x: i + 1 };
    series.forEach((s) => {
      if (i < s.values.length) point[s.label] = s.values[i];
    });
    data.push(point);
  }
  return (
    <>
      <div className="panel-subtitle">{title}</div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 6, right: 12, bottom: 0, left: -12 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2e38" />
          <XAxis dataKey="x" stroke="#8a90a0" fontSize={10} />
          <YAxis stroke="#8a90a0" fontSize={10} width={44} />
          <Tooltip
            contentStyle={{ background: '#1b1e26', border: '1px solid #333', borderRadius: 6 }}
            labelStyle={{ color: '#ccc' }}
          />
          {series.map((s) => (
            <Line
              key={s.label}
              type="monotone"
              dataKey={s.label}
              stroke={s.color}
              strokeWidth={2}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </>
  );
}
