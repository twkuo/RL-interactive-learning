// Minimal SVG histogram — used for the DQN TD-error distribution.
interface Props {
  title: string;
  values: number[];
  bins?: number;
  color?: string;
}

export function Histogram({ title, values, bins = 20, color = '#cf6f8f' }: Props) {
  if (values.length === 0) {
    return (
      <>
        <div className="panel-subtitle">{title}</div>
        <div className="hint">No data yet.</div>
      </>
    );
  }
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (hi <= lo) hi = lo + 1;
  const counts = new Array<number>(bins).fill(0);
  for (const v of values) {
    let b = Math.floor(((v - lo) / (hi - lo)) * bins);
    if (b >= bins) b = bins - 1;
    if (b < 0) b = 0;
    counts[b] += 1;
  }
  const maxC = Math.max(...counts);
  const W = 100;
  const H = 36;
  const bw = W / bins;
  return (
    <>
      <div className="panel-subtitle">{title}</div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={70} preserveAspectRatio="none">
        {counts.map((c, i) => {
          const h = maxC ? (c / maxC) * H : 0;
          return <rect key={i} x={i * bw} y={H - h} width={bw * 0.88} height={h} fill={color} />;
        })}
      </svg>
      <div className="slider-head">
        <span className="mono">{lo.toFixed(3)}</span>
        <span className="mono">{hi.toFixed(3)}</span>
      </div>
    </>
  );
}
