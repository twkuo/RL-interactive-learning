// State panel for continuous environments: shows the decoded continuous fields (x, θ…) plus the discretized cell index.
import { useStore } from '../../state/store';
import { getEnvEntry } from '../../envs/registry';

export function ContinuousStatePanel() {
  const env = useStore((s) => s.env);
  const envId = useStore((s) => s.envId);
  const cur = useStore((s) => s.currentState);
  useStore((s) => s.tick); // trigger re-render

  if (getEnvEntry(envId).renderKind === 'grid') return null;
  const fields = env.describeObs(cur);

  return (
    <div className="panel">
      <div className="panel-title">Environment State (continuous)</div>
      {fields.map((f) => (
        <div className="kv-row" key={f.label}>
          <span>{f.label}</span>
          <span className="mono">
            {f.value.toFixed(3)}
            {f.unit ? ` ${f.unit}` : ''}
          </span>
        </div>
      ))}
      <div className="kv-row">
        <span>Discretized cell s</span>
        <span className="mono">
          {cur} / {env.stateCount()}
        </span>
      </div>
      <div className="hint">
        The continuous state is discretized into {env.stateCount()} cells for the tabular agent; the Q-values on the right are Q(s,·) for this cell.
      </div>
    </div>
  );
}
