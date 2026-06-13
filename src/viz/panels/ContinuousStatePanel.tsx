// State panel for continuous environments. Tabular variants also show the discretized cell index;
// vector (deep-RL) variants feed the raw vector straight into the network.
import { useStore } from '../../state/store';
import { getEnvEntry } from '../../envs/registry';
import type { TabularEnvironment } from '../../core/types';

export function ContinuousStatePanel() {
  const env = useStore((s) => s.env);
  const envId = useStore((s) => s.envId);
  const cur = useStore((s) => s.currentState);
  useStore((s) => s.tick); // trigger re-render

  const entry = getEnvEntry(envId);
  if (entry.renderKind === 'grid') return null;
  const fields = env.describeObs(cur);
  const isBox = entry.obsKind === 'box';
  const tab = !isBox ? (env as TabularEnvironment) : null;

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
      {tab ? (
        <>
          <div className="kv-row">
            <span>Discretized cell s</span>
            <span className="mono">
              {cur as number} / {tab.stateCount()}
            </span>
          </div>
          <div className="hint">
            The continuous state is discretized into {tab.stateCount()} cells for the tabular agent; the Q-values on the right are Q(s,·) for this cell.
          </div>
        </>
      ) : (
        <div className="hint">
          The deep agent reads this raw {fields.length}-dimensional vector directly — no discretization.
          Q(s,·) on the right comes from a neural-network forward pass.
        </div>
      )}
    </div>
  );
}
