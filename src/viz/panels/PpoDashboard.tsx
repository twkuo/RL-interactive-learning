// PPO training dashboard: the deep-RL replacement for the per-step "update rule" panel.
// PPO trains on rollouts (clipped surrogate + value loss + entropy) in the Web Worker, so we show
// the training internals: policy/value loss curves, entropy, approx-KL, clip fraction, and V(s).
import { useStore } from '../../state/store';
import { MetricChart } from '../charts/MetricChart';

interface Valued {
  stateValue?(s: number[]): number;
}

export function PpoDashboard() {
  const status = useStore((s) => s.trainingStatus);
  const ep = useStore((s) => s.trainingEpisode);
  const total = useStore((s) => s.trainingTotal);
  const history = useStore((s) => s.metricHistory);
  const latest = useStore((s) => s.metricLatest);
  const bestAvg = useStore((s) => s.bestAvg);
  const err = useStore((s) => s.trainingError);
  const agentLoading = useStore((s) => s.agentLoading);
  const agent = useStore((s) => s.agent);
  const currentState = useStore((s) => s.currentState);
  useStore((s) => s.tick);

  const policyLoss = history.policyLoss ?? [];
  const valueLoss = history.valueLoss ?? [];
  const entropy = latest.entropy;
  const approxKL = latest.approxKL;
  const clipFrac = latest.clipFrac;
  const steps = latest.trainSteps ?? 0;

  let v: number | null = null;
  const va = agent as unknown as Valued;
  if (!agentLoading && Array.isArray(currentState) && va.stateValue) {
    v = va.stateValue(currentState as number[]);
  }

  return (
    <div className="panel">
      <div className="panel-title">PPO Training Dashboard</div>
      {agentLoading && <div className="hint">Loading the neural-network runtime…</div>}
      {err && <div className="verdict bad">Training error: {err}</div>}
      <div className="hint">
        PPO is an actor-critic policy gradient with a clipped objective, trained on rollouts in the
        Web Worker — not a hand-derivable per-step update. Step through the trained policy to see
        π(a|s) sampling; inference uses the BEST policy found, so training longer never hurts it.
      </div>

      <div className="kv-row">
        <span>Status</span>
        <span className="mono">
          {status}
          {status === 'running' && total ? ` · episode ${ep}/${total}` : ''}
        </span>
      </div>
      <div className="kv-row">
        <span>Best avg return (kept for inference)</span>
        <span className="mono">{bestAvg > 0 ? bestAvg.toFixed(1) : '—'}</span>
      </div>
      <div className="kv-row">
        <span>V(current state)</span>
        <span className="mono">{v !== null ? v.toFixed(2) : '—'}</span>
      </div>
      <div className="kv-row">
        <span>Entropy (policy randomness)</span>
        <span className="mono">{entropy !== undefined ? entropy.toFixed(3) : '—'}</span>
      </div>
      <div className="kv-row">
        <span>Approx KL (policy change)</span>
        <span className="mono">{approxKL !== undefined ? approxKL.toFixed(4) : '—'}</span>
      </div>
      <div className="kv-row">
        <span>Clip fraction</span>
        <span className="mono">{clipFrac !== undefined ? clipFrac.toFixed(3) : '—'}</span>
      </div>
      <div className="kv-row">
        <span>Env steps</span>
        <span className="mono">{steps}</span>
      </div>

      <MetricChart
        title="Policy loss (per episode)"
        series={[{ label: 'policyLoss', color: '#5b9bd5', values: policyLoss }]}
        height={110}
      />
      <MetricChart
        title="Value loss (per episode)"
        series={[{ label: 'valueLoss', color: '#e0a458', values: valueLoss }]}
        height={110}
      />
    </div>
  );
}
