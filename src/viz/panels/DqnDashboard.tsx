// DQN training dashboard: the deep-RL replacement for the per-step "update rule" panel.
// A neural-net gradient step isn't hand-derivable, so we surface the training internals instead —
// replay buffer fill, target-network sync, ε schedule, loss curve, and the TD-error distribution.
import { useStore } from '../../state/store';
import { MetricChart } from '../charts/MetricChart';
import { Histogram } from '../charts/Histogram';

export function DqnDashboard() {
  const status = useStore((s) => s.trainingStatus);
  const ep = useStore((s) => s.trainingEpisode);
  const total = useStore((s) => s.trainingTotal);
  const loss = useStore((s) => s.lossHistory);
  const td = useStore((s) => s.tdErrorHistory);
  const fill = useStore((s) => s.bufferFill);
  const eps = useStore((s) => s.trainEpsilon);
  const steps = useStore((s) => s.trainStepsCount);
  const targetSync = useStore((s) => s.targetSyncEvery);
  const err = useStore((s) => s.trainingError);
  const bestAvg = useStore((s) => s.bestAvg);
  const agentLoading = useStore((s) => s.agentLoading);
  useStore((s) => s.tick);

  const lastLoss = loss.length ? loss[loss.length - 1] : null;
  const lastTd = td.length ? td[td.length - 1] : null;
  const stepsToSync = targetSync ? targetSync - (steps % targetSync) : 0;
  const pct = Math.round(fill * 100);

  return (
    <div className="panel">
      <div className="panel-title">DQN Training Dashboard</div>
      {agentLoading && <div className="hint">Loading the neural-network runtime…</div>}
      {err && <div className="verdict bad">Training error: {err}</div>}
      <div className="hint">
        A DQN update is a minibatch gradient step on a neural network — not hand-derivable like
        tabular Q-learning. Watch the training internals here, then step through the trained policy
        to see Q(s,·) from a forward pass. Inference uses the BEST policy found during training
        (greedy), so training longer never makes inference worse.
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

      <div className="slider-head" style={{ marginTop: 8 }}>
        <span>Replay buffer</span>
        <span className="mono">{pct}%</span>
      </div>
      <div className="bar-track" style={{ height: 10 }}>
        <div className="bar-fill" style={{ width: `${pct}%`, background: 'var(--blue)' }} />
      </div>

      <div className="kv-row">
        <span>ε (exploration)</span>
        <span className="mono">{eps.toFixed(3)}</span>
      </div>
      <div className="kv-row">
        <span>Gradient steps</span>
        <span className="mono">{steps}</span>
      </div>
      <div className="kv-row">
        <span>Target network sync</span>
        <span className="mono">{targetSync ? `every ${targetSync} · next in ${stepsToSync}` : '—'}</span>
      </div>
      <div className="kv-row">
        <span>Loss (latest)</span>
        <span className="mono">{lastLoss !== null ? lastLoss.toFixed(4) : '—'}</span>
      </div>
      <div className="kv-row">
        <span>|TD error| (latest)</span>
        <span className="mono">{lastTd !== null ? lastTd.toFixed(4) : '—'}</span>
      </div>

      <MetricChart title="Loss per episode" series={[{ label: 'loss', color: '#e0a458', values: loss }]} height={120} />
      <Histogram title="TD-error distribution (recent episodes)" values={td.slice(-200)} bins={20} color="#cf6f8f" />
    </div>
  );
}
