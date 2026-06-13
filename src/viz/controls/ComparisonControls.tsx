// Experiment comparison controls: build overlaid learning curves and manage them (also the legend).
//  - Tabular: "Run current config & add" runs a fresh controlled agent (synchronous).
//  - Any algo (incl. DQN): "Pin current learning curve" persists the run you already trained
//    (no re-run) so you can overlay e.g. DQN vs Q-Learning on the same CartPole task.
import { useStore } from '../../state/store';
import { getAlgoEntry } from '../../algos/registry';
import { CURRENT_RUN_COLOR } from '../charts/RewardChart';

function avgLast(arr: number[], n: number): number {
  const last = arr.slice(-n);
  if (last.length === 0) return 0;
  return last.reduce((a, b) => a + b, 0) / last.length;
}

export function ComparisonControls() {
  const comparisonRuns = useStore((s) => s.comparisonRuns);
  const compareEpisodes = useStore((s) => s.compareEpisodes);
  const setCompareEpisodes = useStore((s) => s.setCompareEpisodes);
  const runComparison = useStore((s) => s.runComparison);
  const snapshotCurrentRun = useStore((s) => s.snapshotCurrentRun);
  const removeComparisonRun = useStore((s) => s.removeComparisonRun);
  const clearComparisonRuns = useStore((s) => s.clearComparisonRuns);
  const episodeReturns = useStore((s) => s.episodeReturns);
  const algoId = useStore((s) => s.algoId);

  const deep = !!getAlgoEntry(algoId).deep;
  const hasCurrent = episodeReturns.length > 0;
  const hasLegend = comparisonRuns.length > 0 || hasCurrent;

  return (
    <div className="panel">
      <div className="panel-title">Compare Configurations (algorithm / hyperparameters)</div>

      {!deep && (
        <div className="compare-row">
          <label className="compare-eps">
            Episodes per run
            <input
              type="number"
              min={10}
              max={5000}
              step={50}
              value={compareEpisodes}
              onChange={(e) => setCompareEpisodes(Math.max(10, parseInt(e.target.value, 10) || 10))}
            />
          </label>
          <button className="btn btn-primary compare-run" onClick={runComparison}>
            Run current config & add to comparison
          </button>
        </div>
      )}

      <button className="btn" onClick={snapshotCurrentRun} disabled={!hasCurrent}>
        Pin current learning curve to comparison
      </button>

      <div className="hint">
        {deep
          ? 'Train DQN on the left, then pin its curve here. Switching between DQN and a tabular method on CartPole keeps these curves — so you can overlay "DQN vs Q-Learning" on the same task.'
          : 'Run a fresh controlled agent and overlay it, or pin the current interactive curve. Switching algorithms keeps the curves as long as the environment stays the same task.'}
      </div>

      {hasLegend && (
        <div className="legend-list">
          {hasCurrent && (
            <div className="legend-item">
              <span className="swatch" style={{ background: CURRENT_RUN_COLOR }} />
              <span className="legend-label">Current interactive agent</span>
              <span className="mono legend-avg">
                last-50 avg {avgLast(episodeReturns.map((r) => r.return), 50).toFixed(2)}
              </span>
            </div>
          )}
          {comparisonRuns.map((r) => (
            <div className="legend-item" key={r.id}>
              <span className="swatch" style={{ background: r.color }} />
              <span className="legend-label" title={r.label}>
                {r.label}
              </span>
              <span className="mono legend-avg">last-50 avg {avgLast(r.returns, 50).toFixed(2)}</span>
              <button
                className="legend-remove"
                onClick={() => removeComparisonRun(r.id)}
                title="remove"
              >
                Remove
              </button>
            </div>
          ))}
          {comparisonRuns.length > 0 && (
            <button className="btn legend-clear" onClick={clearComparisonRuns}>
              Clear all comparison curves
            </button>
          )}
        </div>
      )}
    </div>
  );
}
