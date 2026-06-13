// Experiment comparison controls: run the current config once and add it to the comparison, manage comparison curves (also serves as the legend).
import { useStore } from '../../state/store';
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
  const removeComparisonRun = useStore((s) => s.removeComparisonRun);
  const clearComparisonRuns = useStore((s) => s.clearComparisonRuns);
  const episodeReturns = useStore((s) => s.episodeReturns);

  const hasLegend = comparisonRuns.length > 0 || episodeReturns.length > 0;

  return (
    <div className="panel">
      <div className="panel-title">Compare Configurations (algorithm / hyperparameters)</div>
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
      <div className="hint">
        Runs a brand-new agent for one round using the current algorithm + hyperparameters and adds its learning curve; press again after changing settings to overlay another run (does not affect the interactive board above).
      </div>

      {hasLegend && (
        <div className="legend-list">
          {episodeReturns.length > 0 && (
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
