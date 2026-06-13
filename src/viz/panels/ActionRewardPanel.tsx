// Action / Reward / counters panel.
import { useStore } from '../../state/store';

export function ActionRewardPanel() {
  const env = useStore((s) => s.env);
  const explanation = useStore((s) => s.explanation);
  const revealed = useStore((s) => s.revealed);
  const lastStep = useStore((s) => s.lastStep);
  const cumulativeReward = useStore((s) => s.cumulativeReward);
  const episode = useStore((s) => s.episode);
  const stepInEpisode = useStore((s) => s.stepInEpisode);
  const totalSteps = useStore((s) => s.totalSteps);

  const meanings = env.actionMeanings();
  const actionLabel =
    revealed && explanation ? meanings[explanation.chosenAction] : '(not revealed yet)';

  return (
    <div className="panel">
      <div className="panel-title">Action / Reward</div>
      <div className="kv-row">
        <span>Current action a</span>
        <b>{actionLabel}</b>
      </div>
      <div className="kv-row">
        <span>Last reward r</span>
        <span className="mono">{lastStep ? lastStep.reward.toFixed(3) : '—'}</span>
      </div>
      <div className="kv-row">
        <span>Cumulative reward (episode)</span>
        <span className="mono">{cumulativeReward.toFixed(3)}</span>
      </div>
      <div className="kv-row">
        <span>Episode</span>
        <span className="mono">{episode}</span>
      </div>
      <div className="kv-row">
        <span>Steps this episode</span>
        <span className="mono">{stepInEpisode}</span>
      </div>
      <div className="kv-row">
        <span>Total steps</span>
        <span className="mono">{totalSteps}</span>
      </div>
      {lastStep && (lastStep.terminated || lastStep.truncated) && (
        <div className="status-row">
          {lastStep.terminated && <span className="chip chip-term">terminated (natural)</span>}
          {lastStep.truncated && <span className="chip chip-trunc">truncated (step limit)</span>}
        </div>
      )}
    </div>
  );
}
