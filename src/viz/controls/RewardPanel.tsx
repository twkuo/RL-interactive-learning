// Reward design panel: shows the environment's reward description and lets the user edit reward parameters in real time (reward design).
import { useStore } from '../../state/store';
import { Slider } from './Slider';

export function RewardPanel() {
  const env = useStore((s) => s.env);
  const setRewardParam = useStore((s) => s.setRewardParam);
  useStore((s) => s.tick); // trigger re-render

  const params = env.rewardParams?.() ?? [];
  if (params.length === 0) return null;
  const desc = env.rewardDescription?.();

  return (
    <div className="panel">
      <div className="panel-title">Reward Design</div>
      {desc && <div className="hint">{desc}</div>}
      {params.map((p) => (
        <Slider
          key={p.key}
          label={p.label}
          value={p.value}
          min={p.min}
          max={p.max}
          step={p.step}
          decimals={p.decimals ?? 2}
          onChange={(v) => setRewardParam(p.key, v)}
        />
      ))}
      <div className="hint">After changing rewards, press "Reset agent" to retrain, or "Run & add to comparison" to use the new rewards — reward design directly changes the learned policy!</div>
      {params
        .filter((p) => p.hint)
        .map((p) => (
          <div className="hint" key={p.key}>
            · <b>{p.label}</b>: {p.hint}
          </div>
        ))}
    </div>
  );
}
