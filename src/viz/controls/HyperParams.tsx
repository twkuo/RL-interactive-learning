import { useStore } from '../../state/store';
import { getAlgoEntry } from '../../algos/registry';
import { Slider } from './Slider';

export function HyperParams() {
  const hp = useStore((s) => s.hyperparams);
  const set = useStore((s) => s.setHyperparam);
  const algoId = useStore((s) => s.algoId);
  const maxSteps = useStore((s) => s.maxSteps);
  const setMaxSteps = useStore((s) => s.setMaxSteps);
  const usesEps = getAlgoEntry(algoId).usesEpsilon;

  return (
    <div className="panel">
      <div className="panel-title">Hyperparameters</div>
      <Slider label="Learning rate α" value={hp.alpha} min={0} max={1} step={0.01} onChange={(v) => set('alpha', v)} />
      <Slider
        label="Discount γ"
        value={hp.gamma}
        min={0}
        max={0.999}
        step={0.005}
        decimals={3}
        onChange={(v) => set('gamma', v)}
      />
      {usesEps && (
        <>
          <Slider
            label="Exploration rate ε"
            value={hp.epsilon}
            min={0}
            max={1}
            step={0.01}
            onChange={(v) => set('epsilon', v)}
          />
          <Slider
            label="ε decay / episode"
            value={hp.epsilonDecay}
            min={0.9}
            max={1}
            step={0.005}
            decimals={3}
            onChange={(v) => set('epsilonDecay', v)}
          />
        </>
      )}
      <div className="panel-subtitle">Episode settings</div>
      <Slider
        label="Max steps/episode (maxSteps)"
        value={maxSteps}
        min={10}
        max={1000}
        step={10}
        decimals={0}
        onChange={setMaxSteps}
      />
    </div>
  );
}
