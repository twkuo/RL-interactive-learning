// Run controls: phase-aware step-by-step reveal buttons + auto-play + quick train + predict toggle + reset.
import { useStore } from '../../state/store';
import { Slider } from './Slider';

export function RunControls() {
  const phase = useStore((s) => s.phase);
  const revealed = useStore((s) => s.revealed);
  const revealAction = useStore((s) => s.revealAction);
  const executeStep = useStore((s) => s.executeStep);
  const advance = useStore((s) => s.advance);
  const quickStep = useStore((s) => s.quickStep);
  const train = useStore((s) => s.train);
  const resetAgent = useStore((s) => s.resetAgent);
  const resetEpisode = useStore((s) => s.resetEpisode);
  const isPlaying = useStore((s) => s.isPlaying);
  const play = useStore((s) => s.play);
  const pause = useStore((s) => s.pause);
  const speed = useStore((s) => s.speed);
  const setSpeed = useStore((s) => s.setSpeed);
  const predictMode = useStore((s) => s.predictMode);
  const setPredictMode = useStore((s) => s.setPredictMode);

  let primaryLabel: string;
  let primaryFn: () => void;
  if (phase === 'deciding' && !revealed) {
    primaryLabel = '1. Reveal action';
    primaryFn = revealAction;
  } else if (phase === 'deciding') {
    primaryLabel = '2. Execute & update';
    primaryFn = executeStep;
  } else {
    primaryLabel = '3. Advance to next step';
    primaryFn = advance;
  }

  return (
    <div className="panel">
      <div className="panel-title">Run Controls</div>

      <button className="btn btn-primary" onClick={primaryFn}>
        {primaryLabel}
      </button>
      <button className="btn" onClick={quickStep}>
        Full step (all at once)
      </button>

      <div className="control-row">
        <button className="btn" onClick={() => (isPlaying ? pause() : play())}>
          {isPlaying ? 'Pause' : 'Auto-play'}
        </button>
      </div>
      <Slider
        label="Speed (ms/step)"
        value={speed}
        min={50}
        max={1000}
        step={50}
        decimals={0}
        onChange={setSpeed}
      />

      <label className="checkbox">
        <input
          type="checkbox"
          checked={predictMode}
          onChange={(e) => setPredictMode(e.target.checked)}
        />
        Predict mode (guess then reveal)
      </label>

      <div className="panel-subtitle">Quick train (runs synchronously)</div>
      <div className="control-row">
        <button className="btn" onClick={() => train(50)}>
          Train 50
        </button>
        <button className="btn" onClick={() => train(200)}>
          Train 200
        </button>
        <button className="btn" onClick={() => train(1000)}>
          Train 1000
        </button>
      </div>

      <div className="control-row">
        <button className="btn btn-warn" onClick={resetEpisode}>
          Restart episode
        </button>
        <button className="btn btn-warn" onClick={resetAgent}>
          Reset agent
        </button>
      </div>
    </div>
  );
}
