// Run controls: phase-aware step-by-step reveal buttons + auto-play + train + predict toggle + reset.
// Deep algorithms train in a Web Worker (with a live dashboard + cancel); tabular train synchronously.
import { useStore } from '../../state/store';
import { getAlgoEntry } from '../../algos/registry';
import { Slider } from './Slider';

export function RunControls() {
  const phase = useStore((s) => s.phase);
  const revealed = useStore((s) => s.revealed);
  const revealAction = useStore((s) => s.revealAction);
  const executeStep = useStore((s) => s.executeStep);
  const advance = useStore((s) => s.advance);
  const quickStep = useStore((s) => s.quickStep);
  const train = useStore((s) => s.train);
  const cancelTraining = useStore((s) => s.cancelTraining);
  const resetAgent = useStore((s) => s.resetAgent);
  const resetEpisode = useStore((s) => s.resetEpisode);
  const isPlaying = useStore((s) => s.isPlaying);
  const play = useStore((s) => s.play);
  const pause = useStore((s) => s.pause);
  const speed = useStore((s) => s.speed);
  const setSpeed = useStore((s) => s.setSpeed);
  const predictMode = useStore((s) => s.predictMode);
  const setPredictMode = useStore((s) => s.setPredictMode);
  const algoId = useStore((s) => s.algoId);
  const trainingStatus = useStore((s) => s.trainingStatus);
  const trainingEpisode = useStore((s) => s.trainingEpisode);
  const trainingTotal = useStore((s) => s.trainingTotal);
  const agentLoading = useStore((s) => s.agentLoading);

  const deep = !!getAlgoEntry(algoId).deep;
  const running = trainingStatus === 'running';
  const busy = running || agentLoading;

  let primaryLabel: string;
  let primaryFn: () => void;
  if (phase === 'deciding' && !revealed) {
    primaryLabel = '1. Reveal action';
    primaryFn = revealAction;
  } else if (phase === 'deciding') {
    // Deep RL does no per-step value update — stepping is pure inference (forward pass).
    primaryLabel = deep ? '2. Execute step (inference)' : '2. Execute & update';
    primaryFn = executeStep;
  } else {
    primaryLabel = '3. Advance to next step';
    primaryFn = advance;
  }

  const trainCounts = deep ? [100, 300, 600] : [50, 200, 1000];

  return (
    <div className="panel">
      <div className="panel-title">Run Controls</div>

      <button className="btn btn-primary" onClick={primaryFn} disabled={busy}>
        {primaryLabel}
      </button>
      <button className="btn" onClick={quickStep} disabled={busy}>
        Full step (all at once)
      </button>
      {deep && (
        <div className="hint">
          Stepping and auto-play just run the trained network (inference) — no learning. The network
          changes only when you Train.
        </div>
      )}

      <div className="control-row">
        <button className="btn" onClick={() => (isPlaying ? pause() : play())} disabled={busy}>
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
          disabled={busy}
        />
        Predict mode (guess then reveal)
      </label>

      <div className="panel-subtitle">
        {deep ? 'Train (Web Worker — UI stays responsive)' : 'Quick train (runs synchronously)'}
      </div>
      {running ? (
        <>
          <div className="hint">
            Training… episode {trainingEpisode}/{trainingTotal}. The dashboard on the right updates live.
          </div>
          <button className="btn btn-warn" onClick={cancelTraining}>
            Stop training
          </button>
        </>
      ) : (
        <div className="control-row">
          {trainCounts.map((n) => (
            <button key={n} className="btn" onClick={() => train(n)} disabled={busy}>
              Train {n}
            </button>
          ))}
        </div>
      )}
      {deep && !running && (
        <div className="hint">Each Train continues the current network; use Reset agent to start fresh.</div>
      )}

      <div className="control-row">
        <button className="btn btn-warn" onClick={resetEpisode} disabled={busy}>
          Restart episode
        </button>
        <button className="btn btn-warn" onClick={resetAgent} disabled={running}>
          Reset agent
        </button>
      </div>
    </div>
  );
}
