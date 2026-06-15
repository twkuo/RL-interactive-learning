// Product core: fully unpack how an action gets selected, and support "predict first, then reveal".
import { useStore } from '../../state/store';
import { normalize } from '../../canvas/colormap';

export function DecisionPanel() {
  const explanation = useStore((s) => s.explanation);
  const revealed = useStore((s) => s.revealed);
  const phase = useStore((s) => s.phase);
  const predictMode = useStore((s) => s.predictMode);
  const predictedAction = useStore((s) => s.predictedAction);
  const predictionCorrect = useStore((s) => s.predictionCorrect);
  const predict = useStore((s) => s.predict);
  const revealAction = useStore((s) => s.revealAction);

  if (!explanation) return null;

  // Continuous (Gaussian) policy: no discrete actions to bar-chart — show μ, σ, and the sample.
  if (explanation.policyKind === 'gaussian') {
    const mean = explanation.mean ?? 0;
    const std = explanation.std ?? 0;
    const sampled = explanation.continuousAction ?? mean;
    const label = explanation.actionMeanings[0] ?? 'action';
    return (
      <div className="panel">
        <div className="panel-title">Action Selection Breakdown</div>
        <div className="hint">
          Continuous Gaussian policy: the network outputs a mean μ and a spread σ; the action is
          sampled from N(μ, σ) and clamped to the valid range.
        </div>
        <div className="kv-row">
          <span>Mean μ</span>
          <span className="mono">{mean.toFixed(3)}</span>
        </div>
        <div className="kv-row">
          <span>Std σ (exploration)</span>
          <span className="mono">{std.toFixed(3)}</span>
        </div>
        {phase === 'deciding' && !revealed && (
          <button className="btn btn-reveal" onClick={revealAction}>
            Reveal action
          </button>
        )}
        {revealed && (
          <div className="reveal-box">
            <div className="kv-row">
              <span>Sampled {label}</span>
              <b className="mono">{sampled.toFixed(3)}</b>
            </div>
            <div className="rationale">{explanation.rationale}</div>
          </div>
        )}
      </div>
    );
  }

  const meanings = explanation.actionMeanings;
  const isQ = explanation.qValues != null;
  const values = explanation.qValues ?? explanation.actionProbs ?? [];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const greedy = explanation.greedyAction;
  const chosen = explanation.chosenAction;
  const showChoice = revealed; // Only show the chosen action and explore/exploit after reveal

  const canPredict = predictMode && phase === 'deciding' && !revealed;

  return (
    <div className="panel">
      <div className="panel-title">Action Selection Breakdown</div>

      <div className="hint">
        {explanation.policyKind === 'softmax'
          ? 'Policy π(a|s) (softmax). The highest probability is the greedy action.'
          : isQ
            ? 'Q-values for each action in the current state. The largest is the greedy action.'
            : 'Fixed policy: action probabilities below.'}
      </div>

      {explanation.epsilon !== undefined && (
        <div className="kv-row">
          <span>Exploration rate ε</span>
          <span className="mono">{explanation.epsilon.toFixed(3)}</span>
        </div>
      )}

      <div className="bars">
        {values.map((v, a) => {
          const isGreedy = a === greedy;
          const isChosen = showChoice && a === chosen;
          const isPredicted = predictedAction === a;
          const width = `${(normalize(v, lo, hi) * 100).toFixed(0)}%`;
          return (
            <button
              key={a}
              className="bar-row"
              disabled={!canPredict}
              onClick={() => canPredict && predict(a)}
              style={{
                cursor: canPredict ? 'pointer' : 'default',
                outline: isPredicted ? '2px solid #8ab4ff' : 'none',
              }}
            >
              <div className="bar-label">
                {meanings[a]}
              </div>
              <div className="bar-track">
                <div
                  className="bar-fill"
                  style={{
                    width,
                    background: isGreedy ? '#2e9e6b' : '#3b6ea5',
                    boxShadow: isChosen ? '0 0 0 2px #ffd54a inset' : 'none',
                  }}
                />
                <span className="bar-value mono">{v.toFixed(3)}</span>
              </div>
              {isChosen && <span className="chip chip-chosen">chosen</span>}
            </button>
          );
        })}
      </div>

      {canPredict && (
        <div className="hint predict-hint">
          Predict mode: click the action you think will be chosen, then press Reveal.
          {predictedAction !== null && (
            <> Your prediction: <b>{meanings[predictedAction]}</b></>
          )}
        </div>
      )}

      {phase === 'deciding' && !revealed && (
        <button className="btn btn-reveal" onClick={revealAction}>
          Reveal action
        </button>
      )}

      {showChoice && (
        <div className="reveal-box">
          {explanation.randomDraw !== undefined && (
            <div className="kv-row">
              <span>Random draw</span>
              <span className="mono">{explanation.randomDraw.toFixed(3)}</span>
            </div>
          )}
          <div className="kv-row">
            <span>Verdict</span>
            <span>
              {explanation.policyKind === 'softmax' ? (
                <span className="chip chip-explore">Sampled from π</span>
              ) : explanation.isExploring ? (
                <span className="chip chip-explore">Explore</span>
              ) : (
                <span className="chip chip-exploit">Exploit</span>
              )}
            </span>
          </div>
          <div className="kv-row">
            <span>Chosen action</span>
            <b>{meanings[chosen]}</b>
          </div>
          <div className="rationale">{explanation.rationale}</div>
          {predictMode && predictionCorrect !== null && (
            <div className={predictionCorrect ? 'verdict ok' : 'verdict bad'}>
              {predictionCorrect ? 'Correct!' : 'Wrong'}
              {predictedAction !== null && <> (you guessed "{meanings[predictedAction]}")</>}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
