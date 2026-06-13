// State inspector: hover a cell to see any state's V, each action's reward r(s,a), and Q(s,a) or π(a|s).
import { useStore } from '../../state/store';
import { isModeledEnv } from '../../core/types';
import { argmax } from '../../core/utils';

export function StateInspector() {
  const env = useStore((s) => s.env);
  const agent = useStore((s) => s.agent);
  const currentState = useStore((s) => s.currentState);
  const inspectedState = useStore((s) => s.inspectedState);
  useStore((s) => s.tick); // trigger re-render

  const grid = env.grid;
  if (!grid) return null;

  const s = inspectedState ?? currentState;
  const cols = grid.cols;
  const row = Math.floor(s / cols);
  const col = s % cols;
  const nStates = env.stateCount();
  const nActions = env.actionSpace.n;
  const meanings = env.actionMeanings();

  const isWall = grid.walls.includes(s);
  const isGoal = grid.goals.includes(s);
  const isHole = grid.holes.includes(s);
  const terminal = isGoal || isHole;

  const V = agent.getV ? agent.getV(nStates)[s] : null;
  const Qflat = agent.usesQ && agent.getQ ? agent.getQ(nStates, nActions) : null;
  const Pflat = !Qflat && agent.getPolicy ? agent.getPolicy(nStates, nActions) : null;
  const vals = Qflat
    ? Array.from({ length: nActions }, (_, a) => Qflat[s * nActions + a])
    : Pflat
      ? Array.from({ length: nActions }, (_, a) => Pflat[s * nActions + a])
      : null;
  const greedy = vals ? argmax(vals) : -1;
  const valHeader = Qflat ? 'Q(s,a)' : 'π(a|s)';

  const modeled = isModeledEnv(env) ? env : null;
  const rewardOf = (a: number): { r: number; stochastic: boolean } | null => {
    if (!modeled) return null;
    const outs = modeled.transitions(s, a);
    const r = outs.reduce((sum, o) => sum + o.prob * o.reward, 0);
    return { r, stochastic: outs.length > 1 };
  };

  return (
    <div className="panel">
      <div className="panel-title">State Inspector (hover any cell)</div>
      <div className="kv-row">
        <span>Inspecting</span>
        <span>{inspectedState !== null ? 'hovered cell' : 'current state'}</span>
      </div>
      <div className="kv-row">
        <span>State s</span>
        <span className="mono">
          {s} (row {row}, col {col})
        </span>
      </div>
      <div className="kv-row">
        <span>V(s)</span>
        <span className="mono">{V !== null ? V.toFixed(3) : '— (no V for policy methods)'}</span>
      </div>

      {isWall ? (
        <div className="hint">This is a wall (impassable).</div>
      ) : terminal ? (
        <div className="hint">{isGoal ? 'Goal' : 'Trap'}: terminal state, no action values.</div>
      ) : (
        <>
          <div className="inspect-grid">
            <div className="inspect-head">
              <span>Action a</span>
              <span>reward r(s,a)</span>
              <span>{valHeader}</span>
            </div>
            {meanings.map((m, a) => {
              const rw = rewardOf(a);
              const isG = a === greedy;
              return (
                <div className={'inspect-row' + (isG ? ' greedy' : '')} key={a}>
                  <span>
                    {m}
                  </span>
                  <span className="mono">
                    {rw ? (rw.stochastic ? '≈' : '') + rw.r.toFixed(2) : '—'}
                  </span>
                  <span className="mono">{vals ? vals[a].toFixed(3) : '—'}</span>
                </div>
              );
            })}
          </div>
          <div className="hint">
            reward depends on the next cell reached (goal/trap have special values, otherwise the step cost); slippery envs show the expected value (≈).
            {Pflat && ' π is the current policy probability (REINFORCE learns a policy, not Q/V).'}
          </div>
        </>
      )}
    </div>
  );
}
