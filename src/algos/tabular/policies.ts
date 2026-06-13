// ε-greedy selection + breakdown info.
import type { RNG } from '../../core/rng';
import type { ActionExplanation, Obs } from '../../core/types';
import { argmaxTie } from '../../core/utils';

export interface EpsGreedyResult {
  action: number;
  greedy: number;
  randomDraw: number;
  isExploring: boolean;
}

export function epsilonGreedy(q: number[], epsilon: number, rng: RNG): EpsGreedyResult {
  const greedy = argmaxTie(q, rng);
  const draw = rng.next();
  if (draw < epsilon) {
    const action = rng.int(q.length);
    return { action, greedy, randomDraw: draw, isExploring: true };
  }
  return { action: greedy, greedy, randomDraw: draw, isExploring: false };
}

// Assemble an ActionExplanation from the ε-greedy result (shared by Q-learning / SARSA).
export function explainEpsGreedy(
  state: Obs,
  q: number[],
  actionMeanings: string[],
  epsilon: number,
  greedyOnly: boolean,
  rng: RNG,
): { action: number; explanation: ActionExplanation } {
  const greedy = argmaxTie(q, rng);
  if (greedyOnly) {
    return {
      action: greedy,
      explanation: {
        state,
        actionMeanings,
        policyKind: 'greedy',
        qValues: q,
        greedyAction: greedy,
        epsilon,
        chosenAction: greedy,
        isExploring: false,
        rationale: `Greedy: pick the max-Q action "${actionMeanings[greedy]}" (Q=${q[greedy].toFixed(3)})`,
      },
    };
  }
  const res = epsilonGreedy(q, epsilon, rng);
  const rationale = res.isExploring
    ? `Explore: drew ${res.randomDraw.toFixed(3)} < ε=${epsilon.toFixed(
        3,
      )} → random action "${actionMeanings[res.action]}"`
    : `Exploit: drew ${res.randomDraw.toFixed(3)} ≥ ε=${epsilon.toFixed(
        3,
      )} → max-Q action "${actionMeanings[res.greedy]}" (Q=${q[res.greedy].toFixed(3)})`;
  return {
    action: res.action,
    explanation: {
      state,
      actionMeanings,
      policyKind: 'epsilon-greedy',
      qValues: q,
      greedyAction: res.greedy,
      epsilon,
      randomDraw: res.randomDraw,
      isExploring: res.isExploring,
      chosenAction: res.action,
      rationale,
    },
  };
}
