/// <reference lib="webworker" />
// Training Web Worker: runs heavy neural-network training off the main thread so the UI
// never freezes. Uses the TensorFlow.js wasm backend (no OffscreenCanvas needed, works
// everywhere). Streams per-episode progress + metrics, then ships the trained weights back.
import * as tf from '@tensorflow/tfjs';
import { setWasmPaths } from '@tensorflow/tfjs-backend-wasm';
import wasmSimdPath from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-simd.wasm?url';
import wasmThreadedSimdPath from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm-threaded-simd.wasm?url';
import wasmPath from '@tensorflow/tfjs-backend-wasm/dist/tfjs-backend-wasm.wasm?url';
import type { VecEnvironment } from '../core/types';
import { getEnvEntry } from '../envs/registry';
import { DQN } from '../algos/deep/DQN';
import type { WeightDump } from '../core/nn/weights';
import type { FromWorker, StartMsg, ToWorker } from './protocol';

setWasmPaths({
  'tfjs-backend-wasm.wasm': wasmPath,
  'tfjs-backend-wasm-simd.wasm': wasmSimdPath,
  'tfjs-backend-wasm-threaded-simd.wasm': wasmThreadedSimdPath,
});

const ctx = self as unknown as DedicatedWorkerGlobalScope;
const post = (m: FromWorker) => ctx.postMessage(m);

let cancelled = false;
let backendReady: Promise<void> | null = null;
// Persisted across "start" messages so consecutive Train presses CONTINUE the same network
// (weights, replay buffer, and ε all carry over). A fresh run (or a reset) rebuilds it.
let dqnAgent: DQN | null = null;
// Best policy seen so far (persists across continue-training presses; reset on a fresh run).
let bestAvg = -Infinity;
let bestWeights: WeightDump | null = null;

function ensureBackend(): Promise<void> {
  if (!backendReady) {
    backendReady = (async () => {
      try {
        await tf.setBackend('wasm');
        await tf.ready();
      } catch {
        // Fall back to the pure-JS CPU backend if wasm is unavailable.
        await tf.setBackend('cpu');
        await tf.ready();
      }
    })();
  }
  return backendReady;
}

ctx.onmessage = async (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  if (msg.type === 'cancel') {
    cancelled = true;
    return;
  }
  if (msg.type === 'start') {
    cancelled = false;
    try {
      await ensureBackend();
      await runDQN(msg);
    } catch (err) {
      post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
    }
  }
};

// Evaluate the current GREEDY policy (no exploration) over n episodes; returns the mean return.
function greedyEval(agent: DQN, env: VecEnvironment, n: number): number {
  let total = 0;
  for (let e = 0; e < n; e++) {
    let o = env.resetSync();
    let done = false;
    let steps = 0;
    while (!done && steps < env.maxSteps) {
      const a = agent.selectAction(o, true);
      const r = env.stepSync(a);
      o = r.observation as number[];
      total += r.reward;
      done = r.terminated || r.truncated;
      steps += 1;
    }
  }
  return total / n;
}

async function runDQN(msg: StartMsg): Promise<void> {
  const env = getEnvEntry(msg.envId).create(msg.seed) as VecEnvironment;
  env.maxSteps = msg.maxSteps;
  for (const p of msg.rewardParams) env.setRewardParam?.(p.key, p.value);
  // Separate env for greedy evaluation (independent of the training env's RNG/state).
  const evalEnv = getEnvEntry(msg.envId).create(msg.seed + 7) as VecEnvironment;
  evalEnv.maxSteps = msg.maxSteps;
  for (const p of msg.rewardParams) evalEnv.setRewardParam?.(p.key, p.value);

  const obs = env.observationSpace;
  const inputDim = obs.shape[0];
  const nActions = env.actionSpace.kind === 'discrete' ? env.actionSpace.n : 1;

  // Fresh run → new network; continuation → keep the existing agent (its weights/replay/ε persist).
  let agent: DQN;
  if (msg.fresh || !dqnAgent) {
    dqnAgent?.dispose();
    agent = new DQN(inputDim, nActions, env.actionMeanings(), msg.hyperparams, obs.low, obs.high, msg.seed + 1);
    dqnAgent = agent;
    bestAvg = -Infinity;
    bestWeights = null;
  } else {
    agent = dqnAgent;
  }
  const keepBest = (msg.hyperparams.keepBest ?? 1) >= 0.5;

  const returns: number[] = [];
  const recent: number[] = [];
  let lastLoss = 0;
  let lastTd = 0;

  for (let ep = 0; ep < msg.episodes; ep++) {
    if (cancelled) break;
    let s = env.resetSync() as number[];
    let done = false;
    let ret = 0;
    while (!done) {
      const a = agent.selectAction(s);
      const r = env.stepSync(a);
      const s2 = r.observation as number[];
      agent.pushTransition(s, a, r.reward, s2, r.terminated);
      const m = agent.trainStep();
      if (m) {
        lastLoss = m.loss;
        lastTd = m.tdError;
      }
      ret += r.reward;
      s = s2;
      done = r.terminated || r.truncated;
    }
    agent.decayEpsilon();
    returns.push(ret);
    recent.push(ret);
    if (recent.length > 30) recent.shift();
    const avgReturn = recent.reduce((x, y) => x + y, 0) / recent.length;
    post({
      type: 'progress',
      episode: ep + 1,
      ret,
      avgReturn,
      metrics: {
        loss: lastLoss,
        tdError: lastTd,
        epsilon: agent.epsilonValue,
        bufferFill: agent.replaySize / agent.bufferCapacity,
        trainSteps: agent.stepsTrained,
        targetSync: agent.targetSyncEvery,
      },
    });
    // Periodically score the GREEDY policy and keep the best — this is exactly what inference uses,
    // so training longer can't make the served policy worse.
    if (keepBest && (ep + 1) % 15 === 0) {
      const g = greedyEval(agent, evalEnv, 3);
      if (g > bestAvg) {
        bestAvg = g;
        bestWeights = agent.dumpWeights();
      }
    }
    // Yield so queued 'cancel' messages are processed and the worker stays responsive.
    await new Promise((res) => setTimeout(res, 0));
  }

  // Final greedy evaluation: scores the served policy, and (if keepBest) can become the kept one.
  const finalGreedy = greedyEval(agent, evalEnv, 5);
  if (keepBest && finalGreedy > bestAvg) {
    bestAvg = finalGreedy;
    bestWeights = agent.dumpWeights();
  }
  post({
    type: 'done',
    returns,
    // keepBest → serve the best greedy policy found; otherwise serve the final weights.
    weights: keepBest && bestWeights ? bestWeights : agent.dumpWeights(),
    epsilon: agent.epsilonValue,
    bestAvg: keepBest && bestWeights ? bestAvg : finalGreedy,
  });
  // Keep the agent alive so the next Train can continue from here (disposed on a fresh run).
}
