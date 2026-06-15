// Global state + the "single-step reveal sequence" state machine.
// Reveal sequence: deciding (show Q → predict → reveal action) → result (execute + update) → advance (move to the next state).
//
// Two training paths:
//  - Tabular algorithms train SYNCHRONOUSLY on the main thread (runEpisodes) — instant for small envs.
//  - Deep algorithms (DQN, …) train in a Web Worker (TensorFlow.js); progress streams back and the
//    trained weights are loaded into the main-thread agent so the step-by-step view keeps working.
import { create } from 'zustand';
import type {
  ActionExplanation,
  Agent,
  AnyEnv,
  Obs,
  StepResult,
  TabularEnvironment,
  UpdateInfo,
} from '../core/types';
import { ENV_REGISTRY, getEnvEntry } from '../envs/registry';
import {
  ALGO_REGISTRY,
  DEFAULT_HYPERPARAMS,
  algoSupportsEnv,
  defaultHyperparams,
  getAlgoEntry,
} from '../algos/registry';
import type { FromWorker, StartMsg, ToWorker } from '../training/protocol';
import type { WeightDump } from '../core/nn/weights';

const SEED = 42;
const DEFAULT_BOX_ENV = 'cartpole-vec';
const DEFAULT_DISCRETE_ENV = 'gridworld';

export type Phase = 'deciding' | 'result';
export type TrainingStatus = 'idle' | 'running' | 'done';

interface Decision {
  action: number;
  explanation: ActionExplanation;
}

export interface EpisodeReturn {
  episode: number;
  return: number;
}

// Deep agents expose these beyond the base Agent interface (weight transfer + cleanup).
interface Trainable {
  loadWeightDump?(w: WeightDump[]): void;
  setEpsilon?(e: number): void;
  dispose?(): void;
}

interface RLState {
  // ---- settings ----
  envId: string;
  algoId: string;
  hyperparams: Record<string, number>;
  predictMode: boolean;
  speed: number; // ms / auto step
  maxSteps: number; // per-episode step limit (reaching it triggers truncated)
  // ---- runtime objects ----
  env: AnyEnv;
  agent: Agent;
  agentLoading: boolean; // true while a deep agent's module/model is loading (async)
  // ---- reveal sequence state ----
  currentState: Obs;
  inspectedState: number | null; // grid cell the mouse hovers (inspection); null = current state
  phase: Phase;
  revealed: boolean;
  explanation: ActionExplanation | null;
  chosenAction: number | null;
  carry: Decision | null; // SARSA: the action already decided for the next state
  lastStep: StepResult | null;
  lastUpdate: UpdateInfo | null;
  lastEpisodeUpdate: UpdateInfo | null; // end-of-episode update summary (MC/REINFORCE)
  predictedAction: number | null;
  predictionCorrect: boolean | null;
  // ---- counters ----
  episode: number;
  stepInEpisode: number;
  totalSteps: number;
  cumulativeReward: number;
  episodeReturns: EpisodeReturn[];
  comparisonRuns: ComparisonRun[];
  compareEpisodes: number;
  runCounter: number;
  compareStatus: 'idle' | 'running'; // a deep "Run & add" comparison run is training in a transient worker
  compareProgress: number;
  compareTotal: number;
  isPlaying: boolean;
  tick: number; // triggers a canvas redraw
  // ---- deep-RL training slice (Web Worker) ----
  trainingStatus: TrainingStatus;
  trainingEpisode: number;
  trainingTotal: number;
  metricHistory: Record<string, number[]>; // per-metric time series (loss/tdError for DQN; policyLoss/… for PPO)
  metricLatest: Record<string, number>; // latest scalar metric values (bufferFill, epsilon, approxKL, …)
  bestAvg: number; // best greedy return found during training (the policy kept for inference)
  trainingError: string | null;
  // ---- actions ----
  setEnv: (id: string) => void;
  setAlgo: (id: string) => void;
  setHyperparam: (key: string, value: number) => void;
  setPredictMode: (on: boolean) => void;
  setSpeed: (ms: number) => void;
  setMaxSteps: (n: number) => void;
  setRewardParam: (key: string, value: number) => void;
  setInspectedState: (s: number | null) => void;
  predict: (a: number) => void;
  revealAction: () => void;
  executeStep: () => void;
  advance: () => void;
  quickStep: () => void;
  train: (episodes: number) => void;
  startTraining: (episodes: number) => void;
  cancelTraining: () => void;
  resetAgent: () => void;
  resetEpisode: () => void;
  play: () => void;
  pause: () => void;
  runComparison: () => void;
  snapshotCurrentRun: () => void;
  setCompareEpisodes: (n: number) => void;
  removeComparisonRun: (id: string) => void;
  clearComparisonRuns: () => void;
}

function decide(agent: Agent, state: Obs, carry: Decision | null): Decision {
  if (carry) return carry;
  return agent.act(state);
}

export interface ComparisonRun {
  id: string;
  label: string;
  color: string;
  returns: number[];
}

// Color palette for the comparison curves (mutually distinguishable, easy on the eyes).
const COMPARE_PALETTE = ['#5b9bd5', '#5fc587', '#e0a458', '#cf6f8f', '#9b8cdb', '#46b8b0'];

// Run n episodes on the given env+agent, returning each episode's return and the total step count. Pure function, does not touch the store.
function runEpisodes(
  env: TabularEnvironment,
  agent: Agent,
  n: number,
): { returns: number[]; steps: number } {
  const returns: number[] = [];
  let steps = 0;
  for (let i = 0; i < n; i++) {
    let s = env.resetSync() as number;
    let a = agent.selectAction(s);
    let ret = 0;
    let done = false;
    while (!done) {
      const r = env.stepSync(a);
      const sN = r.observation as number;
      const finished = r.terminated || r.truncated;
      const aN = finished ? a : agent.selectAction(sN);
      agent.update({
        state: s,
        action: a,
        reward: r.reward,
        nextState: sN,
        terminated: r.terminated,
        truncated: r.truncated,
        nextAction: aN,
      });
      ret += r.reward;
      steps += 1;
      s = sN;
      a = aN;
      done = finished;
    }
    agent.onEpisodeEnd();
    returns.push(ret);
  }
  return { returns, steps };
}

// Build the comparison-curve label from the algorithm + its key hyperparameters.
function autoLabel(algoId: string, hp: Record<string, number>): string {
  const entry = getAlgoEntry(algoId);
  if (entry.deep) {
    return `${entry.name} lr${+(hp.lr ?? 0).toFixed(4)} γ${(hp.gamma ?? 0).toFixed(2)}`;
  }
  let label = `${entry.name} α${hp.alpha.toFixed(2)} γ${hp.gamma.toFixed(2)}`;
  if (entry.usesEpsilon) label += ` ε${hp.epsilon.toFixed(2)}`;
  return label;
}

// Suffix describing how the current env's reward / maxSteps differ from its defaults (for run labels).
function comparisonSuffix(envId: string, env: AnyEnv, maxSteps: number): string {
  const tmp = getEnvEntry(envId).create(SEED);
  const defParams = tmp.rewardParams?.() ?? [];
  const defMax = tmp.maxSteps;
  const defMap = new Map(defParams.map((d) => [d.key, d.value]));
  const diffs = (env.rewardParams?.() ?? [])
    .filter((p) => Math.abs(p.value - (defMap.get(p.key) ?? p.value)) > 1e-9)
    .map((p) => `${p.label}${+p.value.toFixed(3)}`);
  if (maxSteps !== defMax) diffs.push(`maxSteps${maxSteps}`);
  return diffs.length ? ' · ' + diffs.join(' ') : '';
}

// Pick an env compatible with the algorithm (deep needs continuous/box obs; tabular needs discrete).
function compatibleEnvId(curEnvId: string, algoId: string): string {
  const algo = getAlgoEntry(algoId);
  const cur = getEnvEntry(curEnvId);
  if (algoSupportsEnv(algo, cur)) return curEnvId; // already compatible
  // Prefer a compatible env in the SAME comparison group (e.g. cartpole-vec <-> cartpole), so
  // switching DQN <-> tabular stays on the same task and keeps the comparison chart intact.
  const sameGroup = ENV_REGISTRY.find((e) => e.compareGroup === cur.compareGroup && algoSupportsEnv(algo, e));
  if (sameGroup) return sameGroup.id;
  const any = ENV_REGISTRY.find((e) => algoSupportsEnv(algo, e));
  if (any) return any.id;
  return algo.requires === 'box-obs' ? DEFAULT_BOX_ENV : DEFAULT_DISCRETE_ENV;
}

// When switching environments, pick an algorithm compatible with the new env (keep current if it fits).
function compatibleAlgoId(envId: string, curAlgoId: string): string {
  const env = getEnvEntry(envId);
  if (algoSupportsEnv(getAlgoEntry(curAlgoId), env)) return curAlgoId;
  const any = ALGO_REGISTRY.find((a) => algoSupportsEnv(a, env));
  return any ? any.id : curAlgoId;
}

// Identity of an algorithm's hyperparameter "shape": switching between algos with the same shape
// keeps the current values (so tabular-vs-tabular comparison stays controlled); a different shape resets.
function hpShapeKey(algoId: string): string {
  const e = getAlgoEntry(algoId);
  return e.hyperparamSpec ? e.hyperparamSpec.map((s) => s.key).join(',') : 'tabular';
}

// Fresh interactive state after (re)creating an agent: reset the episode and decide the first action.
function freshInteractive(agent: Agent, env: AnyEnv): Partial<RLState> {
  const s0 = env.resetSync(SEED);
  const d = agent.act(s0);
  return {
    currentState: s0,
    phase: 'deciding',
    revealed: false,
    explanation: d.explanation,
    chosenAction: d.action,
    carry: null,
    lastStep: null,
    lastUpdate: null,
    lastEpisodeUpdate: null,
    predictedAction: null,
    predictionCorrect: null,
    episode: 0,
    stepInEpisode: 0,
    totalSteps: 0,
    cumulativeReward: 0,
    episodeReturns: [],
    isPlaying: false,
  };
}

function resetTrainingSlice(): Partial<RLState> {
  return {
    trainingStatus: 'idle',
    trainingEpisode: 0,
    trainingTotal: 0,
    metricHistory: {},
    metricLatest: {},
    bestAvg: 0,
    trainingError: null,
  };
}

export const useStore = create<RLState>((set, get) => {
  // ---- training Web Worker (created lazily on first deep training) ----
  let worker: Worker | null = null;
  // Whether the worker holds a trained agent we can CONTINUE from. Reset whenever the agent is
  // rebuilt (reset / algo / env / hyperparam change) so the next deep Train starts fresh.
  let sessionTrained = false;
  const ensureWorker = (): Worker => {
    if (worker) return worker;
    worker = new Worker(new URL('../training/trainer.worker.ts', import.meta.url), {
      type: 'module',
    });
    worker.onmessage = (e: MessageEvent<FromWorker>) => {
      const msg = e.data;
      if (msg.type === 'progress') {
        set((s) => {
          const hist: Record<string, number[]> = { ...s.metricHistory };
          for (const k in msg.metrics) hist[k] = [...(hist[k] ?? []), msg.metrics[k]];
          return {
            trainingEpisode: msg.episode,
            // Append with a cumulative index so the curve continues smoothly across Train presses.
            episodeReturns: [...s.episodeReturns, { episode: s.episodeReturns.length + 1, return: msg.ret }],
            metricHistory: hist,
            metricLatest: msg.metrics,
            tick: s.tick + 1,
          };
        });
      } else if (msg.type === 'done') {
        sessionTrained = true; // worker now holds a trained agent → the next Train continues it
        const ag = get().agent as unknown as Trainable;
        if (ag.loadWeightDump) {
          ag.loadWeightDump(msg.weights); // BEST weights found during training
          ag.setEpsilon?.(0); // inference = greedy: watch the best learned policy cleanly
          // Refresh the interactive decision on the current state so it reflects the trained policy.
          const d = get().agent.act(get().currentState);
          set((s) => ({
            trainingStatus: 'done',
            bestAvg: msg.bestAvg,
            phase: 'deciding',
            revealed: false,
            explanation: d.explanation,
            chosenAction: d.action,
            tick: s.tick + 1,
          }));
        } else {
          set((s) => ({ trainingStatus: 'done', tick: s.tick + 1 }));
        }
      } else if (msg.type === 'error') {
        set((s) => ({ trainingStatus: 'idle', trainingError: msg.message, tick: s.tick + 1 }));
      }
    };
    return worker;
  };

  // ---- transient worker for a deep "Run & add to comparison": trains a FRESH agent headlessly and
  // adds its learning curve, without touching the interactive agent; terminated when done. ----
  const startCompareWorker = (st: RLState) => {
    const w = new Worker(new URL('../training/trainer.worker.ts', import.meta.url), { type: 'module' });
    const label = autoLabel(st.algoId, st.hyperparams) + comparisonSuffix(st.envId, st.env, st.maxSteps);
    const color = COMPARE_PALETTE[st.runCounter % COMPARE_PALETTE.length];
    const rewardParams = (st.env.rewardParams?.() ?? []).map((p) => ({ key: p.key, value: p.value }));
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      const msg = e.data;
      if (msg.type === 'progress') {
        set((s) => ({ compareProgress: msg.episode, tick: s.tick + 1 }));
      } else if (msg.type === 'done') {
        const run: ComparisonRun = { id: 'run-' + get().runCounter, label, color, returns: msg.returns };
        set((s) => ({
          comparisonRuns: [...s.comparisonRuns, run],
          runCounter: s.runCounter + 1,
          compareStatus: 'idle',
          compareProgress: 0,
          tick: s.tick + 1,
        }));
        w.terminate();
      } else if (msg.type === 'error') {
        set((s) => ({ compareStatus: 'idle', compareProgress: 0, trainingError: msg.message, tick: s.tick + 1 }));
        w.terminate();
      }
    };
    set({ compareStatus: 'running', compareProgress: 0, compareTotal: st.compareEpisodes });
    const startMsg: StartMsg = {
      type: 'start',
      algoId: st.algoId,
      envId: st.envId,
      hyperparams: { ...st.hyperparams, keepBest: 0 }, // skip greedy-eval overhead; we only need the returns
      rewardParams,
      maxSteps: st.maxSteps,
      episodes: st.compareEpisodes,
      seed: SEED,
      fresh: true,
    };
    w.postMessage(startMsg);
  };

  // ---- (re)create the agent for the current env+algo, sync (tabular) or async (deep). ----
  const buildAgent = (envId: string, algoId: string, env: AnyEnv, hp: Record<string, number>, extra: Partial<RLState>) => {
    const entry = getAlgoEntry(algoId);
    (get().agent as unknown as Trainable).dispose?.();
    sessionTrained = false; // new agent → the next deep Train starts a fresh network
    const head: Partial<RLState> = {
      envId,
      algoId,
      env,
      hyperparams: hp,
      maxSteps: env.maxSteps,
      ...resetTrainingSlice(),
      ...extra,
    };
    if (entry.deep) {
      // Lazy import keeps tfjs out of the main bundle. Until it resolves, show a loading state.
      const s0 = env.resetSync(SEED);
      set((s) => ({
        ...head,
        agentLoading: true,
        currentState: s0,
        phase: 'deciding',
        revealed: false,
        explanation: null,
        chosenAction: null,
        carry: null,
        lastStep: null,
        lastUpdate: null,
        lastEpisodeUpdate: null,
        predictedAction: null,
        predictionCorrect: null,
        episode: 0,
        stepInEpisode: 0,
        totalSteps: 0,
        cumulativeReward: 0,
        episodeReturns: [],
        isPlaying: false,
        tick: s.tick + 1,
      }));
      entry.load!().then((mod) => {
        if (get().algoId !== algoId || get().env !== env) return; // user moved on
        const a = mod.create(env, hp);
        const d = a.act(get().currentState);
        set((s) => ({
          agent: a,
          agentLoading: false,
          explanation: d.explanation,
          chosenAction: d.action,
          tick: s.tick + 1,
        }));
      });
    } else {
      const a = entry.create!(env as TabularEnvironment, hp);
      set((s) => ({ ...head, agent: a, agentLoading: false, ...freshInteractive(a, env), tick: s.tick + 1 }));
    }
  };

  // initialization (tabular default — no tfjs loaded)
  const env = getEnvEntry('gridworld').create(SEED);
  const hp = { ...DEFAULT_HYPERPARAMS };
  const agent = getAlgoEntry('q-learning').create!(env as TabularEnvironment, hp);
  const s0 = env.resetSync(SEED);
  const firstDecision = agent.act(s0);

  return {
    envId: 'gridworld',
    algoId: 'q-learning',
    hyperparams: hp,
    predictMode: false,
    speed: 350,
    maxSteps: env.maxSteps,
    env,
    agent,
    agentLoading: false,
    currentState: s0,
    inspectedState: null,
    phase: 'deciding',
    revealed: false,
    explanation: firstDecision.explanation,
    chosenAction: firstDecision.action,
    carry: null,
    lastStep: null,
    lastUpdate: null,
    lastEpisodeUpdate: null,
    predictedAction: null,
    predictionCorrect: null,
    episode: 0,
    stepInEpisode: 0,
    totalSteps: 0,
    cumulativeReward: 0,
    episodeReturns: [],
    comparisonRuns: [],
    compareEpisodes: 300,
    runCounter: 0,
    compareStatus: 'idle',
    compareProgress: 0,
    compareTotal: 0,
    isPlaying: false,
    tick: 0,
    trainingStatus: 'idle',
    trainingEpisode: 0,
    trainingTotal: 0,
    metricHistory: {},
    metricLatest: {},
    bestAvg: 0,
    trainingError: null,

    setEnv: (id) => {
      const algoId = compatibleAlgoId(id, get().algoId);
      const groupChanged = getEnvEntry(id).compareGroup !== getEnvEntry(get().envId).compareGroup;
      const e = getEnvEntry(id).create(SEED);
      const hyperparams =
        algoId === get().algoId ? get().hyperparams : defaultHyperparams(getAlgoEntry(algoId));
      buildAgent(id, algoId, e, hyperparams, groupChanged ? { comparisonRuns: [] } : {});
    },

    setAlgo: (id) => {
      const envId = compatibleEnvId(get().envId, id);
      const envChanged = envId !== get().envId;
      // Keep the comparison chart unless the comparison GROUP changes (so DQN <-> tabular on
      // CartPole keeps its curves; only a real task change like CartPole -> GridWorld resets it).
      const groupChanged = getEnvEntry(envId).compareGroup !== getEnvEntry(get().envId).compareGroup;
      const e = envChanged ? getEnvEntry(envId).create(SEED) : get().env;
      const hyperparams =
        hpShapeKey(get().algoId) === hpShapeKey(id) ? get().hyperparams : defaultHyperparams(getAlgoEntry(id));
      buildAgent(envId, id, e, hyperparams, groupChanged ? { comparisonRuns: [] } : {});
    },

    setHyperparam: (key, value) => {
      const hyperparams = { ...get().hyperparams, [key]: value };
      get().agent.hyperparams[key] = value;
      sessionTrained = false; // changed settings take effect on the next (fresh) deep training run
      set({ hyperparams });
    },

    setPredictMode: (on) => set({ predictMode: on, predictedAction: null, predictionCorrect: null }),

    setSpeed: (ms) => set({ speed: ms }),

    setMaxSteps: (n) => {
      get().env.maxSteps = n; // apply immediately to the interactive environment
      set({ maxSteps: n });
    },

    setRewardParam: (key, value) => {
      get().env.setRewardParam?.(key, value); // change reward immediately; reset agent / retrain to learn under it
      set((s) => ({ tick: s.tick + 1 }));
    },

    setInspectedState: (s) => {
      if (get().inspectedState === s) return;
      set((st) => ({ inspectedState: s, tick: st.tick + 1 }));
    },

    predict: (a) => {
      if (get().revealed) return;
      set({ predictedAction: a });
    },

    revealAction: () => {
      const st = get();
      if (st.revealed || st.phase !== 'deciding' || st.agentLoading) return;
      const predictionCorrect =
        st.predictMode && st.predictedAction !== null && st.chosenAction !== null
          ? st.predictedAction === st.chosenAction
          : null;
      set((s) => ({ revealed: true, predictionCorrect, tick: s.tick + 1 }));
    },

    executeStep: () => {
      const st = get();
      if (st.phase !== 'deciding' || !st.revealed || st.chosenAction === null || st.agentLoading) return;
      const a = st.chosenAction;
      const step = st.env.stepSync(a);
      const sN = step.observation;
      const finished = step.terminated || step.truncated;
      // Draw the next action first (the on-policy a') and keep it for the next state
      let carry: Decision | null = null;
      let nextAction: number | undefined;
      if (!finished) {
        carry = st.agent.act(sN);
        nextAction = carry.action;
      }
      const upd = st.agent.update({
        state: st.currentState,
        action: a,
        reward: step.reward,
        nextState: sN,
        terminated: step.terminated,
        truncated: step.truncated,
        nextAction,
      });
      const epUpd: UpdateInfo | null = finished ? (st.agent.onEpisodeEnd() ?? null) : null;
      set((s) => ({
        phase: 'result',
        lastStep: step,
        lastUpdate: upd,
        lastEpisodeUpdate: epUpd,
        carry,
        cumulativeReward: s.cumulativeReward + step.reward,
        stepInEpisode: s.stepInEpisode + 1,
        totalSteps: s.totalSteps + 1,
        tick: s.tick + 1,
      }));
    },

    advance: () => {
      const st = get();
      if (st.phase !== 'result' || !st.lastStep) return;
      const finished = st.lastStep.terminated || st.lastStep.truncated;
      if (finished) {
        const ep = st.episode + 1;
        const ret = st.cumulativeReward;
        const s0n = st.env.resetSync();
        const d = st.agent.act(s0n);
        set((s) => ({
          episode: ep,
          episodeReturns: [...s.episodeReturns, { episode: ep, return: ret }],
          currentState: s0n,
          phase: 'deciding',
          revealed: false,
          explanation: d.explanation,
          chosenAction: d.action,
          carry: null,
          cumulativeReward: 0,
          stepInEpisode: 0,
          predictedAction: null,
          predictionCorrect: null,
          lastStep: null,
          lastUpdate: null,
          tick: s.tick + 1,
        }));
      } else {
        const sN = st.lastStep.observation;
        const d = decide(st.agent, sN, st.carry);
        set((s) => ({
          currentState: sN,
          phase: 'deciding',
          revealed: false,
          explanation: d.explanation,
          chosenAction: d.action,
          carry: null,
          predictedAction: null,
          predictionCorrect: null,
          lastStep: null,
          lastUpdate: null,
          tick: s.tick + 1,
        }));
      }
    },

    quickStep: () => {
      if (get().agentLoading) return;
      const { phase, revealed } = get();
      if (phase === 'deciding' && !revealed) get().revealAction();
      if (get().phase === 'deciding') get().executeStep();
      if (get().phase === 'result') get().advance();
    },

    train: (episodes) => {
      if (getAlgoEntry(get().algoId).deep) {
        get().startTraining(episodes);
        return;
      }
      const st = get();
      const env2 = st.env as TabularEnvironment;
      const startEp = st.episode;
      const { returns: rets, steps } = runEpisodes(env2, st.agent, episodes);
      const episodeReturns = [...st.episodeReturns];
      rets.forEach((ret, i) => episodeReturns.push({ episode: startEp + i + 1, return: ret }));
      const s0n = env2.resetSync() as number;
      const d = st.agent.act(s0n);
      set((s) => ({
        episode: startEp + episodes,
        totalSteps: st.totalSteps + steps,
        episodeReturns,
        currentState: s0n,
        phase: 'deciding',
        revealed: false,
        explanation: d.explanation,
        chosenAction: d.action,
        carry: null,
        cumulativeReward: 0,
        stepInEpisode: 0,
        predictedAction: null,
        predictionCorrect: null,
        lastStep: null,
        lastUpdate: null,
        tick: s.tick + 1,
      }));
    },

    startTraining: (episodes) => {
      const st = get();
      if (!getAlgoEntry(st.algoId).deep) return;
      const w = ensureWorker();
      // Continue the current network unless this is the first run after a reset / settings change.
      const fresh = !sessionTrained;
      const rewardParams = (st.env.rewardParams?.() ?? []).map((p) => ({ key: p.key, value: p.value }));
      set((s) => ({
        trainingStatus: 'running',
        trainingEpisode: 0,
        trainingTotal: episodes,
        trainingError: null,
        // Fresh run clears the curves + ε display; a continuation keeps appending to them.
        ...(fresh ? { episodeReturns: [], metricHistory: {}, metricLatest: {}, bestAvg: 0 } : {}),
        tick: s.tick + 1,
      }));
      const startMsg: StartMsg = {
        type: 'start',
        algoId: st.algoId,
        envId: st.envId,
        hyperparams: st.hyperparams,
        rewardParams,
        maxSteps: st.maxSteps,
        episodes,
        seed: SEED,
        fresh,
      };
      w.postMessage(startMsg);
    },

    cancelTraining: () => {
      if (worker) {
        const cancel: ToWorker = { type: 'cancel' };
        worker.postMessage(cancel);
      }
      set((s) => ({ trainingStatus: 'done', tick: s.tick + 1 }));
    },

    resetAgent: () => {
      buildAgent(get().envId, get().algoId, get().env, get().hyperparams, {});
    },

    resetEpisode: () => {
      const st = get();
      const s0n = st.env.resetSync();
      const d = st.agent.act(s0n);
      set((s) => ({
        currentState: s0n,
        phase: 'deciding',
        revealed: false,
        explanation: d.explanation,
        chosenAction: d.action,
        carry: null,
        cumulativeReward: 0,
        stepInEpisode: 0,
        predictedAction: null,
        predictionCorrect: null,
        lastStep: null,
        lastUpdate: null,
        tick: s.tick + 1,
      }));
    },

    play: () => set({ isPlaying: true }),
    pause: () => set({ isPlaying: false }),

    runComparison: () => {
      const st = get();
      const entry = getAlgoEntry(st.algoId);
      if (entry.deep) {
        // Async: train a fresh agent in a transient worker, then add its curve (see startCompareWorker).
        if (st.compareStatus === 'running' || st.trainingStatus === 'running') return;
        startCompareWorker(st);
        return;
      }
      // Tabular: synchronous headless run with a fresh agent + env.
      const tmpEnv = getEnvEntry(st.envId).create(SEED) as TabularEnvironment;
      tmpEnv.maxSteps = st.maxSteps;
      (st.env.rewardParams?.() ?? []).forEach((p) => tmpEnv.setRewardParam?.(p.key, p.value));
      const tmpAgent = entry.create!(tmpEnv, st.hyperparams);
      const { returns } = runEpisodes(tmpEnv, tmpAgent, st.compareEpisodes);
      const run: ComparisonRun = {
        id: 'run-' + st.runCounter,
        label: autoLabel(st.algoId, st.hyperparams) + comparisonSuffix(st.envId, st.env, st.maxSteps),
        color: COMPARE_PALETTE[st.runCounter % COMPARE_PALETTE.length],
        returns,
      };
      set((s) => ({
        comparisonRuns: [...s.comparisonRuns, run],
        runCounter: s.runCounter + 1,
        tick: s.tick + 1,
      }));
    },

    setCompareEpisodes: (n) => set({ compareEpisodes: n }),

    removeComparisonRun: (id) =>
      set((s) => ({ comparisonRuns: s.comparisonRuns.filter((r) => r.id !== id) })),

    clearComparisonRuns: () => set({ comparisonRuns: [] }),

    // Persist the current interactive learning curve as a comparison run WITHOUT re-running it.
    // Lets you keep an (expensive) DQN run and overlay it against tabular methods on the same task.
    snapshotCurrentRun: () => {
      const st = get();
      const returns = st.episodeReturns.map((e) => e.return);
      if (returns.length === 0) return;
      const run: ComparisonRun = {
        id: 'run-' + st.runCounter,
        label: autoLabel(st.algoId, st.hyperparams),
        color: COMPARE_PALETTE[st.runCounter % COMPARE_PALETTE.length],
        returns,
      };
      set((s) => ({
        comparisonRuns: [...s.comparisonRuns, run],
        runCounter: s.runCounter + 1,
        tick: s.tick + 1,
      }));
    },
  };
});

// During development, attach the store to window for easier debugging/verification (not present in production builds).
if (import.meta.env.DEV) {
  (window as unknown as { rlstore?: typeof useStore }).rlstore = useStore;
}
