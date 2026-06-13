// Global state + the "single-step reveal sequence" state machine.
// Reveal sequence: deciding (show Q → predict → reveal action) → result (execute + update) → advance (move to the next state).
import { create } from 'zustand';
import type {
  ActionExplanation,
  Agent,
  StepResult,
  TabularEnvironment,
  UpdateInfo,
} from '../core/types';
import { getEnvEntry } from '../envs/registry';
import { DEFAULT_HYPERPARAMS, getAlgoEntry } from '../algos/registry';

const SEED = 42;

export type Phase = 'deciding' | 'result';

interface Decision {
  action: number;
  explanation: ActionExplanation;
}

export interface EpisodeReturn {
  episode: number;
  return: number;
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
  env: TabularEnvironment;
  agent: Agent;
  // ---- reveal sequence state ----
  currentState: number;
  inspectedState: number | null; // the cell the mouse is hovering over (for inspection), null = view the current state
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
  isPlaying: boolean;
  tick: number; // triggers a canvas redraw
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
  resetAgent: () => void;
  resetEpisode: () => void;
  play: () => void;
  pause: () => void;
  runComparison: () => void;
  setCompareEpisodes: (n: number) => void;
  removeComparisonRun: (id: string) => void;
  clearComparisonRuns: () => void;
}

function decide(agent: Agent, state: number, carry: Decision | null): Decision {
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
  let label = `${entry.name} α${hp.alpha.toFixed(2)} γ${hp.gamma.toFixed(2)}`;
  if (entry.usesEpsilon) label += ` ε${hp.epsilon.toFixed(2)}`;
  return label;
}

export const useStore = create<RLState>((set, get) => {
  // initialization
  const env = getEnvEntry('gridworld').create(SEED);
  const hp = { ...DEFAULT_HYPERPARAMS };
  const agent = getAlgoEntry('q-learning').create(env, hp);
  const s0 = env.resetSync(SEED) as number;
  const first = agent.act(s0);

  return {
    envId: 'gridworld',
    algoId: 'q-learning',
    hyperparams: hp,
    predictMode: false,
    speed: 350,
    maxSteps: env.maxSteps,
    env,
    agent,
    currentState: s0,
    inspectedState: null,
    phase: 'deciding',
    revealed: false,
    explanation: first.explanation,
    chosenAction: first.action,
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
    isPlaying: false,
    tick: 0,

    setEnv: (id) => {
      const e = getEnvEntry(id).create(SEED);
      const a = getAlgoEntry(get().algoId).create(e, get().hyperparams);
      const st = e.resetSync(SEED) as number;
      const d = a.act(st);
      set({
        envId: id,
        env: e,
        agent: a,
        maxSteps: e.maxSteps, // when switching environments, reset to that environment's default step limit
        currentState: st,
        phase: 'deciding',
        revealed: false,
        explanation: d.explanation,
        chosenAction: d.action,
        carry: null,
        lastStep: null,
        lastUpdate: null,
        predictedAction: null,
        predictionCorrect: null,
        episode: 0,
        stepInEpisode: 0,
        totalSteps: 0,
        cumulativeReward: 0,
        episodeReturns: [],
        comparisonRuns: [],
        isPlaying: false,
        tick: get().tick + 1,
      });
    },

    setAlgo: (id) => {
      const e = get().env;
      const a = getAlgoEntry(id).create(e, get().hyperparams);
      const st = e.resetSync(SEED) as number;
      const d = a.act(st);
      set({
        algoId: id,
        agent: a,
        currentState: st,
        phase: 'deciding',
        revealed: false,
        explanation: d.explanation,
        chosenAction: d.action,
        carry: null,
        lastStep: null,
        lastUpdate: null,
        predictedAction: null,
        predictionCorrect: null,
        episode: 0,
        stepInEpisode: 0,
        totalSteps: 0,
        cumulativeReward: 0,
        episodeReturns: [],
        isPlaying: false,
        tick: get().tick + 1,
      });
    },

    setHyperparam: (key, value) => {
      const hyperparams = { ...get().hyperparams, [key]: value };
      get().agent.hyperparams[key] = value;
      set({ hyperparams });
    },

    setPredictMode: (on) => set({ predictMode: on, predictedAction: null, predictionCorrect: null }),

    setSpeed: (ms) => set({ speed: ms }),

    setMaxSteps: (n) => {
      get().env.maxSteps = n; // apply immediately to the interactive environment
      set({ maxSteps: n });
    },

    setRewardParam: (key, value) => {
      get().env.setRewardParam?.(key, value); // change the environment reward immediately; click "Reset Agent" or "Run a round" to use the new reward
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
      if (st.revealed || st.phase !== 'deciding') return;
      const predictionCorrect =
        st.predictMode && st.predictedAction !== null && st.chosenAction !== null
          ? st.predictedAction === st.chosenAction
          : null;
      set((s) => ({ revealed: true, predictionCorrect, tick: s.tick + 1 }));
    },

    executeStep: () => {
      const st = get();
      if (st.phase !== 'deciding' || !st.revealed || st.chosenAction === null) return;
      const a = st.chosenAction;
      const step = st.env.stepSync(a);
      const sN = step.observation as number;
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
      // On episode end (including truncation), call onEpisodeEnd: step-wise methods decay ε (return undefined), MC/REINFORCE perform learning and return a summary
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
        // onEpisodeEnd was already called in executeStep; here we only do episode counting and reset
        const ep = st.episode + 1;
        const ret = st.cumulativeReward;
        const s0 = st.env.resetSync() as number;
        const d = st.agent.act(s0);
        set((s) => ({
          episode: ep,
          episodeReturns: [...s.episodeReturns, { episode: ep, return: ret }],
          currentState: s0,
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
        const sN = st.lastStep.observation as number;
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
      const { phase, revealed } = get();
      if (phase === 'deciding' && !revealed) get().revealAction();
      if (get().phase === 'deciding') get().executeStep();
      if (get().phase === 'result') get().advance();
    },

    train: (episodes) => {
      const st = get();
      const { env, agent } = st;
      const startEp = st.episode;
      const { returns: rets, steps } = runEpisodes(env, agent, episodes);
      const episodeReturns = [...st.episodeReturns];
      rets.forEach((ret, i) => episodeReturns.push({ episode: startEp + i + 1, return: ret }));
      // Start a fresh episode to enter interactive observation
      const s0 = env.resetSync() as number;
      const d = agent.act(s0);
      set((s) => ({
        episode: startEp + episodes,
        totalSteps: st.totalSteps + steps,
        episodeReturns,
        currentState: s0,
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

    resetAgent: () => {
      const e = get().env;
      const a = getAlgoEntry(get().algoId).create(e, get().hyperparams);
      const st = e.resetSync(SEED) as number;
      const d = a.act(st);
      set((s) => ({
        agent: a,
        currentState: st,
        phase: 'deciding',
        revealed: false,
        explanation: d.explanation,
        chosenAction: d.action,
        carry: null,
        lastStep: null,
        lastUpdate: null,
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
    },

    resetEpisode: () => {
      const st = get();
      const s0 = st.env.resetSync() as number;
      const d = st.agent.act(s0);
      set((s) => ({
        currentState: s0,
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
      // headless: run with a brand-new env+agent, without touching the interactive agent/board
      const tmpEnv = getEnvEntry(st.envId).create(SEED);
      const defParams = tmpEnv.rewardParams?.() ?? []; // default rewards (record them before applying)
      const defMax = tmpEnv.maxSteps;
      tmpEnv.maxSteps = st.maxSteps; // the comparison run also applies the currently configured step limit
      const curParams = st.env.rewardParams?.() ?? [];
      curParams.forEach((p) => tmpEnv.setRewardParam?.(p.key, p.value)); // keep consistent with the current reward settings
      const tmpAgent = getAlgoEntry(st.algoId).create(tmpEnv, st.hyperparams);
      const { returns } = runEpisodes(tmpEnv, tmpAgent, st.compareEpisodes);
      // Label: algorithm + hyperparameters, plus any reward/step settings that "differ from the defaults" (default values are not shown)
      const defMap = new Map(defParams.map((d) => [d.key, d.value]));
      const diffs = curParams
        .filter((p) => Math.abs(p.value - (defMap.get(p.key) ?? p.value)) > 1e-9)
        .map((p) => `${p.label}${+p.value.toFixed(3)}`);
      if (st.maxSteps !== defMax) diffs.push(`maxSteps${st.maxSteps}`);
      const suffix = diffs.length ? ' · ' + diffs.join(' ') : '';
      const run: ComparisonRun = {
        id: 'run-' + st.runCounter,
        label: autoLabel(st.algoId, st.hyperparams) + suffix,
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
  };
});

// During development, attach the store to window for easier debugging/verification (not present in production builds).
if (import.meta.env.DEV) {
  (window as unknown as { rlstore?: typeof useStore }).rlstore = useStore;
}
