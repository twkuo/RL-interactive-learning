// Message protocol between the main thread and the training Web Worker.
// Structured-clone safe (plain objects / arrays only).
import type { WeightDump } from '../core/nn/weights';

export interface StartMsg {
  type: 'start';
  algoId: string; // 'dqn' (later: 'ppo')
  envId: string; // e.g. 'cartpole-vec'
  hyperparams: Record<string, number>;
  rewardParams: Array<{ key: string; value: number }>;
  maxSteps: number;
  episodes: number;
  seed: number;
  fresh: boolean; // true = start a new network; false = continue training the worker's current agent
}

export interface CancelMsg {
  type: 'cancel';
}

export type ToWorker = StartMsg | CancelMsg;

export interface ProgressMsg {
  type: 'progress';
  episode: number;
  ret: number;
  avgReturn: number; // moving average over recent episodes
  metrics: Record<string, number>; // loss, tdError, epsilon, bufferFill, trainSteps, ...
}

export interface DoneMsg {
  type: 'done';
  returns: number[];
  weights: WeightDump; // BEST weights found during training (not necessarily the final ones)
  epsilon: number;
  bestAvg: number; // best moving-average return achieved (what the returned weights scored)
}

export interface ErrorMsg {
  type: 'error';
  message: string;
}

export type FromWorker = ProgressMsg | DoneMsg | ErrorMsg;
