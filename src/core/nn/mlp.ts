// Small multilayer-perceptron builder (TensorFlow.js). Used for DQN Q-networks and
// (later) PPO actor/critic heads. Kept tiny so training runs on a regular CPU.
import * as tf from '@tensorflow/tfjs';

export interface MLPSpec {
  inputDim: number;
  hidden: number[]; // at least one hidden layer
  outputDim: number;
  outputActivation?: 'linear' | 'softmax';
}

export function buildMLP(spec: MLPSpec): tf.Sequential {
  const model = tf.sequential();
  model.add(
    tf.layers.dense({ inputShape: [spec.inputDim], units: spec.hidden[0], activation: 'relu' }),
  );
  for (let i = 1; i < spec.hidden.length; i++) {
    model.add(tf.layers.dense({ units: spec.hidden[i], activation: 'relu' }));
  }
  model.add(tf.layers.dense({ units: spec.outputDim, activation: spec.outputActivation ?? 'linear' }));
  return model;
}
