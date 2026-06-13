// Serialize / restore model weights as plain arrays, so trained weights can be
// postMessage'd from the training Web Worker back to the main thread (structured-clone safe).
import * as tf from '@tensorflow/tfjs';

export interface WeightDump {
  data: number[][]; // flattened values, one entry per weight tensor
  shapes: number[][]; // matching shape per weight tensor
}

export function dumpWeights(model: tf.LayersModel): WeightDump {
  const ws = model.getWeights();
  return {
    data: ws.map((w) => Array.from(w.dataSync())),
    shapes: ws.map((w) => w.shape.slice()),
  };
}

export function loadWeights(model: tf.LayersModel, dump: WeightDump): void {
  const tensors = dump.data.map((arr, i) => tf.tensor(arr, dump.shapes[i]));
  model.setWeights(tensors); // copies values into the model's variables
  tensors.forEach((t) => t.dispose());
}
