import { useStore } from '../../state/store';
import { ALGO_REGISTRY } from '../../algos/registry';

export function AlgoSelector() {
  const algoId = useStore((s) => s.algoId);
  const setAlgo = useStore((s) => s.setAlgo);
  const tabular = ALGO_REGISTRY.filter((a) => !a.deep);
  const deep = ALGO_REGISTRY.filter((a) => a.deep);
  return (
    <div className="control">
      <label>Algorithm</label>
      <select value={algoId} onChange={(e) => setAlgo(e.target.value)}>
        <optgroup label="Tabular">
          {tabular.map((a) => (
            <option key={a.id} value={a.id}>
              {a.name}
            </option>
          ))}
        </optgroup>
        <optgroup label="Deep RL (neural network)">
          {deep.map((a) => (
            <option key={a.id} value={a.id}>
              {a.name}
            </option>
          ))}
        </optgroup>
      </select>
    </div>
  );
}
