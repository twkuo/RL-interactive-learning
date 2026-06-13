import { useStore } from '../../state/store';
import { ALGO_REGISTRY } from '../../algos/registry';

export function AlgoSelector() {
  const algoId = useStore((s) => s.algoId);
  const setAlgo = useStore((s) => s.setAlgo);
  return (
    <div className="control">
      <label>Algorithm</label>
      <select value={algoId} onChange={(e) => setAlgo(e.target.value)}>
        {ALGO_REGISTRY.map((a) => (
          <option key={a.id} value={a.id}>
            {a.name}
          </option>
        ))}
      </select>
    </div>
  );
}
