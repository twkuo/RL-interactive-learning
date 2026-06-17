import { useStore } from '../../state/store';
import { ALGO_REGISTRY } from '../../algos/registry';

export function AlgoSelector() {
  const algoId = useStore((s) => s.algoId);
  const setAlgo = useStore((s) => s.setAlgo);
  // Show ALL algorithms (tabular + deep), always. Picking one that needs a different observation
  // or action kind than the current environment makes the store switch to a compatible env
  // (setAlgo -> compatibleEnvId), so you can always cross between the tabular and deep-RL worlds.
  // (The env dropdown stays filtered by the current algo, which keeps it free of duplicates.)
  const tabular = ALGO_REGISTRY.filter((a) => !a.deep);
  const deep = ALGO_REGISTRY.filter((a) => a.deep);
  return (
    <div className="control">
      <label>Algorithm</label>
      <select value={algoId} onChange={(e) => setAlgo(e.target.value)}>
        {tabular.length > 0 && (
          <optgroup label="Tabular">
            {tabular.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name}
              </option>
            ))}
          </optgroup>
        )}
        {deep.length > 0 && (
          <optgroup label="Deep RL (neural network)">
            {deep.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name}
              </option>
            ))}
          </optgroup>
        )}
      </select>
    </div>
  );
}
