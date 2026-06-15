import { useStore } from '../../state/store';
import { ALGO_REGISTRY, algoSupportsEnv } from '../../algos/registry';
import { getEnvEntry } from '../../envs/registry';

export function AlgoSelector() {
  const algoId = useStore((s) => s.algoId);
  const envId = useStore((s) => s.envId);
  const setAlgo = useStore((s) => s.setAlgo);
  // Only show algorithms compatible with the current environment (e.g. continuous-action
  // Pendulum → PPO only; discrete-state grids → tabular only).
  const env = getEnvEntry(envId);
  const tabular = ALGO_REGISTRY.filter((a) => !a.deep && algoSupportsEnv(a, env));
  const deep = ALGO_REGISTRY.filter((a) => a.deep && algoSupportsEnv(a, env));
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
