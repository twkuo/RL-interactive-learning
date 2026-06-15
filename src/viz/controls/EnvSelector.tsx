import { useStore } from '../../state/store';
import { ENV_REGISTRY } from '../../envs/registry';
import { algoSupportsEnv, getAlgoEntry } from '../../algos/registry';

export function EnvSelector() {
  const envId = useStore((s) => s.envId);
  const algoId = useStore((s) => s.algoId);
  const setEnv = useStore((s) => s.setEnv);
  // Only show environments compatible with the current algorithm (observation encoding + action kind).
  const algo = getAlgoEntry(algoId);
  const envs = ENV_REGISTRY.filter((e) => algoSupportsEnv(algo, e));
  return (
    <div className="control">
      <label>Environment</label>
      <select value={envId} onChange={(e) => setEnv(e.target.value)}>
        {envs.map((e) => (
          <option key={e.id} value={e.id}>
            {e.name}
          </option>
        ))}
      </select>
    </div>
  );
}
