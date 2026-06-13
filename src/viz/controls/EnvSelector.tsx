import { useStore } from '../../state/store';
import { ENV_REGISTRY } from '../../envs/registry';

export function EnvSelector() {
  const envId = useStore((s) => s.envId);
  const setEnv = useStore((s) => s.setEnv);
  return (
    <div className="control">
      <label>Environment</label>
      <select value={envId} onChange={(e) => setEnv(e.target.value)}>
        {ENV_REGISTRY.map((e) => (
          <option key={e.id} value={e.id}>
            {e.name}
          </option>
        ))}
      </select>
    </div>
  );
}
