// Switch renderer based on the environment's renderKind.
import { useStore } from '../../state/store';
import { getEnvEntry } from '../../envs/registry';
import { GridRenderer } from './GridRenderer';
import { CartPoleRenderer } from './CartPoleRenderer';
import { MountainCarRenderer } from './MountainCarRenderer';

export function EnvView() {
  const envId = useStore((s) => s.envId);
  const kind = getEnvEntry(envId).renderKind;
  if (kind === 'cartpole') return <CartPoleRenderer />;
  if (kind === 'mountaincar') return <MountainCarRenderer />;
  return <GridRenderer />;
}
