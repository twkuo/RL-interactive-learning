// Switch renderer based on the environment's renderKind.
import { useStore } from '../../state/store';
import { getEnvEntry } from '../../envs/registry';
import { GridRenderer } from './GridRenderer';
import { CartPoleRenderer } from './CartPoleRenderer';
import { MountainCarRenderer } from './MountainCarRenderer';
import { AcrobotRenderer } from './AcrobotRenderer';
import { PendulumRenderer } from './PendulumRenderer';
import { LunarLanderRenderer } from './LunarLanderRenderer';
import { BanditRenderer } from './BanditRenderer';

export function EnvView() {
  const envId = useStore((s) => s.envId);
  const kind = getEnvEntry(envId).renderKind;
  if (kind === 'bandit') return <BanditRenderer />;
  if (kind === 'cartpole') return <CartPoleRenderer />;
  if (kind === 'mountaincar') return <MountainCarRenderer />;
  if (kind === 'acrobot') return <AcrobotRenderer />;
  if (kind === 'pendulum') return <PendulumRenderer />;
  if (kind === 'lunarlander') return <LunarLanderRenderer />;
  return <GridRenderer />;
}
