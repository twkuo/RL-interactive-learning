import { useEffect } from 'react';
import { useStore } from '../state/store';
import { EnvSelector } from '../viz/controls/EnvSelector';
import { AlgoSelector } from '../viz/controls/AlgoSelector';
import { HyperParams } from '../viz/controls/HyperParams';
import { RunControls } from '../viz/controls/RunControls';
import { RewardPanel } from '../viz/controls/RewardPanel';
import { EnvView } from '../viz/renderers/EnvView';
import { ActionRewardPanel } from '../viz/panels/ActionRewardPanel';
import { StateInspector } from '../viz/panels/StateInspector';
import { ContinuousStatePanel } from '../viz/panels/ContinuousStatePanel';
import { DecisionPanel } from '../viz/panels/DecisionPanel';
import { UpdateRulePanel } from '../viz/panels/UpdateRulePanel';
import { RewardChart } from '../viz/charts/RewardChart';
import { ComparisonControls } from '../viz/controls/ComparisonControls';

export function Playground() {
  const isPlaying = useStore((s) => s.isPlaying);
  const speed = useStore((s) => s.speed);

  // Auto-play: drive full steps on a timer (use getState to avoid a stale closure).
  useEffect(() => {
    if (!isPlaying) return;
    const id = setInterval(() => useStore.getState().quickStep(), speed);
    return () => clearInterval(id);
  }, [isPlaying, speed]);

  return (
    <div className="app">
      <header className="appbar">
        <h1>Interactive RL Playground</h1>
        <span className="subtitle">See every step: state · action · reward · V · Q · decisions</span>
      </header>

      <div className="layout">
        <aside className="col-left">
          <RunControls />
          <div className="panel">
            <div className="panel-title">Settings</div>
            <EnvSelector />
            <AlgoSelector />
          </div>
          <RewardPanel />
        </aside>

        <main className="col-center">
          <div className="canvas-wrap">
            <EnvView />
          </div>
          <StateInspector />
          <ContinuousStatePanel />
          <ActionRewardPanel />
        </main>

        <aside className="col-right">
          <DecisionPanel />
          <UpdateRulePanel />
          <HyperParams />
        </aside>
      </div>

      <div className="bottom">
        <ComparisonControls />
        <RewardChart />
      </div>
    </div>
  );
}
