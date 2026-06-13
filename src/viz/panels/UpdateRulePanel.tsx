// Teaching feature: update rule + live numbers. Step-by-step methods highlight the TD error; MC/REINFORCE update only at episode end.
import katex from 'katex';
import { useStore } from '../../state/store';
import { EPISODIC_FORMULAS, getAlgoEntry, type FormulaKind } from '../../algos/registry';
import type { UpdateInfo } from '../../core/types';
import { DqnDashboard } from './DqnDashboard';

const f = (x: number | undefined) => (x === undefined || !Number.isFinite(x) ? '0.000' : x.toFixed(3));
const RED = '#e06c75';

function Tex({ tex }: { tex: string }) {
  const html = katex.renderToString(tex, { throwOnError: false, displayMode: true });
  return <div className="katex-block" dangerouslySetInnerHTML={{ __html: html }} />;
}

function symbolicOf(formula: FormulaKind): string {
  switch (formula) {
    case 'q-learning':
      return String.raw`Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma\,\max_{a'}Q(s',a') - Q(s,a)\,]`;
    case 'sarsa':
      return String.raw`Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma\,Q(s',a') - Q(s,a)\,]`;
    case 'expected-sarsa':
      return String.raw`Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma\,\mathbb{E}_{a'\sim\pi}[Q(s',a')] - Q(s,a)\,]`;
    case 'double-q':
      return String.raw`Q_A(s,a) \leftarrow Q_A(s,a) + \alpha\,[\,r + \gamma\,Q_B(s',\arg\max_{a'}Q_A(s',a')) - Q_A(s,a)\,]`;
    case 'mc':
      return String.raw`Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,G_t - Q(s,a)\,]\quad(\text{first-visit})`;
    case 'reinforce':
      return String.raw`\theta(s,a) \leftarrow \theta(s,a) + \alpha\,\gamma^t\,G_t\,(\mathbf{1}[a=a_t] - \pi(a\mid s))`;
    case 'dqn':
      return String.raw`\mathcal{L}(\theta) = \text{Huber}\!\big(y - Q_\theta(s,a)\big),\ \ y = r + \gamma\,Q_{\theta^-}\!\big(s',\,\arg\max_{a'}Q_\theta(s',a')\big)`;
  }
}

function numericLines(formula: FormulaKind, i: UpdateInfo): string[] {
  const r = f(i.reward);
  const g = f(i.gamma);
  const a = f(i.alpha);
  const td = f(i.tdError);
  const qSA = f(i.qSA);
  if (formula === 'double-q') {
    const tbl = i.updatedTable === 1 ? 'B' : 'A';
    const oth = i.updatedTable === 1 ? 'A' : 'B';
    return [
      String.raw`\delta = (${r} + ${g}\times \underbrace{${f(i.otherNext)}}_{Q_${oth}(s',a^*)}) - ${qSA} = \textcolor{${RED}}{${td}}`,
      String.raw`Q_${tbl}(s,a) \leftarrow ${qSA} + ${a}\times \textcolor{${RED}}{${td}} = \mathbf{${f(i.newQ)}}`,
    ];
  }
  let boot: string;
  let bootSym: string;
  if (formula === 'q-learning') {
    boot = f(i.maxQNext);
    bootSym = String.raw`\max_{a'}Q(s',a')`;
  } else if (formula === 'expected-sarsa') {
    boot = f(i.expectedQNext);
    bootSym = String.raw`\mathbb{E}_{a'\sim\pi}[Q(s',a')]`;
  } else {
    boot = f(i.qNextSA);
    bootSym = String.raw`Q(s',a')`;
  }
  return [
    String.raw`\delta = (${r} + ${g}\times \underbrace{${boot}}_{${bootSym}}) - ${qSA} = \textcolor{${RED}}{${td}}`,
    String.raw`Q(s,a) \leftarrow ${qSA} + ${a}\times \textcolor{${RED}}{${td}} = \mathbf{${f(i.newQ)}}`,
  ];
}

export function UpdateRulePanel() {
  const algoId = useStore((s) => s.algoId);
  const lastUpdate = useStore((s) => s.lastUpdate);
  const lastEpisodeUpdate = useStore((s) => s.lastEpisodeUpdate);
  const phase = useStore((s) => s.phase);
  const lastStep = useStore((s) => s.lastStep);
  const stepInEpisode = useStore((s) => s.stepInEpisode);
  const entry = getAlgoEntry(algoId);
  const formula = entry.formula;
  const episodic = EPISODIC_FORMULAS.includes(formula);

  // Deep RL has no hand-derivable per-step update — show the training dashboard instead.
  if (entry.deep) return <DqnDashboard />;

  const justEnded =
    phase === 'result' && !!lastStep && (lastStep.terminated || lastStep.truncated) && !!lastEpisodeUpdate;

  return (
    <div className="panel">
      <div className="panel-title">Value Update Rule (live numbers)</div>
      <Tex tex={symbolicOf(formula)} />

      {episodic ? (
        justEnded && lastEpisodeUpdate ? (
          <div className="verdict ok">
            Episode finished! Episode return G₀ = {f(lastEpisodeUpdate.finalG)};
            {formula === 'mc'
              ? `updated ${lastEpisodeUpdate.numUpdates ?? 0} first-visit (s,a) pairs.${
                  lastEpisodeUpdate.bootstrapped
                    ? `Time-limit truncation → tail bootstrapped with V(s_T)=${f(lastEpisodeUpdate.tail)}.`
                    : 'Natural termination (pure MC, no bootstrap).'
                }`
              : `policy-gradient update over ${lastEpisodeUpdate.numUpdates ?? 0} steps (all actions each step).`}
          </div>
        ) : (
          <div className="hint">Recording this episode's trajectory… ({stepInEpisode} steps so far); updates at episode end.</div>
        )
      ) : lastUpdate ? (
        <>
          {numericLines(formula, lastUpdate).map((line, idx) => (
            <Tex key={idx} tex={line} />
          ))}
          <div className="hint">
            δ is the TD error (red).
            {formula === 'double-q' && 'This step randomly updates one table and evaluates with the other.'}
            {lastUpdate.terminated === 1 && 'This step terminated → successor value (bootstrap term) zeroed.'}
          </div>
        </>
      ) : (
        <div className="hint">After a step, the real s, a, r, s′ values are plugged into the formula here.</div>
      )}
    </div>
  );
}
