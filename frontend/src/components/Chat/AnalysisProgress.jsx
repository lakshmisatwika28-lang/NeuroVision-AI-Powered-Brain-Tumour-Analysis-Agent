import { Check, Loader2 } from 'lucide-react';
import { PIPELINE_STAGES } from '../../api/neurovisionApi';
import './AnalysisProgress.css';

export default function AnalysisProgress({ completedStages }) {
  return (
    <div className="analysis-progress panel">
      <div className="analysis-progress__header">
        <span className="analysis-progress__pulse" />
        NeuroVision is analyzing the uploaded MRI…
      </div>
      <ul className="analysis-progress__list">
        {PIPELINE_STAGES.map((stage, i) => {
          const done = completedStages.includes(stage.key);
          const isCurrent = !done && (i === 0 || completedStages.includes(PIPELINE_STAGES[i - 1].key));
          return (
            <li key={stage.key} className={done ? 'is-done' : isCurrent ? 'is-active' : ''}>
              <span className="analysis-progress__icon">
                {done ? <Check size={13} /> : isCurrent ? <Loader2 size={13} className="spin" /> : null}
              </span>
              {stage.label}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
