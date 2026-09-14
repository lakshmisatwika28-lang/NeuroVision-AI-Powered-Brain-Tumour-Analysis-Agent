import { RiskBadge } from '../common/Badges';
import Disclaimer from '../common/Disclaimer';
import './RiskAnalysis.css';

export default function RiskAnalysis({ data }) {
  return (
    <div className="risk-panel panel">
      <div className="risk-panel__header">
        <span className="eyebrow">RESEARCH-DEFINED RISK STRATIFICATION</span>
      </div>

      <div className="risk-panel__main">
        <div>
          <div className="risk-panel__label">Risk level</div>
          <RiskBadge level={data.riskLevel} />
        </div>
        <div className="risk-panel__score">
          <div className="risk-panel__label">Risk score</div>
          <div className="risk-panel__score-value mono">{data.riskScore}</div>
        </div>
      </div>

      <div className="risk-panel__factors">
        <div className="risk-panel__label">Contributing factors</div>
        <ul>
          {data.factors.map((f) => (
            <li key={f}>{f}</li>
          ))}
        </ul>
      </div>

      <Disclaimer>{data.disclaimer}</Disclaimer>
    </div>
  );
}
