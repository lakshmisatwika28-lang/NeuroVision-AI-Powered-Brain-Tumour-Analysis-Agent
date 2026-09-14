import { riskColor } from '../../utils/formatters';

export function RiskBadge({ level }) {
  const color = riskColor(level);
  return (
    <span
      className="risk-badge"
      style={{ color, borderColor: color, background: `${color}1a` }}
    >
      <span className="risk-badge__dot" style={{ background: color }} />
      {level?.toUpperCase() || 'UNKNOWN'}
    </span>
  );
}

export function StatusDot({ label = 'System Ready', active = true }) {
  return (
    <span className="status-dot">
      <span className={`status-dot__led ${active ? 'status-dot__led--on' : ''}`} />
      {label}
    </span>
  );
}

export function ClassBadge({ name }) {
  return <span className="class-badge">{name}</span>;
}
