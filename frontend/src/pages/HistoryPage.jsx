import { useNavigate } from 'react-router-dom';
import { Trash2, ScanLine, AlertTriangle } from 'lucide-react';
import { RiskBadge } from '../components/common/Badges';
import Disclaimer from '../components/common/Disclaimer';
import { formatRelativeDay, formatTime } from '../utils/formatters';
import '../styles/page.css';
import './HistoryPage.css';

export default function HistoryPage({ history, onSelect, onDelete, onClearAll }) {
  const navigate = useNavigate();

  return (
    <div className="page">
      <div className="page__inner">
        <div className="page__header">
          <h1>Analysis History</h1>
          <p>Every analysis you run is stored locally in this browser for this prototype.</p>
        </div>

        <Disclaimer>
          History is stored only in this browser's local storage — it is not uploaded or shared,
          and clearing your browser data will remove it permanently.
        </Disclaimer>

        <div className="page__toolbar">
          <span className="eyebrow">{history.length} SAVED ANALYSES</span>
          {history.length > 0 && (
            <button className="btn btn-ghost" onClick={onClearAll}>
              <Trash2 size={14} /> Delete History
            </button>
          )}
        </div>

        {history.length === 0 && (
          <div className="page__empty panel">
            <ScanLine size={26} />
            <p>No analyses yet. Run your first analysis from the New Analysis screen.</p>
          </div>
        )}

        <div className="history-list">
          {history.map((entry) => (
            <button
              className="history-card panel"
              key={entry.id}
              onClick={() => {
                onSelect(entry);
                navigate('/');
              }}
            >
              <div className="history-card__main">
                <div className="history-card__title">
                  {entry.modelComparison && !entry.modelComparison.agree && (
                    <AlertTriangle size={14} className="history-card__warn" />
                  )}
                  {entry.predictedClass ? `${entry.predictedClass} Analysis` : 'Analysis (model disagreement)'}
                </div>
                <div className="history-card__meta">
                  {formatRelativeDay(entry.timestamp)} · {formatTime(entry.timestamp)} ·{' '}
                  {entry.imageName || 'MRI scan'}
                </div>
              </div>
              <div className="history-card__side">
                {entry.riskAnalysis && <RiskBadge level={entry.riskAnalysis.riskLevel} />}
                <span
                  className="history-card__delete"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(entry.id);
                  }}
                  role="button"
                  aria-label="Delete analysis"
                >
                  <Trash2 size={14} />
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
