import { NavLink, useNavigate } from 'react-router-dom';
import {
  Plus,
  History,
  BookOpen,
  BarChart3,
  Info,
  Settings,
  X,
  ScanLine,
  Trash2,
} from 'lucide-react';
import Logo from '../common/Logo';
import { formatRelativeDay } from '../../utils/formatters';
import './Sidebar.css';

export default function Sidebar({
  history,
  onNewAnalysis,
  onSelectHistory,
  onDeleteHistory,
  activeId,
  open,
  onClose,
}) {
  const navigate = useNavigate();

  const grouped = groupByDay(history);

  const handleNew = () => {
    onNewAnalysis();
    navigate('/');
    onClose?.();
  };

  return (
    <>
      {open && <div className="sidebar__scrim" onClick={onClose} />}
      <aside className={`sidebar ${open ? 'sidebar--open' : ''}`}>
        <div className="sidebar__top">
          <div className="sidebar__brand">
            <Logo size={26} />
            <span>NeuroVision</span>
          </div>
          <button className="sidebar__close" onClick={onClose} aria-label="Close sidebar">
            <X size={18} />
          </button>
        </div>

        <button className="sidebar__new" onClick={handleNew}>
          <Plus size={16} />
          New Analysis
        </button>

        <nav className="sidebar__nav">
          <NavLink to="/history" className="sidebar__nav-link" onClick={onClose}>
            <History size={16} /> Analysis History
          </NavLink>
          <NavLink to="/research" className="sidebar__nav-link" onClick={onClose}>
            <BookOpen size={16} /> Research &amp; Prevention
          </NavLink>
          <NavLink to="/model-insights" className="sidebar__nav-link" onClick={onClose}>
            <BarChart3 size={16} /> Model Insights
          </NavLink>
          <NavLink to="/about" className="sidebar__nav-link" onClick={onClose}>
            <Info size={16} /> About NeuroVision
          </NavLink>
        </nav>

        <div className="sidebar__recent scrollY">
          <div className="sidebar__recent-header">
            <span className="eyebrow">RECENT ANALYSES</span>
            {history.length > 0 && (
              <button className="sidebar__clear" onClick={onDeleteHistory} title="Delete all history">
                <Trash2 size={13} />
              </button>
            )}
          </div>

          {history.length === 0 && (
            <p className="sidebar__empty">No analyses yet. Upload an MRI to get started.</p>
          )}

          {Object.entries(grouped).map(([day, entries]) => (
            <div key={day} className="sidebar__group">
              <div className="sidebar__group-label">{day}</div>
              {entries.map((entry) => (
                <button
                  key={entry.id}
                  className={`sidebar__item ${activeId === entry.id ? 'sidebar__item--active' : ''}`}
                  onClick={() => {
                    onSelectHistory(entry);
                    navigate('/');
                    onClose?.();
                  }}
                >
                  <ScanLine size={14} />
                  <span className="sidebar__item-text">
                    {entry.predictedClass ? `${entry.predictedClass} Analysis` : 'Analysis (disagreement)'}
                  </span>
                </button>
              ))}
            </div>
          ))}
        </div>

        <div className="sidebar__bottom">
          <button className="sidebar__nav-link" onClick={() => {}}>
            <Settings size={16} /> Settings
          </button>
          <p className="sidebar__disclaimer">
            Research prototype. Not a medical device. Local history is stored only on this device.
          </p>
        </div>
      </aside>
    </>
  );
}

function groupByDay(history) {
  return history.reduce((acc, entry) => {
    const day = formatRelativeDay(entry.timestamp);
    acc[day] = acc[day] || [];
    acc[day].push(entry);
    return acc;
  }, {});
}
