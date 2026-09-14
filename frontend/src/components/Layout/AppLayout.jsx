import { useState } from 'react';
import { Menu } from 'lucide-react';
import Sidebar from '../Sidebar/Sidebar';
import { StatusDot } from '../common/Badges';
import './AppLayout.css';

export default function AppLayout({
  children,
  history,
  onNewAnalysis,
  onSelectHistory,
  onDeleteHistory,
  activeId,
  headerTitle,
  headerSubtitle,
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="app-layout">
      <Sidebar
        history={history}
        onNewAnalysis={onNewAnalysis}
        onSelectHistory={onSelectHistory}
        onDeleteHistory={onDeleteHistory}
        activeId={activeId}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="app-layout__main">
        <header className="app-header">
          <button className="app-header__menu" onClick={() => setSidebarOpen(true)} aria-label="Open menu">
            <Menu size={20} />
          </button>
          <div className="app-header__title">
            <span className="app-header__name">{headerTitle || 'NeuroVision'}</span>
            <span className="app-header__sub">{headerSubtitle || 'Research AI Assistant'}</span>
          </div>
          <div className="app-header__status">
            <StatusDot />
          </div>
        </header>

        <main className="app-layout__content">{children}</main>
      </div>
    </div>
  );
}
