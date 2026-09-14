import { useState, useCallback } from 'react';
import { Routes, Route } from 'react-router-dom';
import IntroAnimation from './components/Intro/IntroAnimation';
import AppLayout from './components/Layout/AppLayout';
import Home from './pages/Home';
import HistoryPage from './pages/HistoryPage';
import ResearchPage from './pages/ResearchPage';
import ModelInsightsPage from './pages/ModelInsightsPage';
import AboutPage from './pages/AboutPage';
import { useAnalysisHistory } from './hooks/useAnalysisHistory';
import { hasSeenIntro, markIntroSeen } from './utils/storage';

export default function App() {
  const [introDone, setIntroDone] = useState(hasSeenIntro());
  const { history, addEntry, removeEntry, clearAll } = useAnalysisHistory();

  const [activeEntry, setActiveEntry] = useState(null);
  const [sessionKey, setSessionKey] = useState('initial');

  const handleIntroComplete = useCallback(() => {
    markIntroSeen();
    setIntroDone(true);
  }, []);

  const handleNewAnalysis = useCallback(() => {
    setActiveEntry(null);
    setSessionKey(`new-${Date.now()}`);
  }, []);

  const handleSelectHistory = useCallback((entry) => {
    setActiveEntry(entry);
    setSessionKey(`hist-${entry.id}`);
  }, []);

  const handleAnalysisComplete = useCallback(
    (result) => {
      const entry = addEntry(result);
      setActiveEntry(entry);
    },
    [addEntry]
  );

  const handleDeleteHistory = useCallback(() => {
    const confirmed = window.confirm(
      'Delete all locally stored analysis history? This cannot be undone.'
    );
    if (confirmed) {
      clearAll();
      setActiveEntry(null);
      setSessionKey(`new-${Date.now()}`);
    }
  }, [clearAll]);

  if (!introDone) {
    return <IntroAnimation onComplete={handleIntroComplete} />;
  }

  return (
    <AppLayout
      history={history}
      onNewAnalysis={handleNewAnalysis}
      onSelectHistory={handleSelectHistory}
      onDeleteHistory={handleDeleteHistory}
      activeId={activeEntry?.id}
    >
      <Routes>
        <Route
          path="/"
          element={
            <Home
              key={sessionKey}
              initialEntry={activeEntry}
              onAnalysisComplete={handleAnalysisComplete}
            />
          }
        />
        <Route
          path="/history"
          element={
            <HistoryPage
              history={history}
              onSelect={handleSelectHistory}
              onDelete={removeEntry}
              onClearAll={handleDeleteHistory}
            />
          }
        />
        <Route path="/research" element={<ResearchPage />} />
        <Route path="/model-insights" element={<ModelInsightsPage />} />
        <Route path="/about" element={<AboutPage />} />
      </Routes>
    </AppLayout>
  );
}
