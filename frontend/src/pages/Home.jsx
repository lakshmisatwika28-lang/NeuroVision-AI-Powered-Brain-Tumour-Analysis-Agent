import { useEffect, useRef, useState } from 'react';
import { BrainCircuit } from 'lucide-react';
import MRIUpload from '../components/Upload/MRIUpload';
import ChatMessage from '../components/Chat/ChatMessage';
import AnalysisProgress from '../components/Chat/AnalysisProgress';
import ResultsDashboard from '../components/Results/ResultsDashboard';
import ChatInput from '../components/Chat/ChatInput';
import PromptChips from '../components/Chat/PromptChips';
import Disclaimer from '../components/common/Disclaimer';
import { analyzeMRI, askAssistant } from '../api/neurovisionApi';
import './Home.css';

export default function Home({ initialEntry, onAnalysisComplete }) {
  const [phase, setPhase] = useState(initialEntry ? 'result' : 'empty');
  const [completedStages, setCompletedStages] = useState([]);
  const [result, setResult] = useState(initialEntry?.result || null);

  const [analysisId, setAnalysisId] = useState(
    initialEntry?.id ||
    initialEntry?.result?.analysisId ||
    initialEntry?.result?.id ||
    null
  );

  const [qa, setQa] = useState([]);
  const [asking, setAsking] = useState(false);

  const scrollRef = useRef();

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: 'smooth',
    });
  }, [phase, completedStages, qa, result]);

  const handleAnalyze = async (file) => {
    setPhase('analyzing');
    setCompletedStages([]);
    setQa([]);
    setAnalysisId(null);

    try {
      const res = await analyzeMRI(file, (stageKey) => {
        setCompletedStages((prev) => {
          if (prev.includes(stageKey)) return prev;
          return [...prev, stageKey];
        });
      });

      console.log('NeuroVision analysis response:', res);

      setResult(res);

      // Backend analysis ID
      let id =
        res?.analysisId ||
        res?.id ||
        null;

      // Save to frontend history
      if (onAnalysisComplete) {
        const saved = onAnalysisComplete(res);

        if (saved?.id) {
          id = saved.id;
        }
      }

      setAnalysisId(id);

      console.log('Analysis ID:', id);

      setPhase('result');
    } catch (err) {
      console.error('Analysis failed:', err);

      setPhase('empty');

      // eslint-disable-next-line no-alert
      alert(`Analysis failed: ${err.message}`);
    }
  };

  const handleAsk = async (question) => {
    if (!question.trim() || !result) return;

    setQa((prev) => [
      ...prev,
      {
        role: 'user',
        text: question,
      },
    ]);

    setAsking(true);

    try {
      console.log('AI question:', question);
      console.log('AI context:', result);

      const answer = await askAssistant(question, result);

      setQa((prev) => [
        ...prev,
        {
          role: 'ai',
          text: answer,
        },
      ]);
    } catch (err) {
      console.error('AI assistant error:', err);

      setQa((prev) => [
        ...prev,
        {
          role: 'ai',
          text: `Sorry, I couldn't answer that question. ${err.message}`,
        },
      ]);
    } finally {
      setAsking(false);
    }
  };

  return (
    <div className="home">
      <div className="home__scroll scrollY" ref={scrollRef}>
        <div className="home__inner">

          {phase === 'empty' && (
            <div className="home__welcome fade-in">
              <div className="home__welcome-icon">
                <BrainCircuit size={30} />
              </div>

              <h1>Welcome to NeuroVision</h1>

              <p>
                Upload a brain MRI to begin an AI-assisted research analysis.
              </p>

              <MRIUpload onAnalyze={handleAnalyze} />

              <Disclaimer className="home__disclaimer">
                NeuroVision is a research and educational prototype. Results
                are not a medical diagnosis and must not be used for clinical
                decision-making.
              </Disclaimer>
            </div>
          )}

          {phase !== 'empty' && (
            <div className="home__conversation">

              <ChatMessage role="user">
                Analyze this MRI
              </ChatMessage>

              {phase === 'analyzing' && (
                <ChatMessage role="ai">
                  <AnalysisProgress
                    completedStages={completedStages}
                  />
                </ChatMessage>
              )}

              {phase === 'result' && result && (
                <ChatMessage role="ai">
                  <ResultsDashboard
                    result={result}
                    analysisId={analysisId}
                  />
                </ChatMessage>
              )}

              {phase === 'result' &&
                qa.map((msg, i) => (
                  <ChatMessage
                    role={msg.role}
                    key={`${msg.role}-${i}`}
                  >
                    {msg.text}
                  </ChatMessage>
                ))}

              {phase === 'result' && asking && (
                <ChatMessage role="ai">
                  <span className="home__typing">
                    Thinking…
                  </span>
                </ChatMessage>
              )}

            </div>
          )}

        </div>
      </div>

      {phase === 'result' && (
        <div className="home__composer">

          {qa.length === 0 && (
            <PromptChips onSelect={handleAsk} />
          )}

          <ChatInput
            onSend={handleAsk}
            disabled={asking}
            placeholder="Ask the NeuroVision assistant about this analysis…"
          />

        </div>
      )}
    </div>
  );
}