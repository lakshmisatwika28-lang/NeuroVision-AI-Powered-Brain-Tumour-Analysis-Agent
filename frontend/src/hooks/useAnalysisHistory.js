import { useCallback, useEffect, useState } from 'react';
import { addHistoryEntry, clearHistory, deleteHistoryEntry, loadHistory } from '../utils/storage';

export function useAnalysisHistory() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  const addEntry = useCallback((analysisResult, label) => {
    const entry = {
      id: `an-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      timestamp: new Date().toISOString(),
      label: label || analysisResult.finalClass || 'Analysis',
      predictedClass: analysisResult.finalClass,
      yoloConfidence: analysisResult.predictions?.yolo?.confidence ?? null,
      resnetConfidence: analysisResult.predictions?.resnet?.confidence ?? null,
      tumorCharacteristics: analysisResult.tumorCharacteristics,
      riskAnalysis: analysisResult.riskAnalysis,
      modelComparison: analysisResult.modelComparison,
      gradcamAvailable: !!analysisResult.gradcam?.available,
      imageName: analysisResult.image?.name,
      result: analysisResult,
    };
    setHistory(addHistoryEntry(entry));
    return entry;
  }, []);

  const removeEntry = useCallback((id) => {
    setHistory(deleteHistoryEntry(id));
  }, []);

  const clearAll = useCallback(() => {
    setHistory(clearHistory());
  }, []);

  return { history, addEntry, removeEntry, clearAll };
}
