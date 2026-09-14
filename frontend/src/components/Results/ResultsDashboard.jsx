import ResultSummaryCard from './ResultSummaryCard';
import ModelComparisonCard from './ModelComparisonCard';
import MRIVisualization from './MRIVisualization';
import TumorCharacteristics from './TumorCharacteristics';
import RiskAnalysis from './RiskAnalysis';
import GradCAMSection from './GradCAMSection';
import ReportButton from './ReportButton';
import './ResultsDashboard.css';

export default function ResultsDashboard({ result, analysisId }) {
  const finalAnalysisId = analysisId || result?.analysisId || result?.id;

  const firstDetection =
    result?.detections && result.detections.length > 0
      ? result.detections[0]
      : null;

  return (
    <div className="results-dashboard">
      <ResultSummaryCard result={result} />

      <ModelComparisonCard
        modelComparison={result.modelComparison}
      />

      <MRIVisualization
        imageUrl={result.image?.previewUrl}
        detections={result.detections || []}
      />

      <div className="results-dashboard__row">
        <TumorCharacteristics
          data={result.tumorCharacteristics}
        />

        <RiskAnalysis
          data={result.riskAnalysis}
        />
      </div>

      {firstDetection && (
        <GradCAMSection
          imageUrl={result.image?.previewUrl}
          detection={firstDetection}
          gradcam={result.gradcam}
        />
      )}

      <div className="results-dashboard__footer">
        <ReportButton analysisId={finalAnalysisId} />
      </div>
    </div>
  );
}