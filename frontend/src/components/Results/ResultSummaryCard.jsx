import { CircleAlert, ScanSearch } from 'lucide-react';
import ConfidenceBar from '../common/ConfidenceBar';
import './ResultSummaryCard.css';

export default function ResultSummaryCard({ result }) {
  const { modelComparison } = result;
  const agree = modelComparison.agree;
  const displayClass = agree ? modelComparison.resnet.class : null;

  return (
    <div className="result-summary panel fade-in">
      <div className="result-summary__eyebrow">
        <ScanSearch size={13} />
        AI ANALYSIS
      </div>

      {agree ? (
        <>
          <div className="result-summary__label">Tumor detected</div>
          <div className="result-summary__class">{displayClass}</div>
          <ConfidenceBar
            value={modelComparison.resnet.confidence}
            label="Combined model confidence (ResNet50 classification)"
            color="var(--accent)"
          />
        </>
      ) : (
        <div className="result-summary__disagree">
          <CircleAlert size={22} />
          <div>
            <div className="result-summary__label">Models disagree on classification</div>
            <p>See the model comparison and full breakdown below before drawing conclusions.</p>
          </div>
        </div>
      )}
    </div>
  );
}
