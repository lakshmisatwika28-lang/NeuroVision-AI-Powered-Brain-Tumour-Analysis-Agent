import { AlertTriangle, Crosshair, Layers } from 'lucide-react';
import ConfidenceBar from '../common/ConfidenceBar';
import { ClassBadge } from '../common/Badges';
import './ModelComparisonCard.css';

export default function ModelComparisonCard({ modelComparison }) {
  const { yolo, resnet, agree } = modelComparison;

  return (
    <div className="model-compare panel">
      <div className="model-compare__title">Model comparison</div>

      <div className="model-compare__grid">
        <div className="model-compare__model">
          <div className="model-compare__model-head">
            <Crosshair size={14} />
            YOLO26
            <span className="model-compare__role">Detection</span>
          </div>
          <ClassBadge name={yolo.class} />
          <ConfidenceBar value={yolo.confidence} color="var(--accent)" />
        </div>

        <div className="model-compare__divider" />

        <div className="model-compare__model">
          <div className="model-compare__model-head">
            <Layers size={14} />
            ResNet50
            <span className="model-compare__role">Classification</span>
          </div>
          <ClassBadge name={resnet.class} />
          <ConfidenceBar value={resnet.confidence} color="var(--accent-2)" />
        </div>
      </div>

      {!agree && (
        <div className="model-compare__warning">
          <AlertTriangle size={15} />
          <div>
            <strong>Model disagreement</strong>
            <p>
              YOLO26: {yolo.class} &nbsp;·&nbsp; ResNet50: {resnet.class}
            </p>
            <p className="model-compare__warning-note">
              Independent model predictions differ. This result should be reviewed as part of the
              research workflow rather than treated as a single confirmed output.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
