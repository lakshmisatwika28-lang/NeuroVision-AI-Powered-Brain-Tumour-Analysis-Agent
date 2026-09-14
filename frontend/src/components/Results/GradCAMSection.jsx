import ConfidenceBar from '../common/ConfidenceBar';
import Disclaimer from '../common/Disclaimer';
import { ClassBadge } from '../common/Badges';
import './GradCAMSection.css';

export default function GradCAMSection({ imageUrl, detection, gradcam }) {
  const box = detection?.box;
  const bgPos = box
    ? `${(box.x + box.width / 2) * 100}% ${(box.y + box.height / 2) * 100}%`
    : '50% 50%';
  const zoom = box ? Math.max(180, 100 / Math.max(box.width, box.height)) : 220;

  const cropStyle = {
    backgroundImage: `url(${imageUrl})`,
    backgroundSize: `${zoom}%`,
    backgroundPosition: bgPos,
  };

  return (
    <div className="gradcam panel">
      <div className="gradcam__header">
        <span className="eyebrow">MODEL EXPLAINABILITY</span>
        <h3>Why did ResNet50 make this prediction?</h3>
      </div>

      <div className="gradcam__images">
        <div className="gradcam__frame">
          <div className="gradcam__crop" style={cropStyle} />
          <span className="gradcam__frame-label">Original tumor crop</span>
        </div>
        <div className="gradcam__frame">
          <div className="gradcam__crop" style={cropStyle} />
          <div className="gradcam__heat-full" />
          <span className="gradcam__frame-label">Grad-CAM overlay</span>
        </div>
      </div>

      <div className="gradcam__result">
        <div>
          <div className="gradcam__result-label">Predicted class</div>
          <ClassBadge name={gradcam.predictedClass} />
        </div>
        <div className="gradcam__result-conf">
          <ConfidenceBar value={gradcam.confidence} label="Confidence" color="var(--accent-2)" />
        </div>
      </div>

      <Disclaimer>
        Grad-CAM highlights image regions that contributed to the model's prediction. This
        visualization is intended for model interpretability and does not establish clinical
        significance.
      </Disclaimer>
    </div>
  );
}
