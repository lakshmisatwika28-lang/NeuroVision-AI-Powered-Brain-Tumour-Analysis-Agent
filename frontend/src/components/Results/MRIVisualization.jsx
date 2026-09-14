import { useState } from 'react';
import './MRIVisualization.css';

const TABS = [
  { key: 'original', label: 'Original MRI' },
  { key: 'yolo', label: 'YOLO Detection' },
  { key: 'gradcam', label: 'Grad-CAM' },
];

export default function MRIVisualization({ imageUrl, detections = [] }) {
  const [tab, setTab] = useState('original');

  return (
    <div className="mri-viz panel">
      <div className="mri-viz__tabs">
        {TABS.map((t) => (
          <button
            key={t.key}
            className={`mri-viz__tab ${tab === t.key ? 'mri-viz__tab--active' : ''}`}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="mri-viz__stage">
        <img src={imageUrl} alt="Brain MRI scan" className="mri-viz__image" />

        {tab === 'yolo' && (
          <div className="mri-viz__overlay fade-in">
            {detections.map((det) => (
              <div
                key={det.id}
                className="mri-viz__box"
                style={{
                  left: `${det.box.x * 100}%`,
                  top: `${det.box.y * 100}%`,
                  width: `${det.box.width * 100}%`,
                  height: `${det.box.height * 100}%`,
                }}
              >
                <span className="mri-viz__box-label">
                  {det.class} · {det.confidence.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        )}

        {tab === 'gradcam' && (
          <div className="mri-viz__overlay fade-in">
            {detections.map((det) => (
              <div
                key={det.id}
                className="mri-viz__heat"
                style={{
                  left: `${(det.box.x + det.box.width / 2) * 100}%`,
                  top: `${(det.box.y + det.box.height / 2) * 100}%`,
                  width: `${det.box.width * 220}%`,
                  height: `${det.box.height * 220}%`,
                }}
              />
            ))}
          </div>
        )}
      </div>

      {detections.length > 1 && (
        <div className="mri-viz__multi-note">
          {detections.length} tumor regions detected — see individual cards below.
        </div>
      )}
    </div>
  );
}
