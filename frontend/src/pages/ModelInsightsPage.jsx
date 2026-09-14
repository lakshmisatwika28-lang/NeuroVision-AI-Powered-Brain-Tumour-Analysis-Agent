import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Crosshair, Layers, Flame, HelpCircle } from 'lucide-react';
import Disclaimer from '../components/common/Disclaimer';
import '../styles/page.css';
import './ModelInsightsPage.css';

const CLASSIFIER_ACCURACY = [
  { name: 'Custom CNN', value: 85.31 },
  { name: 'ResNet50', value: 95.83 },
];

const YOLO_METRICS = [
  { name: 'mAP@50', value: 92.03 },
  { name: 'mAP@50–95', value: 70.95 },
];

export default function ModelInsightsPage() {
  return (
    <div className="page">
      <div className="page__inner">
        <div className="page__header">
          <h1>Model Insights</h1>
          <p>How NeuroVision's detection and classification models work, and how they perform.</p>
        </div>

        <Disclaimer>
          All figures below are results on our held-out research dataset. They describe model
          performance in a research setting, not real-world clinical performance.
        </Disclaimer>

        <section className="model-card panel">
          <div className="model-card__head">
            <Crosshair size={18} />
            <div>
              <h2>YOLO26 — tumor detection &amp; localization</h2>
              <p>Scans the full MRI, draws a bounding box around any suspected tumor, and gives an initial class guess with a confidence score.</p>
            </div>
          </div>
          <div className="model-card__chart">
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={YOLO_METRICS} layout="vertical" margin={{ left: 10, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis type="number" domain={[0, 100]} tick={{ fill: 'var(--text-3)', fontSize: 11 }} stroke="var(--border)" />
                <YAxis type="category" dataKey="name" tick={{ fill: 'var(--text-2)', fontSize: 12 }} stroke="var(--border)" width={90} />
                <Tooltip
                  contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 12 }}
                  formatter={(v) => [`${v}%`, 'Score']}
                />
                <Bar dataKey="value" fill="var(--accent)" radius={[0, 4, 4, 0]} barSize={22} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="model-card__note">Held-out test mAP@50: 92.03% · mAP@50–95: 70.95%</p>
        </section>

        <section className="model-card panel">
          <div className="model-card__head">
            <Layers size={18} />
            <div>
              <h2>ResNet50 — tumor classification</h2>
              <p>An ImageNet-pretrained ResNet50 with the final layer replaced for 3 tumor classes, fine-tuned on tumor-region crops via transfer learning.</p>
            </div>
          </div>
          <div className="model-card__chart">
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={CLASSIFIER_ACCURACY} margin={{ top: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="name" tick={{ fill: 'var(--text-2)', fontSize: 12 }} stroke="var(--border)" />
                <YAxis domain={[0, 100]} tick={{ fill: 'var(--text-3)', fontSize: 11 }} stroke="var(--border)" />
                <Tooltip
                  contentStyle={{ background: 'var(--surface-2)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 12 }}
                  formatter={(v) => [`${v}%`, 'Test accuracy']}
                />
                <Bar dataKey="value" fill="var(--accent-2)" radius={[4, 4, 0, 0]} barSize={48} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="model-card__note">
            Held-out test accuracy: 95.83% (3 classes: Glioma, Meningioma, Pituitary). Evaluation
            also includes precision, recall, F1, and a confusion matrix.
          </p>
        </section>

        <section className="model-card panel">
          <div className="model-card__head">
            <Flame size={18} />
            <div>
              <h2>Grad-CAM — explainability</h2>
              <p>
                Gradient-weighted Class Activation Mapping visualizes which pixels in the tumor
                crop most influenced ResNet50's classification decision, supporting model
                interpretability.
              </p>
            </div>
          </div>
        </section>

        <section className="model-card panel model-card--why">
          <div className="model-card__head">
            <HelpCircle size={18} />
            <h2>Why two models?</h2>
          </div>
          <div className="model-why-grid">
            <div>
              <h3>YOLO26 answers:</h3>
              <p>"Where is the suspected tumor, and what class does the detector predict?"</p>
            </div>
            <div>
              <h3>ResNet50 answers:</h3>
              <p>"What tumor class does the extracted tumor region most resemble?"</p>
            </div>
          </div>
          <p className="model-card__note">
            Running two independently trained models gives a cross-check: when they agree, it
            adds confidence; when they disagree, NeuroVision surfaces that disagreement rather
            than silently picking one.
          </p>
        </section>
      </div>
    </div>
  );
}
