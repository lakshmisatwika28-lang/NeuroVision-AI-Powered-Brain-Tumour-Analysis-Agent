import { ArrowDown } from 'lucide-react';
import Logo from '../components/common/Logo';
import '../styles/page.css';
import './AboutPage.css';

const STACK = ['Python', 'PyTorch', 'YOLO26', 'ResNet50', 'Grad-CAM', 'FastAPI', 'React', 'Vite'];

const PIPELINE = [
  'MRI',
  'YOLO26',
  'Tumor localization',
  'Tumor characteristics',
  'ResNet50',
  'Classification',
  'Grad-CAM',
  'Research report',
];

export default function AboutPage() {
  return (
    <div className="page">
      <div className="page__inner">
        <div className="page__header about-header">
          <Logo size={34} />
          <div>
            <h1>About NeuroVision</h1>
            <p>An explainable AI research workflow for brain MRI analysis.</p>
          </div>
        </div>

        <section className="about-section panel">
          <h2>Mission</h2>
          <p>
            Building an explainable AI research workflow for brain MRI analysis — combining
            detection, classification, and visual interpretability so every prediction can be
            examined rather than taken on faith.
          </p>
        </section>

        <section className="about-section panel">
          <h2>Technology</h2>
          <div className="about-stack">
            {STACK.map((t) => (
              <span key={t} className="about-stack__item mono">{t}</span>
            ))}
          </div>
        </section>

        <section className="about-section panel">
          <h2>Pipeline</h2>
          <div className="about-pipeline">
            {PIPELINE.map((step, i) => (
              <div className="about-pipeline__step-wrap" key={step}>
                <div className="about-pipeline__step">{step}</div>
                {i < PIPELINE.length - 1 && <ArrowDown size={14} className="about-pipeline__arrow" />}
              </div>
            ))}
          </div>
        </section>

        <section className="about-section panel about-section--muted">
          <p>
            NeuroVision is an academic / research prototype built for coursework and portfolio
            purposes. It is not a certified or regulated medical device, and its outputs must not
            be used to diagnose, treat, or make clinical decisions about any individual.
          </p>
        </section>
      </div>
    </div>
  );
}
