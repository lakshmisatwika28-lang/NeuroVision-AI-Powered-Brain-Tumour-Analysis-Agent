import { useEffect, useState } from 'react';
import Logo from '../common/Logo';
import './IntroAnimation.css';

export default function IntroAnimation({ onComplete }) {
  const [leaving, setLeaving] = useState(false);

  useEffect(() => {
    const leaveTimer = setTimeout(() => setLeaving(true), 4600);
    const doneTimer = setTimeout(() => onComplete(), 5200);
    return () => {
      clearTimeout(leaveTimer);
      clearTimeout(doneTimer);
    };
  }, [onComplete]);

  return (
    <div className={`intro ${leaving ? 'intro--leaving' : ''}`}>
      <div className="intro__glow" />
      <div className="intro__grid" />

      <div className="intro__scan" />

      <div className="intro__content">
        <div className="intro__mark">
          <div className="intro__ring intro__ring--1" />
          <div className="intro__ring intro__ring--2" />
          <Logo size={72} className="intro__logo" />
        </div>

        <h1 className="intro__title">
          <span>NEURO</span>
          <span className="intro__title-accent">VISION</span>
        </h1>

        <p className="intro__subtitle">
          Explainable Brain MRI Classification &amp; Tumor Risk Analysis
        </p>

        <div className="intro__tag">RESEARCH PROTOTYPE</div>
      </div>
    </div>
  );
}
