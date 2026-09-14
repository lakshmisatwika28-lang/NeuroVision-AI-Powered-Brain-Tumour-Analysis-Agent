import { useEffect, useRef, useState } from 'react';
import { formatConfidence } from '../../utils/formatters';
import './ConfidenceBar.css';

export default function ConfidenceBar({ value, color = 'var(--accent)', label }) {
  const [width, setWidth] = useState(0);
  const ref = useRef();

  useEffect(() => {
    const raf = requestAnimationFrame(() => setWidth(value));
    return () => cancelAnimationFrame(raf);
  }, [value]);

  return (
    <div className="confidence-bar" ref={ref}>
      {label && <div className="confidence-bar__label">{label}</div>}
      <div className="confidence-bar__track">
        <div
          className="confidence-bar__fill"
          style={{ width: `${width}%`, background: color }}
        />
      </div>
      <div className="confidence-bar__value mono">{formatConfidence(value)}</div>
    </div>
  );
}
