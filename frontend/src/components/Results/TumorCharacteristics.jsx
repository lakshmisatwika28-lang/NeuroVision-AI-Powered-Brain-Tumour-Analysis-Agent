import { Ruler, MapPin, BoxSelect, MoveHorizontal, MoveVertical } from 'lucide-react';
import './TumorCharacteristics.css';

export default function TumorCharacteristics({ data }) {
  const metrics = [
    { icon: Ruler, label: 'Relative area', value: `${data.relativeArea}%` },
    { icon: BoxSelect, label: 'Area category', value: data.areaCategory },
    { icon: MapPin, label: 'Location', value: data.combinedLocation },
    { icon: MoveHorizontal, label: 'Horizontal position', value: data.horizontal },
    { icon: MoveVertical, label: 'Vertical position', value: data.vertical },
  ];

  return (
    <div className="tumor-char panel">
      <div className="tumor-char__title">Tumor characteristics</div>
      <div className="tumor-char__grid">
        {metrics.map((m) => (
          <div className="tumor-char__metric" key={m.label}>
            <m.icon size={14} className="tumor-char__icon" />
            <div>
              <div className="tumor-char__value">{m.value}</div>
              <div className="tumor-char__label">{m.label}</div>
            </div>
          </div>
        ))}
      </div>
      <div className="tumor-char__dims mono">
        {data.width}px × {data.height}px &nbsp;·&nbsp; area ≈ {data.area}px²
      </div>
    </div>
  );
}
