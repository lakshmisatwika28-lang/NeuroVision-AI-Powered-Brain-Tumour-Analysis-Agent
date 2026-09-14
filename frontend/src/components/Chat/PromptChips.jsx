import './PromptChips.css';

const DEFAULT_PROMPTS = [
  'What did the model detect?',
  'How confident is the classification?',
  'What regions influenced the prediction?',
  'What does Grad-CAM mean?',
  'Explain YOLO26 vs ResNet50',
  'What are common risk factors for brain tumors?',
];

export default function PromptChips({ prompts = DEFAULT_PROMPTS, onSelect }) {
  return (
    <div className="prompt-chips">
      {prompts.map((p) => (
        <button key={p} className="prompt-chip" onClick={() => onSelect(p)}>
          {p}
        </button>
      ))}
    </div>
  );
}
