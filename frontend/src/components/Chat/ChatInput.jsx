import { useState } from 'react';
import { ArrowUp } from 'lucide-react';
import './ChatInput.css';

export default function ChatInput({ onSend, disabled, placeholder }) {
  const [value, setValue] = useState('');

  const submit = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
  };

  return (
    <div className="chat-input">
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && submit()}
        placeholder={placeholder || 'Ask about this analysis…'}
        disabled={disabled}
      />
      <button onClick={submit} disabled={disabled || !value.trim()} aria-label="Send">
        <ArrowUp size={16} />
      </button>
    </div>
  );
}
