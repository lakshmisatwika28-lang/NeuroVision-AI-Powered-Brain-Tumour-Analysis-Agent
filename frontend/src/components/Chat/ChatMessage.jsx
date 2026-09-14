import Logo from '../common/Logo';
import { User } from 'lucide-react';
import './ChatMessage.css';

export default function ChatMessage({ role, children }) {
  const isUser = role === 'user';
  return (
    <div className={`chat-msg ${isUser ? 'chat-msg--user' : 'chat-msg--ai'} fade-in`}>
      <div className="chat-msg__avatar">
        {isUser ? <User size={15} /> : <Logo size={20} />}
      </div>
      <div className="chat-msg__body">{children}</div>
    </div>
  );
}
