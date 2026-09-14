import { Info } from 'lucide-react';

export default function Disclaimer({ children, className = '' }) {
  return (
    <div className={`disclaimer ${className}`}>
      <Info size={15} />
      <span>{children}</span>
    </div>
  );
}
