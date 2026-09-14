export default function Logo({ size = 32, className = '' }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      <circle cx="32" cy="32" r="30" stroke="var(--accent-dim)" strokeWidth="1" opacity="0.5" />
      <circle cx="32" cy="32" r="22" stroke="var(--accent)" strokeWidth="1.4" opacity="0.8" />
      {/* neural nodes forming a stylized brain silhouette */}
      <path
        d="M22 24c0-5 4-9 9-9 3 0 5.5 1.6 7 3.8 1.4-1 3-1.6 4.8-1.6 4.6 0 8.2 3.7 8.2 8.2 0 1.6-.4 3-1.2 4.3 1.6 1.4 2.6 3.5 2.6 5.8 0 4.3-3.5 7.7-7.7 7.7-.7 0-1.4-.1-2-.3-1.3 2.7-4 4.6-7.2 4.6-2 0-3.8-.8-5.2-2-1.4 1.2-3.2 2-5.2 2-4.4 0-8-3.6-8-8 0-1 .2-1.9.5-2.8-2.4-1.2-4-3.7-4-6.5 0-3 1.8-5.6 4.4-6.8-.6-.9-1-2-1-3.1z"
        stroke="var(--text-1)"
        strokeWidth="1.3"
        opacity="0.9"
      />
      <circle cx="24" cy="26" r="1.6" fill="var(--accent)" />
      <circle cx="34" cy="21" r="1.6" fill="var(--accent)" />
      <circle cx="42" cy="27" r="1.6" fill="var(--accent-2)" />
      <circle cx="40" cy="37" r="1.6" fill="var(--accent)" />
      <circle cx="29" cy="40" r="1.6" fill="var(--accent-2)" />
      <circle cx="22" cy="34" r="1.6" fill="var(--accent)" />
      <line x1="24" y1="26" x2="34" y2="21" stroke="var(--accent)" strokeWidth="0.6" opacity="0.6" />
      <line x1="34" y1="21" x2="42" y2="27" stroke="var(--accent)" strokeWidth="0.6" opacity="0.6" />
      <line x1="42" y1="27" x2="40" y2="37" stroke="var(--accent)" strokeWidth="0.6" opacity="0.6" />
      <line x1="40" y1="37" x2="29" y2="40" stroke="var(--accent)" strokeWidth="0.6" opacity="0.6" />
      <line x1="29" y1="40" x2="22" y2="34" stroke="var(--accent)" strokeWidth="0.6" opacity="0.6" />
      <line x1="22" y1="34" x2="24" y2="26" stroke="var(--accent)" strokeWidth="0.6" opacity="0.6" />
      <line x1="24" y1="26" x2="42" y2="27" stroke="var(--accent-2)" strokeWidth="0.4" opacity="0.4" />
      <line x1="22" y1="34" x2="40" y2="37" stroke="var(--accent-2)" strokeWidth="0.4" opacity="0.4" />
    </svg>
  );
}
