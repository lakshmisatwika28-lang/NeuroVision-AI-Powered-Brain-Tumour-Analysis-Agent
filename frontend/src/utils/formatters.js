export function formatConfidence(value) {
  if (value === null || value === undefined) return '—';
  return `${value.toFixed(1)}%`;
}

export function formatRelativeDay(isoString) {
  const date = new Date(isoString);
  const now = new Date();
  const startOfDay = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate());
  const diffDays = Math.round((startOfDay(now) - startOfDay(date)) / 86400000);

  if (diffDays === 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return date.toLocaleDateString(undefined, { weekday: 'long' });
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function formatTime(isoString) {
  const date = new Date(isoString);
  return date.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
}

export function riskColor(level) {
  if (!level) return 'var(--text-2)';
  const l = level.toLowerCase();
  if (l === 'lower') return 'var(--risk-lower)';
  if (l === 'moderate') return 'var(--risk-moderate)';
  return 'var(--risk-higher)';
}
