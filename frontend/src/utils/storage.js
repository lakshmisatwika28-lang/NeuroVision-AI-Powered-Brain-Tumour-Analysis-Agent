// Local, on-device persistence for the analysis history prototype.
// NOTE: This is a research prototype. History is stored only in this
// browser's localStorage — nothing is transmitted anywhere by this layer.

const KEY = 'neurovision_history_v1';
const INTRO_KEY = 'neurovision_intro_seen';

export function loadHistory() {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function saveHistory(entries) {
  try {
    localStorage.setItem(KEY, JSON.stringify(entries));
  } catch {
    // localStorage unavailable (private mode / quota) — fail silently,
    // history simply won't persist across reloads.
  }
}

export function addHistoryEntry(entry) {
  const entries = loadHistory();
  const next = [entry, ...entries].slice(0, 100);
  saveHistory(next);
  return next;
}

export function deleteHistoryEntry(id) {
  const entries = loadHistory().filter((e) => e.id !== id);
  saveHistory(entries);
  return entries;
}

export function clearHistory() {
  saveHistory([]);
  return [];
}

export function hasSeenIntro() {
  try {
    return sessionStorage.getItem(INTRO_KEY) === 'true';
  } catch {
    return false;
  }
}

export function markIntroSeen() {
  try {
    sessionStorage.setItem(INTRO_KEY, 'true');
  } catch {
    // ignore
  }
}
