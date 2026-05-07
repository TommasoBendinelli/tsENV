export function buildRunShortcutRedirectPath(run: string) {
  const runId = String(run || '').trim();
  return `/?run=${encodeURIComponent(runId)}`;
}
