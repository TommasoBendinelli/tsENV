const parseNumericRunId = (value: unknown): number | null => {
  if (typeof value !== 'string') return null;
  const trimmed = value.trim();
  if (!/^\d+$/.test(trimmed)) return null;
  const parsed = Number(trimmed);
  if (!Number.isSafeInteger(parsed)) return null;
  return parsed;
};

export const getNextNumericRunId = (
  runs: Array<{ run_id: string }>,
  diskRunIds: string[],
): number => {
  let max = 0;
  for (const run of runs) {
    const parsed = parseNumericRunId(run.run_id);
    if (parsed === null) continue;
    max = Math.max(max, parsed);
  }
  for (const runId of diskRunIds) {
    const parsed = parseNumericRunId(runId);
    if (parsed === null) continue;
    max = Math.max(max, parsed);
  }
  return max + 1;
};
