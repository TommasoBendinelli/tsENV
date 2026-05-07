export const parseChoiceList = (value: string) => (
  String(value || '')
    .split(',')
    .map(item => item.trim())
    .filter(item => item.length > 0)
);

export const normalizeSignalCase = (signals: string[]) => (
  Array.from(new Set(signals.map(signal => signal.trim()).filter(Boolean))).sort()
);

