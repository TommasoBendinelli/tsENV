const compareCanonicalKeys = (left: string, right: string) => (
  left < right ? -1 : left > right ? 1 : 0
);

const sortJsonValue = (value: unknown): unknown => {
  if (Array.isArray(value)) return value.map(sortJsonValue);
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => compareCanonicalKeys(a, b))
      .map(([key, item]) => [key, sortJsonValue(item)]);
    return Object.fromEntries(entries);
  }
  return value;
};

const padPythonExponent = (text: string) => text.replace(
  /e([+-])(\d+)$/,
  (_match, sign: string, digits: string) => `e${sign}${digits.padStart(2, '0')}`,
);

const serializePythonNumber = (value: number): string => {
  if (!Number.isFinite(value)) return JSON.stringify(value);
  if (Object.is(value, -0) || value === 0) return '0';
  if (Number.isInteger(value)) return String(Math.trunc(value));
  const magnitude = Math.abs(value);
  if (magnitude < 1e-4 || magnitude >= 1e16) {
    return padPythonExponent(value.toExponential());
  }
  return String(value);
};

const serializePythonJson = (value: unknown): string => {
  if (value === null) return 'null';
  if (typeof value === 'number') return serializePythonNumber(value);
  if (typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((item) => serializePythonJson(item)).join(',')}]`;
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>);
    return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${serializePythonJson(item)}`).join(',')}}`;
  }
  return JSON.stringify(value);
};

export const canonicalJson = (value: unknown): string => serializePythonJson(sortJsonValue(value));
