'use client';

export const DEFAULT_NUMERIC_SIGNIFICANT_DIGITS = 4;

export function resolveNumericSignificantDigits(value: unknown): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_NUMERIC_SIGNIFICANT_DIGITS;
  const rounded = Math.trunc(parsed);
  if (rounded < 1) return 1;
  if (rounded > 15) return 15;
  return rounded;
}

export function roundToSignificantDigits(
  value: number,
  significantDigits: number = DEFAULT_NUMERIC_SIGNIFICANT_DIGITS,
): number {
  if (!Number.isFinite(value)) return value;
  if (value === 0) return 0;
  const digits = resolveNumericSignificantDigits(significantDigits);
  const rounded = Number.parseFloat(value.toPrecision(digits));
  return Object.is(rounded, -0) ? 0 : rounded;
}

export function formatNumericSignificantDigits(
  value: unknown,
  significantDigits: number = DEFAULT_NUMERIC_SIGNIFICANT_DIGITS,
): string {
  if (value === null || value === undefined || value === '') return '';
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return String(value).trim();
  return String(roundToSignificantDigits(parsed, significantDigits));
}

export function formatTimeLabel(time: number | string | null | undefined) {
  if (time === null || time === undefined) return '';
  const timeValue = Number(time);
  if (Number.isFinite(timeValue)) {
    return Number.isInteger(timeValue) ? String(Math.trunc(timeValue)) : String(timeValue);
  }
  return String(time).trim();
}

export function resolveSignalDisplayLabel(
  signal: string,
  signalDisplayNames?: Record<string, string> | null,
) {
  const key = String(signal || '').trim();
  if (!key) return '';
  const mapped = String(signalDisplayNames?.[key] ?? '').trim();
  return mapped || key;
}

export function formatSignalLabel(
  signal: string,
  signalDisplayNames?: Record<string, string> | null,
) {
  return resolveSignalDisplayLabel(signal, signalDisplayNames);
}

export function formatVariableLabel(
  signal: string,
  signalDisplayNames?: Record<string, string> | null,
) {
  return formatSignalLabel(signal, signalDisplayNames);
}

export function collectBaselineSubtreeRunIds(run: {
  run_id?: unknown;
  interventions?: Array<{
    name?: unknown;
    time0_baseline_uuid?: unknown;
  }> | null;
}): string[] {
  const ids: string[] = [];
  const seen = new Set<string>();
  const add = (value: unknown) => {
    const id = String(value ?? '').trim();
    if (!id || seen.has(id)) return;
    seen.add(id);
    ids.push(id);
  };

  add(run?.run_id);
  for (const intervention of run?.interventions ?? []) {
    add(intervention?.name);
    add(intervention?.time0_baseline_uuid);
  }
  return ids;
}

export function shouldIgnoreRowClick(target: EventTarget | null) {
  if (!(target instanceof Element)) return false;
  return Boolean(target.closest('button, input, select, textarea, a, svg'));
}
