import fs from 'fs';
import path from 'path';

export const EXPERIMENT_CONFIG_FILENAME = 'experiment_config.json';

type AnyRecord = Record<string, any>;
export type ObservableSignalType = 'continuous' | 'impulse_like';

const isRecord = (value: unknown): value is AnyRecord => (
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

export function getExperimentConfigPath(modelDir: string) {
  return path.join(modelDir, EXPERIMENT_CONFIG_FILENAME);
}

export function readExperimentConfigFile(modelDir: string): AnyRecord | null {
  const configPath = getExperimentConfigPath(modelDir);
  if (!fs.existsSync(configPath)) return null;
  const parsed = JSON.parse(fs.readFileSync(configPath, 'utf8'));
  return isRecord(parsed) ? parsed : null;
}

export type LegacyParameterSpec = {
  key: string;
  sampling?: {
    min: number;
    max: number;
    type: 'uniform' | 'loguniform';
  } | null;
};

export function migrateLegacyParameterSpecsIntoExperimentConfig(
  experimentConfig: AnyRecord,
  legacyParameters: unknown
): { next: AnyRecord; changed: boolean } {
  const params = Array.isArray(legacyParameters) ? legacyParameters : [];
  const rawExposedVariables = isRecord(experimentConfig.exposed_variables)
    ? experimentConfig.exposed_variables
    : {};
  const nextExposedVariables: AnyRecord = {
    initial_state: isRecord(rawExposedVariables.initial_state)
      ? { ...(rawExposedVariables.initial_state as AnyRecord) }
      : {},
    parameters: isRecord(rawExposedVariables.parameters)
      ? { ...(rawExposedVariables.parameters as AnyRecord) }
      : {},
  };

  let changed = false;
  for (const item of params) {
    if (!isRecord(item)) continue;
    const key = String((item as any).key ?? '').trim();
    if (!key) continue;
    if (Object.prototype.hasOwnProperty.call(nextExposedVariables.parameters, key)) continue;
    const sampling = (item as any).sampling;
    if (!isRecord(sampling)) continue;
    const min = Number((sampling as any).min);
    const max = Number((sampling as any).max);
    const type = String((sampling as any).type ?? '').toLowerCase();
    if (!Number.isFinite(min) || !Number.isFinite(max)) continue;
    if (type !== 'uniform' && type !== 'loguniform') continue;
    nextExposedVariables.parameters[key] = {
      allowed_intervals: [min, max],
      sampling_strategy: type,
    };
    changed = true;
  }

  if (!changed) return { next: experimentConfig, changed: false };
  return {
    next: { ...experimentConfig, exposed_variables: nextExposedVariables },
    changed: true,
  };
}

export function extractConfiguredObservableSignals(experimentConfig: unknown): string[] {
  if (!isRecord(experimentConfig)) return [];
  const rawObservableSignals = experimentConfig.observable_signals;
  if (!isRecord(rawObservableSignals)) return [];
  const signalList = Array.isArray(rawObservableSignals.observable_signals)
    ? rawObservableSignals.observable_signals
    : [];
  return signalList
    .map((signal: unknown) => String(signal ?? '').trim())
    .filter((signal: string) => Boolean(signal) && signal !== 'time');
}

export function extractObservableSignalTypes(
  experimentConfig: unknown
): Record<string, ObservableSignalType> {
  if (!isRecord(experimentConfig)) return {};
  const rawObservableSignals = experimentConfig.observable_signals;
  if (!isRecord(rawObservableSignals) || !isRecord(rawObservableSignals.signal_type)) {
    return {};
  }
  const out: Record<string, ObservableSignalType> = {};
  for (const [rawKey, rawValue] of Object.entries(rawObservableSignals.signal_type)) {
    const key = String(rawKey ?? '').trim();
    if (!key) continue;
    if (!isRecord(rawValue)) continue;
    const signalType = String(rawValue.type ?? '').trim().toLowerCase().replace(/-/g, '_');
    if (signalType === 'continuous' || signalType === 'impulse_like') {
      out[key] = signalType;
    }
  }
  return out;
}
