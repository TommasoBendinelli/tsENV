import type { RunRecord } from '../types';

type RegistryBoundUiState = {
  selectedRunIds: string[];
  expandedInterventions: string[];
  selectedRunRowIds: string[];
  runsData: Record<string, any>;
  bulkTimeByRunId: Record<string, string>;
};

export const collectRegistryRunIds = (
  baselines: RunRecord[] | null | undefined,
): Set<string> => {
  const knownIds = new Set<string>();
  for (const baseline of baselines ?? []) {
    const baselineId = String(baseline?.run_id || '').trim();
    if (baselineId) knownIds.add(baselineId);
    for (const intervention of baseline?.interventions ?? []) {
      const childId = String(intervention?.name || '').trim();
      if (childId) knownIds.add(childId);
      const time0Id = String((intervention as any)?.time0_baseline_uuid || '').trim();
      if (time0Id) knownIds.add(time0Id);
    }
  }
  return knownIds;
};

export const collectRegistryBaselineIds = (
  baselines: RunRecord[] | null | undefined,
): Set<string> => {
  const baselineIds = new Set<string>();
  for (const baseline of baselines ?? []) {
    const baselineId = String(baseline?.run_id || '').trim();
    if (baselineId) baselineIds.add(baselineId);
  }
  return baselineIds;
};

const filterStringArray = (values: string[], allowed: Set<string>): string[] =>
  values.filter((value) => allowed.has(String(value || '').trim()));

const filterRecordKeys = <T,>(
  value: Record<string, T>,
  allowed: Set<string>,
): Record<string, T> => {
  const next: Record<string, T> = {};
  for (const [rawKey, rawValue] of Object.entries(value)) {
    const key = String(rawKey || '').trim();
    if (!key || !allowed.has(key)) continue;
    next[key] = rawValue;
  }
  return next;
};

export const sanitizeRegistryBoundUiState = (
  baselines: RunRecord[] | null | undefined,
  state: RegistryBoundUiState,
): RegistryBoundUiState => {
  const knownRunIds = collectRegistryRunIds(baselines);
  const knownBaselineIds = collectRegistryBaselineIds(baselines);
  return {
    selectedRunIds: filterStringArray(state.selectedRunIds, knownRunIds),
    expandedInterventions: filterStringArray(state.expandedInterventions, knownBaselineIds),
    selectedRunRowIds: filterStringArray(state.selectedRunRowIds, knownBaselineIds),
    runsData: filterRecordKeys(state.runsData, knownRunIds),
    bulkTimeByRunId: filterRecordKeys(state.bulkTimeByRunId, knownBaselineIds),
  };
};
