import type { Intervention, RunRecord } from '../types';

type Allocator = { nextName: (variable: string) => string };

const asPositiveFinite = (value: unknown): number | null => {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
};

const sortInterventionsForClone = (interventions: Intervention[]) => {
  return [...(interventions || [])].sort((a, b) => {
    const depthA = Number((a as any)?.depth || 1);
    const depthB = Number((b as any)?.depth || 1);
    if (depthA !== depthB) return depthA - depthB;
    const timeA = String((a as any)?.timestamp || '');
    const timeB = String((b as any)?.timestamp || '');
    if (timeA !== timeB) return timeA.localeCompare(timeB);
    return String((a as any)?.name || '').localeCompare(String((b as any)?.name || ''));
  });
};

export const buildNameMapForRunClone = (
  interventions: Intervention[],
  allocator: Allocator,
) => {
  const nameMap = new Map<string, string>();
  for (const iv of sortInterventionsForClone(interventions)) {
    const variable = String(iv.variable || '').trim();
    const newName = allocator.nextName(variable);
    nameMap.set(iv.name, newName);
  }
  return nameMap;
};

export const cloneInterventionsForBaseline = (opts: {
  sourceRunId: string;
  sourceInterventions: Intervention[];
  nameMap: Map<string, string>;
  newRunId: string;
  interventionTime: number;
  now: string;
  getValueOverride?: (source: Intervention) => unknown;
  baseFromSource?: boolean;
  baselineEndTimeInputS?: number;
  baselineEndTimeSimulation?: number;
}) => {
  return opts.sourceInterventions.map((iv) => {
    const base: any = opts.baseFromSource ? { ...iv } : {};
    const name = opts.nameMap.get(iv.name) as string;
    const originalParent = iv.parent_id;
    const parent_id =
      originalParent && originalParent !== opts.sourceRunId
        ? (opts.nameMap.get(originalParent) || opts.newRunId)
        : opts.newRunId;
    const value = opts.getValueOverride ? opts.getValueOverride(iv) : iv.value;
    const sourceEndTimeInput = asPositiveFinite((iv as any).end_time_input_s);
    const fallbackEndTimeInput = asPositiveFinite(opts.baselineEndTimeInputS);
    const resolvedEndTimeInput = sourceEndTimeInput ?? fallbackEndTimeInput;
    if (resolvedEndTimeInput === null) {
      throw new Error(
        `Cannot clone intervention '${String(iv.name || '').trim() || 'unknown'}': missing valid end_time_input_s.`,
      );
    }
    const sourceEndTime = asPositiveFinite((iv as any).end_time_simulation);
    const sourceTime0EndTime = asPositiveFinite((iv as any).time0_end_time_simulation);
    const fallbackEndTime = asPositiveFinite(opts.baselineEndTimeSimulation);
    const resolvedEndTime = sourceEndTime ?? fallbackEndTime ?? resolvedEndTimeInput;
    const resolvedTime0EndTime = sourceTime0EndTime
      ? sourceTime0EndTime
      : resolvedEndTime;
    return {
      ...base,
      name,
      parent_id,
      depth: (iv as any).depth || 1,
      intervention_time: opts.interventionTime,
      variable: iv.variable,
      value,
      end_time_input_s: resolvedEndTimeInput,
      end_time_simulation: resolvedEndTime,
      time0_end_time_simulation: resolvedTime0EndTime,
      status: 'not_run',
      timestamp: opts.now,
    } as Intervention;
  });
};

export const cloneBaselineRunShell = (opts: {
  source: RunRecord | null;
  runId: string;
  now: string;
  parameters: Record<string, any>;
  interventionTime: number;
  samplingRateHz: number;
  interventions: Intervention[];
}) => {
  if (!opts.source) {
    throw new Error('Cannot clone run shell without a source baseline run.');
  }
  const base: any = opts.source ? JSON.parse(JSON.stringify(opts.source)) : {};
  const sourceEndTimeInput = asPositiveFinite((opts.source as any)?.end_time_input_s);
  if (sourceEndTimeInput === null) {
    throw new Error(`Run '${String((opts.source as any)?.run_id || '').trim() || 'unknown'}' is missing valid end_time_input_s.`);
  }
  const sourceEndTime = asPositiveFinite((opts.source as any)?.end_time_simulation);
  const resolvedEndTime = sourceEndTime ?? sourceEndTimeInput;
  const next: RunRecord = {
    ...base,
    run_id: opts.runId,
    parent_id: null,
    parameters: opts.parameters,
    intervention_time: opts.interventionTime,
    interventions: opts.interventions,
    sampling_rate_hz: opts.samplingRateHz,
    end_time_input_s: sourceEndTimeInput,
    end_time_simulation: resolvedEndTime,
    status: 'not_run',
    timestamp: opts.now,
  };
  return next;
};
