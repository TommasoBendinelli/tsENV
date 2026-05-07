'use client';

import type { SamplingSpec } from '../types';

type SamplingDetail = {
  initial?: SamplingSpec;
  perturbation?: SamplingSpec;
};

type ExposedVariableGroups = {
  parameters: string[];
  initialState: string[];
};

export function isValidNumberInput(value: unknown) {
  if (value === null || value === undefined) return false;
  if (typeof value === 'string' && value.trim() === '') return false;
  const num = Number(value);
  return Number.isFinite(num);
}

export function parseSamplingSpec(entry: unknown): SamplingSpec | null {
  if (!entry) return null;
  if (typeof entry === 'object') {
    const typed = entry as Record<string, any>;
    if (Array.isArray(typed.allowed_intervals) && typed.allowed_intervals.length >= 2 && typed.sampling_strategy) {
      const min = Number(typed.allowed_intervals[0]);
      const max = Number(typed.allowed_intervals[1]);
      const kind = String(typed.sampling_strategy || '').toLowerCase();
      if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
      if (kind !== 'uniform' && kind !== 'loguniform') return null;
      return { min, max, type: kind as SamplingSpec['type'] };
    }
  }
  return null;
}

export function buildSamplingDetails(payload: unknown) {
  const root = payload as Record<string, any> | null;
  const next: Record<string, SamplingDetail> = {};
  const exposedVariables = root?.exposed_variables;
  if (!exposedVariables || typeof exposedVariables !== 'object') {
    return next;
  }
  for (const groupName of ['initial_state', 'parameters'] as const) {
    const group = (exposedVariables as Record<string, unknown>)[groupName];
    if (!group || typeof group !== 'object') continue;
    for (const [key, entry] of Object.entries(group as Record<string, unknown>)) {
      const spec = parseSamplingSpec(entry);
      if (spec) {
        next[key] = { initial: spec, perturbation: spec };
      }
    }
  }
  return next;
}

export function extractExposedVariableGroups(payload: unknown): ExposedVariableGroups {
  const root = payload as Record<string, any> | null;
  const exposedVariables = root?.exposed_variables;
  if (!exposedVariables || typeof exposedVariables !== 'object') {
    return {
      parameters: [],
      initialState: [],
    };
  }

  const parameters = exposedVariables.parameters;
  const initialState = exposedVariables.initial_state;

  return {
    parameters:
      parameters && typeof parameters === 'object'
        ? Object.keys(parameters as Record<string, unknown>).map((key) => String(key ?? '').trim()).filter(Boolean)
        : [],
    initialState:
      initialState && typeof initialState === 'object'
        ? Object.keys(initialState as Record<string, unknown>).map((key) => String(key ?? '').trim()).filter(Boolean)
        : [],
  };
}

export function buildSamplingIntervals(
  details: Record<string, SamplingDetail>,
  mode: 'initial' | 'perturbation'
) {
  const next: Record<string, SamplingSpec> = {};
  for (const [key, detail] of Object.entries(details)) {
    const spec =
      mode === 'initial'
        ? (detail.initial ?? detail.perturbation)
        : (detail.perturbation ?? detail.initial);
    if (spec) {
      next[key] = spec;
    }
  }
  return next;
}

export function sampleFromSpec(spec: SamplingSpec) {
  const min = Number.isFinite(spec.min) ? spec.min : 0;
  const max = Number.isFinite(spec.max) ? spec.max : min;
  const lo = Math.min(min, max);
  const hi = Math.max(min, max);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  if (spec.type === 'loguniform' && lo > 0 && hi > 0) {
    const logLo = Math.log(lo);
    const logHi = Math.log(hi);
    return Math.exp(logLo + Math.random() * (logHi - logLo));
  }
  return lo + Math.random() * (hi - lo);
}
