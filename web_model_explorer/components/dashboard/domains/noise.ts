'use client';

import type { RunRecord } from '../types';

export function computeRunNoiseById(registry: RunRecord[]) {
  const mapping: Record<string, number> = {};
  for (const run of registry) {
    const noiseValue = Number(run.noise);
    const noise = Number.isFinite(noiseValue) && noiseValue > 0 ? noiseValue : 0;
    mapping[run.run_id] = noise;
    for (const iv of run.interventions) {
      mapping[iv.name] = noise;
    }
  }
  return mapping;
}
