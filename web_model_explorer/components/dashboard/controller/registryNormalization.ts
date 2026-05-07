import type { Intervention, RunRecord } from '../types';

export const normalizeRegistry = (
  nextRegistry: RunRecord[],
  opts: {
    selectedModel: string;
    buildInterventionDisplayNames: (run: RunRecord) => Intervention[];
  }
): RunRecord[] => {
  return nextRegistry.map((run) => {
    const interventionTime = Number(run.intervention_time);
    const normalizedInterventionTime = Number.isFinite(interventionTime) ? interventionTime : 0;
    const noiseValue = Number((run as any).noise);
    const normalizedNoise = Number.isFinite(noiseValue) && noiseValue > 0 ? noiseValue : 0;
    const normalizedInterventions = run.interventions.map((iv) => {
      const childInterventionTime = Number(iv.intervention_time);
      return {
        ...iv,
        intervention_time: Number.isFinite(childInterventionTime) ? childInterventionTime : 0,
      };
    });

    return {
      ...run,
      intervention_time: normalizedInterventionTime,
      interventions: opts.buildInterventionDisplayNames({
        ...(run as any),
        intervention_time: normalizedInterventionTime,
        interventions: normalizedInterventions,
      } as RunRecord),
      noise: normalizedNoise,
    } as RunRecord;
  });
};
