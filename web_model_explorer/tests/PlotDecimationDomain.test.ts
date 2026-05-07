import { describe, expect, test } from 'vitest';
import fixture from './fixtures/plot_decimation_fixture.json';
import {
  computeTimeRateDecimatedIndices,
  computeTimeRateDecimationPlan,
  pickSignedMaxAbsByWindows,
} from '@/components/dashboard/domains/plotDecimation';

describe('plot decimation parity fixture', () => {
  test('matches fixture cases', () => {
    for (const caseData of fixture.cases) {
      const computed = computeTimeRateDecimatedIndices({
        timeValues: caseData.time_values,
        targetSamplingRateHz: caseData.target_sampling_rate_hz,
      });
      const normalized = computed === null
        ? caseData.time_values.map((_, idx) => idx)
        : computed;
      expect(normalized, caseData.id).toEqual(caseData.expected_indices);
    }
  });

  test('mass spring special window rule matches fixture', () => {
    for (const caseData of fixture.cases) {
      if (!Array.isArray((caseData as any).signal_values)) continue;
      const plan = computeTimeRateDecimationPlan({
        timeValues: caseData.time_values,
        targetSamplingRateHz: caseData.target_sampling_rate_hz,
      });
      expect(plan, caseData.id).not.toBeNull();
      const values = (caseData as any).signal_values.map((v: unknown) => Number(v));
      const actual = pickSignedMaxAbsByWindows(values, (plan?.windows || []));
      expect(actual, caseData.id).toEqual((caseData as any).expected_special_values);
    }
  });
});
