import { expect, test } from 'vitest';
import {
  collectRegistryBaselineIds,
  collectRegistryRunIds,
  sanitizeRegistryBoundUiState,
} from '../components/dashboard/controller/registryStateSanitizer';

const baselines = [
  {
    run_id: 'baseline1',
    interventions: [
      {
        name: 'iv1',
        time0_baseline_uuid: 'time0_1',
      },
      {
        name: 'iv2',
        time0_baseline_uuid: 'time0_2',
      },
    ],
  },
] as any;

test('collectRegistryRunIds includes baseline, child, and time0 ids', () => {
  const ids = collectRegistryRunIds(baselines);
  expect(ids.has('baseline1')).toBe(true);
  expect(ids.has('iv1')).toBe(true);
  expect(ids.has('iv2')).toBe(true);
  expect(ids.has('time0_1')).toBe(true);
  expect(ids.has('time0_2')).toBe(true);
});

test('sanitizeRegistryBoundUiState drops stale ids from cached dashboard state', () => {
  const sanitized = sanitizeRegistryBoundUiState(baselines, {
    selectedRunIds: ['stale_child', 'iv1'],
    expandedInterventions: ['baseline1', 'stale_baseline'],
    selectedRunRowIds: ['stale_baseline', 'baseline1'],
    runsData: {
      stale_child: { columns: [] },
      iv1: { columns: [] },
      time0_1: { columns: [] },
    },
    bulkTimeByRunId: {
      baseline1: '8.0',
      stale_baseline: '9.0',
    },
  });

  expect(sanitized.selectedRunIds).toEqual(['iv1']);
  expect(sanitized.expandedInterventions).toEqual(['baseline1']);
  expect(sanitized.selectedRunRowIds).toEqual(['baseline1']);
  expect(Object.keys(sanitized.runsData).sort()).toEqual(['iv1', 'time0_1']);
  expect(sanitized.bulkTimeByRunId).toEqual({ baseline1: '8.0' });
});

test('collectRegistryBaselineIds includes only baseline ids', () => {
  const ids = collectRegistryBaselineIds(baselines);
  expect(Array.from(ids)).toEqual(['baseline1']);
});
