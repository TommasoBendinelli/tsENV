import { expect, test } from 'vitest';
import { normalizeRegistry } from '../components/dashboard/controller/registryNormalization';
import { normalizeModelRecord } from '../app/api/registry/modelSchema';

test('normalizeRegistry preserves each child intervention_time', () => {
  const run: any = {
    run_id: '1',
    parent_id: null,
    parameters: {},
    intervention_time: 10,
    interventions: [
      { name: 'x_001', parent_id: '1', depth: 1, intervention_time: 10, variable: 'x', value: 1, status: 'success', timestamp: 't1' },
      { name: 'x_002', parent_id: 'x_001', depth: 2, intervention_time: 20, variable: 'x', value: 2, status: 'success', timestamp: 't2' },
    ],
    status: 'success',
    timestamp: 'r1',
    sampling_rate_hz: 50,
  };

  const [normalized] = normalizeRegistry([run], {
    selectedModel: 'ModelA',
    buildInterventionDisplayNames: (nextRun) => nextRun.interventions,
  });

  expect(normalized.intervention_time).toBe(10);
  expect(normalized.interventions.map((iv) => iv.intervention_time)).toEqual([10, 20]);
});

test('normalizeRegistry does not use missing baseline intervention_time for children', () => {
  const run: any = {
    run_id: '7d1b3f1a85c049ff9b8bd7c7a18d4041',
    parent_id: null,
    parameters: {},
    intervention_time: null,
    interventions: [
      { name: '7e819efac49b9697b79472c289e98675', parent_id: '7d1b3f1a85c049ff9b8bd7c7a18d4041', depth: 1, intervention_time: 2.05, variable: 'mass', value: 8.017315736158526, status: 'success', timestamp: 't1' },
      { name: 'dcbbcb9aaba84f95b6e6c410a573039a', parent_id: '7d1b3f1a85c049ff9b8bd7c7a18d4041', depth: 1, intervention_time: 1.48, variable: 'drag_coeff', value: 18.435342213785837, status: 'success', timestamp: 't2' },
    ],
    status: 'success',
    timestamp: 'r1',
    sampling_rate_hz: 400,
  };

  const [normalized] = normalizeRegistry([run], {
    selectedModel: 'BallDrop',
    buildInterventionDisplayNames: (nextRun) => nextRun.interventions,
  });

  expect(normalized.intervention_time).toBe(0);
  expect(normalized.interventions.map((iv) => iv.intervention_time)).toEqual([2.05, 1.48]);
});

test('normalizeModelRecord preserves intervention_time on baseline and child', () => {
  const payload: any = {
    version: 1,
    model_id: 'ModelA',
    metadata: {},
    baselines: [
      {
        run_id: '1',
        parent_id: null,
        parameters: {},
        intervention_time: 7,
        interventions: [
          {
            name: 'x_001',
            parent_id: '1',
            depth: 1,
            intervention_time: 7,
            variable: 'x',
            value: 1,
            end_time_simulation: 10,
            time0_end_time_simulation: 10,
            status: 'success',
            timestamp: 't1',
          },
        ],
        sampling_rate_hz: 50,
        end_time_simulation: 10,
        status: 'success',
        timestamp: 'r1',
      },
    ],
  };

  const normalized = normalizeModelRecord(payload, { modelId: 'ModelA' });
  expect(normalized.baselines[0].intervention_time).toBe(7);
  expect(normalized.baselines[0].interventions[0].intervention_time).toBe(7);
});

test('normalizeModelRecord rejects unknown keys (schema additionalProperties=false)', () => {
  const payload: any = {
    version: 1,
    model_id: 'ModelA',
    metadata: {},
    baselines: [],
    extra_key: 123,
  };
  expect(() => normalizeModelRecord(payload, { modelId: 'ModelA' })).toThrow(/Invalid schema/i);
});

test('normalizeModelRecord rejects legacy array format', () => {
  const legacyArray: any = [
    {
      run_id: '1',
      parent_id: null,
      parameters: {},
      interventions: [
        { name: 'x_001', parent_id: '1', depth: 1, intervention_time: 3, variable: 'x', value: 1, status: 'success', timestamp: 't1' },
      ],
      sampling_rate_hz: 50,
      status: 'success',
      timestamp: 'r1',
    },
  ];

  expect(() => normalizeModelRecord(legacyArray, { modelId: 'ModelA' })).toThrow(
    /must be a JSON object/i,
  );
});
