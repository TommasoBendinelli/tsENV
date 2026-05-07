import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { beforeEach, describe, expect, test, vi } from 'vitest';
import { canonicalJson } from '../app/api/pythonCanonicalJson';

vi.mock('../app/api/registry/modelSchema', () => ({
  normalizeModelRecord: (payload: unknown) => payload,
  normalizeRuntimeModelRecord: (payload: unknown) => payload,
}));

vi.mock('../app/api/sharedSchemaAjv', () => ({
  assertValidAgainstSharedSchema: (_schema: string, payload: unknown) => payload,
}));

const writeJson = (filePath: string, payload: unknown) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2), 'utf8');
};

const touch = (filePath: string) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, '', 'utf8');
};

const writeEligibilityMetrics = (
  modelDir: string,
  baselines: Record<string, { eligible: boolean; children?: Record<string, unknown> }>
) => {
  writeJson(path.join(modelDir, 'runs', 'eligibility_metrics.json'), {
    timestamp: '2026-04-24T00:00:00Z',
    noise_adder_md5: null,
    basic_rule_md5: null,
    eligible_baselines: Object.values(baselines).filter((item) => item.eligible).length,
    total_baselines: Object.keys(baselines).length,
    baselines: Object.fromEntries(
      Object.entries(baselines).map(([baselineUuid, summary]) => [
        baselineUuid,
        {
          url: `http://localhost:3001/?run=${baselineUuid}`,
          family_eligible: summary.eligible,
          eligible: summary.eligible,
          children: summary.children ?? {},
        },
      ])
    ),
  });
};

const childDetectabilityMetric = (
  vsBaseline: 'yes' | 'no' | 'error',
  vsTime0Baseline: 'yes' | 'no' | 'error',
) => ({
  url: 'http://localhost:3001/?run=child',
  eligible: vsBaseline === 'yes' && vsTime0Baseline === 'yes',
  detectability: {
    vs_baseline: {
      environment_specific_detectability: vsBaseline === 'yes' ? 'yes' : 'no',
      detectable: vsBaseline === 'yes' ? 'yes' : 'no',
      detectability_output: {
        mean_euclidean_distance_clean_dirty: vsBaseline === 'error' ? [] : [1],
        mean_euclidean_distance_clean_baseline: vsBaseline === 'error' ? [] : [1],
        mean_SNR: vsBaseline === 'error' ? [] : [0],
        first_diff: vsBaseline === 'error' ? [] : [vsBaseline === 'yes' ? 1 : null],
      },
    },
    vs_time0_baseline: {
      detectable: vsTime0Baseline === 'yes' ? 'yes' : 'no',
      detectability_output: {
        mean_euclidean_distance_clean_dirty: vsTime0Baseline === 'error' ? [] : [1],
        mean_euclidean_distance_clean_baseline: vsTime0Baseline === 'error' ? [] : [1],
        mean_SNR: vsTime0Baseline === 'error' ? [] : [0],
        first_diff: vsTime0Baseline === 'error' ? [] : [vsTime0Baseline === 'yes' ? 1 : null],
      },
    },
  },
});

const sha256_32_hex = (text: string) =>
  crypto.createHash('sha256').update(text, 'utf8').digest('hex').slice(0, 32);

const computeParametersHash = (parameters: Record<string, unknown>) => sha256_32_hex(canonicalJson(parameters));
const computeChildParametersHash = (
  parentParametersHash: string,
  childParameters: Record<string, unknown>,
  interventionTime: unknown
) => (
  sha256_32_hex(canonicalJson({
    parent_parameters_hash: parentParametersHash,
    parameters: childParameters,
    intervention_time: interventionTime,
  }))
);
const computeTime0BaselineHash = (childParameterHash: string) => (
  sha256_32_hex(canonicalJson({ parameter_hash: childParameterHash, kind: 'time0' }))
);

const createFixtureModel = () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'registry-route-'));
  const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
  const modelDir = path.join(tmpRoot, 'models', 'simulink', 'SpecFirstModel');
  fs.mkdirSync(fakeWebRoot, { recursive: true });
  fs.mkdirSync(path.join(modelDir, 'generated'), { recursive: true });
  fs.mkdirSync(path.join(modelDir, 'runs'), { recursive: true });

  writeJson(path.join(modelDir, 'generated', 'metadata.json'), {
    simulink_signals_available: ['sig_a'],
    simscape_signals_available: ['sig_b'],
  });
  writeJson(path.join(modelDir, 'experiment_config.json'), {
    exposed_variables: {
      initial_state: {},
      parameters: {},
    },
    sampling_rate_hz: 10,
    end_time_input_s: 12,
    "detectability": {
      "continuous": {
        "min_srd_distance": 0.001,
        "epsilon_SRD": 0.001,
        "minimum_consecurive_below_SRD": 1,
      },
      "impulse_like": {
        "min_srd_distance": 0.3,
        "epsilon_SRD": 1.0,
      },
    },
    observable_signals: {
      observable_signals: ['sig_b', 'sig_a'],
      signal_type: {
        sig_b: { type: 'continuous' },
        sig_a: { type: 'continuous' },
      },
    },
  });
  writeJson(path.join(modelDir, 'description_levels.json'), {
    internal_naming_to_agent_facing_signal: {
      sig_a: 'signal a',
      sig_b: 'signal b',
      time: 'time',
    },
  });

  const baselineUuid = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
  const childUuid = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';
  const time0Uuid = 'cccccccccccccccccccccccccccccccc';
  const interventionTime = 3;
  const rawParameters = {
    x: 0.25,
  };
  const childParameters = { x: 1.5 };
  const parametersHash = computeParametersHash(rawParameters);
  const childHash = computeChildParametersHash(parametersHash, childParameters, interventionTime);

  writeJson(path.join(modelDir, 'model_run_specs.json'), {
    [baselineUuid]: {
      baseline_parameters: rawParameters,
      baseline_parameters_hash: parametersHash,
      children: {
        [childUuid]: {
          intervention_time: interventionTime,
          parameters: childParameters,
          parameter_hash: childHash,
          time0_baseline_hash: computeTime0BaselineHash(childHash),
          time0_baseline_uuid: time0Uuid,
        },
      },
    },
  });

  writeJson(path.join(modelDir, 'runs', 'model_record.json'), {
    [baselineUuid]: {
      parameters_hash: parametersHash,
      run_type: 'baseline',
      class_internal: 'baseline',
      class_agent_facing_name: 'baseline',
      status: 'success',
      timestamp: 't0',
      end_time_simulation: 10,
    },
    [childUuid]: {
      parameters_hash: childHash,
      run_type: 'intervention',
      class_internal: 'x',
      class_agent_facing_name: 'x',
      status: 'failed',
      timestamp: 't1',
      error: 'sim diverged',
    },
    dddddddddddddddddddddddddddddddd: {
      parameters_hash: 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
      run_type: 'intervention',
      class_internal: 'x',
      class_agent_facing_name: 'x',
      status: 'success',
    },
  });

  return { fakeWebRoot, modelDir, baselineUuid, childUuid, time0Uuid };
};

const createRenamedTransmissionLineHashFixtureModel = () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'registry-route-mixed-case-'));
  const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
  const modelDir = path.join(tmpRoot, 'models', 'simulink', 'MixedCaseSpecModel');
  fs.mkdirSync(fakeWebRoot, { recursive: true });
  fs.mkdirSync(path.join(modelDir, 'generated'), { recursive: true });
  fs.mkdirSync(path.join(modelDir, 'runs'), { recursive: true });

  writeJson(path.join(modelDir, 'generated', 'metadata.json'), {
    simulink_signals_available: ['Source Voltage', 'Output'],
  });
  writeJson(path.join(modelDir, 'experiment_config.json'), {
    exposed_variables: {
      initial_state: {},
      parameters: {},
    },
    sampling_rate_hz: 10000000000.0,
    end_time_input_s: 1e-6,
    "detectability": {
      "continuous": {
        "min_srd_distance": 0.001,
        "epsilon_SRD": 0.001,
        "minimum_consecurive_below_SRD": 1,
      },
      "impulse_like": {
        "min_srd_distance": 0.3,
        "epsilon_SRD": 1.0,
      },
    },
    observable_signals: {
      observable_signals: ['sig_a'],
      signal_type: {
        sig_a: { type: 'continuous' },
      },
    },
  });

  const baselineUuid = '3f945c6a0f9d4de68b6b73336e1db9fd';
  const childUuid = '9edc87275b7b4da4adf7af854491e5ab';
  const time0Uuid = '32e2eca767394f619029323264b9ec0e';
  const interventionTime = 2e-7;
  const rawParameters = {
    loadResistance: 48,
    Z: 46,
    C: 8.5e-11,
    segmentLength: 0.085,
    sourceamplitude_1: 4,
    sourceamplitude_2: 2.6,
    sourceamplitude_3: 1.8,
    sourcefrequency_1: 18,
    sourcefrequency_2: 48,
    sourcefrequency_3: 180,
  };
  const parametersHash = computeParametersHash(rawParameters);
  const childParameters = { loadResistance: 86 };
  const childHash = computeChildParametersHash(parametersHash, childParameters, interventionTime);

  writeJson(path.join(modelDir, 'model_run_specs.json'), {
    [baselineUuid]: {
      baseline_parameters: rawParameters,
      baseline_parameters_hash: parametersHash,
      children: {
        [childUuid]: {
          intervention_time: interventionTime,
          parameters: childParameters,
          parameter_hash: childHash,
          time0_baseline_hash: computeTime0BaselineHash(childHash),
          time0_baseline_uuid: time0Uuid,
        },
      },
    },
  });

  writeJson(path.join(modelDir, 'runs', 'model_record.json'), {
    [baselineUuid]: {
      parameters_hash: parametersHash,
      run_type: 'baseline',
      class_internal: 'baseline',
      class_agent_facing_name: 'baseline',
      status: 'success',
      timestamp: 't0',
      end_time_simulation: 9.999999974752427e-7,
    },
    [childUuid]: {
      parameters_hash: childHash,
      run_type: 'intervention',
      class_internal: 'loadResistance',
      class_agent_facing_name: 'loadResistance',
      status: 'success',
      end_time_input_s: 1e-6,
      end_time_simulation: 9.999999974752427e-7,
      time0_end_time_simulation: 9.999999974752427e-7,
      timestamp: 't1',
    },
  });

  return { fakeWebRoot, baselineUuid, childUuid };
};

const createFixtureModelWithoutRuntimeFile = () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'registry-route-missing-runtime-'));
  const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
  const modelDir = path.join(tmpRoot, 'models', 'simulink', 'SpecFirstModel');
  fs.mkdirSync(fakeWebRoot, { recursive: true });
  fs.mkdirSync(path.join(modelDir, 'generated'), { recursive: true });

  writeJson(path.join(modelDir, 'generated', 'metadata.json'), {
    simulink_signals_available: ['sig_a'],
    simscape_signals_available: ['sig_b'],
  });
  writeJson(path.join(modelDir, 'experiment_config.json'), {
    exposed_variables: {
      initial_state: {},
      parameters: {},
    },
    sampling_rate_hz: 10,
    end_time_input_s: 12,
    "detectability": {
      "continuous": {
        "min_srd_distance": 0.001,
        "epsilon_SRD": 0.001,
        "minimum_consecurive_below_SRD": 1,
      },
      "impulse_like": {
        "min_srd_distance": 0.3,
        "epsilon_SRD": 1.0,
      },
    },
    observable_signals: {
      observable_signals: ['sig_b', 'sig_a'],
      signal_type: {
        sig_b: { type: 'continuous' },
        sig_a: { type: 'continuous' },
      },
    },
  });
  writeJson(path.join(modelDir, 'description_levels.json'), {
    internal_naming_to_agent_facing_signal: {
      sig_a: 'signal a',
      sig_b: 'signal b',
      time: 'time',
    },
  });

  const baselineUuid = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
  const childUuid = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';
  const time0Uuid = 'cccccccccccccccccccccccccccccccc';
  const interventionTime = 3;
  const rawParameters = {
    x: 0.25,
  };
  const childParameters = { x: 1.5 };
  const parametersHash = computeParametersHash(rawParameters);
  const childHash = computeChildParametersHash(parametersHash, childParameters, interventionTime);

  writeJson(path.join(modelDir, 'model_run_specs.json'), {
    [baselineUuid]: {
      baseline_parameters: rawParameters,
      baseline_parameters_hash: parametersHash,
      children: {
        [childUuid]: {
          intervention_time: interventionTime,
          parameters: childParameters,
          parameter_hash: childHash,
          time0_baseline_hash: computeTime0BaselineHash(childHash),
          time0_baseline_uuid: time0Uuid,
        },
      },
    },
  });

  touch(path.join(modelDir, 'runs', baselineUuid, 'data.parquet'));
  touch(path.join(tmpRoot, 'tsENV_questions', 'SpecFirstModel', 'dataframes', `${childUuid}.parquet`));

  return { fakeWebRoot, baselineUuid, childUuid, time0Uuid };
};

const createFixtureModelWithSkippedChild = () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'registry-route-skipped-child-'));
  const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
  const modelDir = path.join(tmpRoot, 'models', 'simulink', 'SkippedChildModel');
  fs.mkdirSync(fakeWebRoot, { recursive: true });
  fs.mkdirSync(path.join(modelDir, 'generated'), { recursive: true });
  fs.mkdirSync(path.join(modelDir, 'runs'), { recursive: true });

  writeJson(path.join(modelDir, 'generated', 'metadata.json'), {
    simulink_signals_available: ['sig_a'],
  });
  writeJson(path.join(modelDir, 'experiment_config.json'), {
    exposed_variables: {
      initial_state: {},
      parameters: {},
    },
    sampling_rate_hz: 10,
    end_time_input_s: 12,
    "detectability": {
      "continuous": {
        "min_srd_distance": 0.001,
        "epsilon_SRD": 0.001,
        "minimum_consecurive_below_SRD": 1,
      },
      "impulse_like": {
        "min_srd_distance": 0.3,
        "epsilon_SRD": 1.0,
      },
    },
    observable_signals: {
      observable_signals: ['sig_a'],
      signal_type: {
        sig_a: { type: 'continuous' },
      },
    },
  });
  writeJson(path.join(modelDir, 'description_levels.json'), {
    internal_naming_to_agent_facing_signal: {
      sig_a: 'signal a',
      time: 'time',
    },
  });

  const baselineUuid = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
  const childUuid = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';
  const time0Uuid = 'cccccccccccccccccccccccccccccccc';
  const skippedChildUuid = 'dddddddddddddddddddddddddddddddd';
  const interventionTime = 3;
  const rawParameters = {
    x: 0.25,
  };
  const childParameters = { x: 1.5 };
  const parametersHash = computeParametersHash(rawParameters);
  const childHash = computeChildParametersHash(parametersHash, childParameters, interventionTime);

  writeJson(path.join(modelDir, 'model_run_specs.json'), {
    [baselineUuid]: {
      baseline_parameters: rawParameters,
      baseline_parameters_hash: parametersHash,
      children: {
        [childUuid]: {
          intervention_time: interventionTime,
          parameters: childParameters,
          parameter_hash: childHash,
          time0_baseline_hash: computeTime0BaselineHash(childHash),
          time0_baseline_uuid: time0Uuid,
        },
        [skippedChildUuid]: {
          intervention_time: interventionTime,
          parameters: { x: null },
          parameter_hash: null,
          time0_baseline_hash: null,
          time0_baseline_uuid: null,
        },
      },
    },
  });

  touch(path.join(modelDir, 'runs', baselineUuid, 'data.parquet'));
  touch(path.join(modelDir, 'runs', childUuid, 'data.parquet'));

  return { fakeWebRoot, modelDir, baselineUuid, childUuid, skippedChildUuid, time0Uuid };
};

describe('/api/registry strict storage source', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  test('GET /api/registry builds merged baselines from strict specs + flat runtime map', async () => {
    const { fakeWebRoot, modelDir, baselineUuid, childUuid, time0Uuid } = createFixtureModel();
    touch(path.join(modelDir, 'runs', baselineUuid, 'data.parquet'));
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=SpecFirstModel'));
    expect(res.status).toBe(200);
    const body = await res.json();
    const baseline = body?.modelRecord?.baselines?.[0];

    expect(baseline?.run_id).toBe(baselineUuid);
    expect(baseline?.intervention_time).toBe(3);
    expect(baseline?.end_time_input_s).toBe(12);
    expect(baseline?.status).toBe('success');
    expect(baseline?.eligible).toBeUndefined();
    expect(baseline?.interventions?.[0]?.name).toBe(childUuid);
    expect(baseline?.interventions?.[0]?.time0_baseline_uuid).toBe(time0Uuid);
    expect(baseline?.interventions?.[0]?.status).toBe('failed');
    expect(body?.signalDisplayNames).toEqual({ sig_a: 'signal a', sig_b: 'signal b', time: 'time' });
    expect(body?.availableSignals).toEqual(['sig_b', 'sig_a']);
    cwdSpy.mockRestore();
  });

  test('GET /api/registry exposes documented noise profiles when noise_adder.py exists', async () => {
    const { fakeWebRoot, modelDir } = createFixtureModel();
    fs.writeFileSync(
      path.join(modelDir, 'noise_adder.py'),
      [
        "NOISE_DICT = {'low': {}, 'high': {}}",
        "SNR_THR_DICT = {'low': {'global': [], 'local': []}, 'high': {'global': [], 'local': []}}",
        "def quantify_noise(clean_df, noisy_df, first_diff):",
        "    return {'global': [], 'local': []}",
        "def add_noise(df, first_diff, seed=0, noise_level='low'):",
        "    return df, quantify_noise(df, df, first_diff)",
      ].join('\n'),
      'utf8',
    );
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=SpecFirstModel'));
    expect(res.status).toBe(200);
    const body = await res.json();

    expect(body?.availableNoiseProfiles).toEqual(['none', 'low', 'high']);
    cwdSpy.mockRestore();
  });

  test('GET /api/registry merges true baseline eligibility from eligibility_metrics.json', async () => {
    const { fakeWebRoot, modelDir, baselineUuid } = createFixtureModel();
    touch(path.join(modelDir, 'runs', baselineUuid, 'data.parquet'));
    writeEligibilityMetrics(modelDir, {
      [baselineUuid]: { eligible: true },
    });
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=SpecFirstModel'));
    expect(res.status).toBe(200);
    const body = await res.json();
    const baseline = body?.modelRecord?.baselines?.[0];

    expect(baseline?.run_id).toBe(baselineUuid);
    expect(baseline?.eligible).toBe(true);
    cwdSpy.mockRestore();
  });

  test('GET /api/registry merges false baseline eligibility from eligibility_metrics.json', async () => {
    const { fakeWebRoot, modelDir, baselineUuid } = createFixtureModel();
    touch(path.join(modelDir, 'runs', baselineUuid, 'data.parquet'));
    writeEligibilityMetrics(modelDir, {
      [baselineUuid]: { eligible: false },
    });
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=SpecFirstModel'));
    expect(res.status).toBe(200);
    const body = await res.json();
    const baseline = body?.modelRecord?.baselines?.[0];

    expect(baseline?.run_id).toBe(baselineUuid);
    expect(baseline?.eligible).toBe(false);
    cwdSpy.mockRestore();
  });

  test('GET /api/registry exposes child detectability failures from eligibility_metrics.json', async () => {
    const { fakeWebRoot, modelDir, baselineUuid, childUuid } = createFixtureModel();
    touch(path.join(modelDir, 'runs', baselineUuid, 'data.parquet'));
    touch(path.join(modelDir, 'runs', childUuid, 'data.parquet'));
    writeEligibilityMetrics(modelDir, {
      [baselineUuid]: {
        eligible: false,
        children: {
          [childUuid]: childDetectabilityMetric('yes', 'no'),
        },
      },
    });
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=SpecFirstModel'));
    expect(res.status).toBe(200);
    const body = await res.json();
    const child = body?.modelRecord?.baselines?.[0]?.interventions?.[0];

    expect(child?.name).toBe(childUuid);
    expect(child?.detectability_failed).toBe(true);
    cwdSpy.mockRestore();
  });

  test('GET /api/registry treats baseline detectability error as child failure', async () => {
    const { fakeWebRoot, modelDir, baselineUuid, childUuid } = createFixtureModel();
    touch(path.join(modelDir, 'runs', baselineUuid, 'data.parquet'));
    touch(path.join(modelDir, 'runs', childUuid, 'data.parquet'));
    writeEligibilityMetrics(modelDir, {
      [baselineUuid]: {
        eligible: false,
        children: {
          [childUuid]: childDetectabilityMetric('error', 'yes'),
        },
      },
    });
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=SpecFirstModel'));
    expect(res.status).toBe(200);
    const body = await res.json();
    const child = body?.modelRecord?.baselines?.[0]?.interventions?.[0];

    expect(child?.name).toBe(childUuid);
    expect(child?.detectability_failed).toBe(true);
    cwdSpy.mockRestore();
  });

  test('GET /api/registry leaves child detectability clear when both comparisons pass', async () => {
    const { fakeWebRoot, modelDir, baselineUuid, childUuid } = createFixtureModel();
    touch(path.join(modelDir, 'runs', baselineUuid, 'data.parquet'));
    touch(path.join(modelDir, 'runs', childUuid, 'data.parquet'));
    writeEligibilityMetrics(modelDir, {
      [baselineUuid]: {
        eligible: true,
        children: {
          [childUuid]: childDetectabilityMetric('yes', 'yes'),
        },
      },
    });
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=SpecFirstModel'));
    expect(res.status).toBe(200);
    const body = await res.json();
    const child = body?.modelRecord?.baselines?.[0]?.interventions?.[0];

    expect(child?.name).toBe(childUuid);
    expect(child?.detectability_failed).toBe(false);
    cwdSpy.mockRestore();
  });

  test('GET /api/registry ignores runtime-only stale run ids not present in specs', async () => {
    const { fakeWebRoot } = createFixtureModel();
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=SpecFirstModel'));
    expect(res.status).toBe(200);
    const body = await res.json();
    const baselineIds = new Set((body?.modelRecord?.baselines ?? []).map((item: any) => String(item?.run_id || '')));

    expect(baselineIds.has('dddddddddddddddddddddddddddddddd')).toBe(false);
    expect(body?.diskRuns).toEqual([]);
    cwdSpy.mockRestore();
  });

  test('GET /api/registry downgrades runtime success to not_run when run data is missing', async () => {
    const { fakeWebRoot, baselineUuid } = createFixtureModel();
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=SpecFirstModel'));
    expect(res.status).toBe(200);
    const body = await res.json();
    const baseline = body?.modelRecord?.baselines?.[0];

    expect(baseline?.run_id).toBe(baselineUuid);
    expect(baseline?.status).toBe('not_run');
    expect(baseline?.stale_reason).toBe('missing_data');
    cwdSpy.mockRestore();
  });

  test('GET /api/registry downgrades runtime success to not_run when the spec hash changed', async () => {
    const { fakeWebRoot, modelDir, baselineUuid } = createFixtureModel();
    touch(path.join(modelDir, 'runs', baselineUuid, 'data.parquet'));
    const recordPath = path.join(modelDir, 'runs', 'model_record.json');
    const record = JSON.parse(fs.readFileSync(recordPath, 'utf8'));
    record[baselineUuid].parameters_hash = 'ffffffffffffffffffffffffffffffff';
    writeJson(recordPath, record);
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=SpecFirstModel'));
    expect(res.status).toBe(200);
    const body = await res.json();
    const baseline = body?.modelRecord?.baselines?.[0];

    expect(baseline?.run_id).toBe(baselineUuid);
    expect(baseline?.status).toBe('not_run');
    expect(baseline?.stale_reason).toBe('hash_mismatch');
    cwdSpy.mockRestore();
  });

  test('GET /api/registry derives runtime state from available run data when model_record.json is missing', async () => {
    const { fakeWebRoot, baselineUuid, childUuid, time0Uuid } = createFixtureModelWithoutRuntimeFile();
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=SpecFirstModel'));
    expect(res.status).toBe(200);
    const body = await res.json();
    const baseline = body?.modelRecord?.baselines?.[0];
    const child = baseline?.interventions?.[0];

    expect(body?.modelRecord?.metadata).toEqual({});
    expect(baseline?.run_id).toBe(baselineUuid);
    expect(baseline?.status).toBe('success');
    expect(child?.name).toBe(childUuid);
    expect(child?.status).toBe('success');
    expect(child?.time0_baseline_uuid).toBe(time0Uuid);
    expect(body?.diskRuns).toEqual([baselineUuid, childUuid]);
    expect(body?.signalDisplayNames).toEqual({ sig_a: 'signal a', sig_b: 'signal b', time: 'time' });
    expect(body?.availableSignals).toEqual(['sig_b', 'sig_a']);
    cwdSpy.mockRestore();
  });

  test('GET /api/registry accepts skipped children from specs and omits them from interventions', async () => {
    const { fakeWebRoot, baselineUuid, childUuid, skippedChildUuid, time0Uuid } = createFixtureModelWithSkippedChild();
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=SkippedChildModel'));
    expect(res.status).toBe(200);
    const body = await res.json();
    const baseline = body?.modelRecord?.baselines?.[0];
    const interventionIds = (baseline?.interventions ?? []).map((item: any) => String(item?.name || ''));

    expect(baseline?.run_id).toBe(baselineUuid);
    expect(interventionIds).toEqual([childUuid]);
    expect(interventionIds.includes(skippedChildUuid)).toBe(false);
    expect(baseline?.interventions?.[0]?.time0_baseline_uuid).toBe(time0Uuid);
    expect(body?.diskRuns).toEqual([baselineUuid, childUuid]);
    cwdSpy.mockRestore();
  });

  test('GET /api/registry rejects skipped children with non-null hash metadata', async () => {
    const { fakeWebRoot, modelDir } = createFixtureModelWithSkippedChild();
    const specsPath = path.join(modelDir, 'model_run_specs.json');
    const specs = JSON.parse(fs.readFileSync(specsPath, 'utf8'));
    specs.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.children.dddddddddddddddddddddddddddddddd.parameter_hash = 'ffffffffffffffffffffffffffffffff';
    writeJson(specsPath, specs);
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=SkippedChildModel'));
    expect(res.status).toBe(500);
    const body = await res.json();

    expect(String(body?.error || '')).toContain(
      "Skipped child 'dddddddddddddddddddddddddddddddd' under baseline 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' must set parameter_hash, time0_baseline_hash, and time0_baseline_uuid to null."
    );
    cwdSpy.mockRestore();
  });

  test('GET /api/registry accepts Python-generated hashes for renamed TransmissionLine parameter keys', async () => {
    const { fakeWebRoot, baselineUuid, childUuid } = createRenamedTransmissionLineHashFixtureModel();
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/registry/route');
    const res = await GET(new Request('http://localhost/api/registry?model=MixedCaseSpecModel'));
    expect(res.status).toBe(200);
    const body = await res.json();
    const baseline = body?.modelRecord?.baselines?.[0];

    expect(baseline?.run_id).toBe(baselineUuid);
    expect(baseline?.interventions?.[0]?.name).toBe(childUuid);
    cwdSpy.mockRestore();
  });
});
