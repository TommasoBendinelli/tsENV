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

const writeJsonl = (filePath: string, rows: unknown[]) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, rows.map((row) => JSON.stringify(row)).join('\n') + '\n', 'utf8');
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
    vi.unstubAllEnvs();
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
        "def quantify_noise(clean, noisy, reference):",
        "    return {'global': [], 'local': []}",
        "def add_noise(src, seed=0, noise_level='low', ref=None):",
        "    return src, quantify_noise(src, src, ref)",
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

  test('GET /api/policies and policy-aware registry read resolved run graph plans', async () => {
    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'resolved-plan-route-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    const modelDir = path.join(tmpRoot, 'models', 'simulink', 'PlanModel');
    const policyId = 'production_v1';
    const planDir = path.join(modelDir, 'plans', policyId);
    const baselineId = 'base001';
    const childId = 'child001';
    const time0Id = 'time0001';
    fs.mkdirSync(fakeWebRoot, { recursive: true });
    fs.mkdirSync(path.join(modelDir, 'generated'), { recursive: true });
    fs.mkdirSync(path.join(modelDir, 'runs'), { recursive: true });
    writeJson(path.join(modelDir, 'generated', 'metadata.json'), {
      simulink_signals_available: ['height'],
      simscape_signals_available: [],
    });
    writeJson(path.join(modelDir, 'experiment_config.json'), {
      exposed_variables: { parameters: {}, initial_state: {} },
      sampling_rate_hz: 10,
      end_time_input_s: 5,
      observable_signals: { observable_signals: ['height'] },
    });
    writeJson(path.join(modelDir, 'description_levels.json'), {
      internal_naming_to_agent_facing_signal: { height: 'Height' },
    });
    const baselineRecipe = {
      model: 'PlanModel',
      baseline_parameters: { gravity: 9.81 },
      intervention: { parameter: null, time: null, value: null },
    };
    const childRecipe = {
      model: 'PlanModel',
      baseline_parameters: { gravity: 9.81 },
      intervention: { parameter: 'gravity', time: 2, value: 5 },
    };
    const time0Recipe = {
      model: 'PlanModel',
      baseline_parameters: { gravity: 9.81 },
      intervention: { parameter: 'gravity', time: 0, value: 5 },
    };
    writeJsonl(path.join(planDir, 'run_nodes.jsonl'), [
      {
        run_id: baselineId,
        kind: 'baseline',
        family_id: 'fam_one',
        recipe: baselineRecipe,
        recipe_hash: 'hash_base',
        metadata: { policy_id: policyId, source: 'programmatic', validation_profile: 'paper_selected' },
      },
      {
        run_id: childId,
        kind: 'intervention',
        family_id: 'fam_one',
        recipe: childRecipe,
        recipe_hash: 'hash_child',
        metadata: { policy_id: policyId, source: 'programmatic' },
      },
      {
        run_id: time0Id,
        kind: 'time0_baseline',
        family_id: 'fam_one',
        recipe: time0Recipe,
        recipe_hash: 'hash_time0',
        metadata: { policy_id: policyId, source: 'programmatic' },
      },
    ]);
    writeJsonl(path.join(planDir, 'run_edges.jsonl'), [
      {
        edge_type: 'baseline_to_intervention',
        family_id: 'fam_one',
        source_run_id: baselineId,
        target_run_id: childId,
        metadata: { direction: 'decrease', parameter: 'gravity', intervention_time: 2 },
      },
      {
        edge_type: 'intervention_to_time0_baseline',
        family_id: 'fam_one',
        source_run_id: childId,
        target_run_id: time0Id,
        metadata: { direction: 'decrease', parameter: 'gravity', intervention_time: 2 },
      },
    ]);
    writeJson(path.join(planDir, 'generation_report.json'), { status: 'pass', policy_id: policyId });
    writeJson(path.join(planDir, 'validation_report.json'), { status: 'pass', policy_id: policyId });
    writeJson(path.join(modelDir, 'runs', 'model_record.json'), {
      [baselineId]: { status: 'success', run_type: 'baseline', recipe_hash: 'hash_base' },
      [childId]: { status: 'success', run_type: 'intervention', recipe_hash: 'hash_child' },
      [time0Id]: { status: 'success', run_type: 'time0_baseline', recipe_hash: 'hash_time0' },
    });
    touch(path.join(modelDir, 'runs', baselineId, 'data.parquet'));
    touch(path.join(modelDir, 'runs', childId, 'data.parquet'));
    touch(path.join(modelDir, 'runs', time0Id, 'data.parquet'));
    writeJsonl(path.join(modelDir, 'metrics', policyId, 'cheap_filter_metrics.jsonl'), [
      {
        record_type: 'run',
        policy_id: policyId,
        family_id: 'fam_one',
        run_id: childId,
        kind: 'intervention',
        cheap_filter_pass: true,
      },
      {
        record_type: 'family',
        policy_id: policyId,
        family_id: 'fam_one',
        baseline_run_id: baselineId,
        family_cheap_filter_pass: true,
      },
    ]);
    writeJsonl(path.join(modelDir, 'metrics', policyId, 'surrogate_scores', 'surrogate_v1.jsonl'), [
      {
        record_type: 'surrogate_score',
        policy_id: policyId,
        surrogate_id: 'surrogate_v1',
        run_id: childId,
        family_id: 'fam_one',
        kind: 'intervention',
        noise_level: 'high',
        predicted_label: 'gravity',
        true_label_confidence: 0.98,
        confidence_margin: 0.3,
        surrogate_filter_pass: true,
      },
    ]);
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const policiesRoute = await import('../app/api/policies/route');
    const policiesRes = await policiesRoute.GET(new Request('http://localhost/api/policies?model=PlanModel'));
    expect(await policiesRes.json()).toEqual({ policies: [policyId] });

    const registryRoute = await import('../app/api/registry/route');
    const registryRes = await registryRoute.GET(new Request(`http://localhost/api/registry?model=PlanModel&policy=${policyId}`));
    expect(registryRes.status).toBe(200);
    const body = await registryRes.json();
    const baseline = body.modelRecord.baselines[0];
    expect(body.registryPage).toMatchObject({
      mode: 'page',
      page: 1,
      page_size: 250,
      total_families: 1,
      total_pages: 1,
      has_next: false,
      has_previous: false,
    });
    expect(body.modelRecord.metadata.policy_id).toBe(policyId);
    expect(baseline.run_id).toBe(baselineId);
    expect(baseline.family_id).toBe('fam_one');
    expect(baseline.recipe_hash).toBe('hash_base');
    expect(baseline.eligible).toBe(true);
    expect(baseline.intervention_count).toBe(1);
    expect(baseline.interventions).toEqual([]);

    const familyRes = await registryRoute.GET(new Request(`http://localhost/api/registry?model=PlanModel&policy=${policyId}&family_id=fam_one`));
    expect(familyRes.status).toBe(200);
    const familyBody = await familyRes.json();
    const familyBaseline = familyBody.modelRecord.baselines[0];
    const child = familyBaseline.interventions[0];
    expect(familyBody.registryPage).toMatchObject({ mode: 'family', family_id: 'fam_one' });
    expect(child.name).toBe(childId);
    expect(child.direction).toBe('decrease');
    expect(child.time0_baseline_uuid).toBe(time0Id);
    expect(child.cheap_filter_pass).toBe(true);
    expect(child.surrogate_id).toBe('surrogate_v1');
    expect(child.true_label_confidence).toBe(0.98);

    const runRes = await registryRoute.GET(new Request(`http://localhost/api/registry?model=PlanModel&policy=${policyId}&run=${childId}`));
    expect(runRes.status).toBe(200);
    const runBody = await runRes.json();
    expect(runBody.registryPage).toMatchObject({ mode: 'family', run: childId });
    expect(runBody.modelRecord.baselines[0].run_id).toBe(baselineId);

    const fullRes = await registryRoute.GET(new Request(`http://localhost/api/registry?model=PlanModel&policy=${policyId}&mode=full`));
    expect(fullRes.status).toBe(200);
    const fullBody = await fullRes.json();
    expect(fullBody.registryPage).toMatchObject({ mode: 'full', total_families: 1 });
    expect(fullBody.modelRecord.baselines[0].interventions[0].name).toBe(childId);
    cwdSpy.mockRestore();
  });

  test('GET /api/policies and policy-aware registry honor WEB_MODEL_EXPLORER_MODEL_ARTIFACT_DIR', async () => {
    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'external-artifact-route-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    const artifactDir = path.join(tmpRoot, 'shared', 'BallDrop_v2');
    const policyId = 'ball_drop_10000_families_v1';
    const planDir = path.join(artifactDir, 'plans', policyId);
    const baselineId = 'external_base001';
    fs.mkdirSync(fakeWebRoot, { recursive: true });
    fs.mkdirSync(path.join(artifactDir, 'generated'), { recursive: true });
    writeJson(path.join(artifactDir, 'generated', 'metadata.json'), {
      simulink_signals_available: ['height'],
      simscape_signals_available: [],
    });
    writeJson(path.join(artifactDir, 'experiment_config.json'), {
      exposed_variables: { parameters: {}, initial_state: {} },
      sampling_rate_hz: 10,
      end_time_input_s: 5,
      observable_signals: { observable_signals: ['height'] },
    });
    writeJson(path.join(artifactDir, 'description_levels.json'), {
      internal_naming_to_agent_facing_signal: { height: 'Height' },
    });
    writeJsonl(path.join(planDir, 'run_nodes.jsonl'), [
      {
        run_id: baselineId,
        kind: 'baseline',
        family_id: 'fam_external',
        recipe: {
          model: 'BallDrop',
          baseline_parameters: { gravity: 9.81 },
          intervention: { parameter: null, time: null, value: null },
        },
        recipe_hash: 'hash_external_base',
        metadata: { policy_id: policyId, source: 'programmatic' },
      },
    ]);
    writeJsonl(path.join(planDir, 'run_edges.jsonl'), []);
    writeJson(path.join(planDir, 'generation_report.json'), { status: 'pass', policy_id: policyId });
    writeJson(path.join(planDir, 'validation_report.json'), { status: 'pass', policy_id: policyId });
    writeJson(path.join(artifactDir, 'runs', 'model_record.json'), {
      [baselineId]: { status: 'success', run_type: 'baseline', recipe_hash: 'hash_external_base' },
    });
    touch(path.join(artifactDir, 'runs', baselineId, 'data.parquet'));
    vi.stubEnv('WEB_MODEL_EXPLORER_MODEL_ARTIFACT_DIR', artifactDir);
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const policiesRoute = await import('../app/api/policies/route');
    const policiesRes = await policiesRoute.GET(new Request('http://localhost/api/policies?model=BallDrop'));
    expect(await policiesRes.json()).toEqual({ policies: [policyId] });

    const registryRoute = await import('../app/api/registry/route');
    const registryRes = await registryRoute.GET(new Request(`http://localhost/api/registry?model=BallDrop&policy=${policyId}`));
    expect(registryRes.status).toBe(200);
    const body = await registryRes.json();
    expect(body.modelRecord.metadata.policy_id).toBe(policyId);
    expect(body.modelRecord.baselines[0].run_id).toBe(baselineId);
    cwdSpy.mockRestore();
  });
});
