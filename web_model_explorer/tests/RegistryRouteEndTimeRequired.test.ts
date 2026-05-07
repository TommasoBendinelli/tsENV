import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { beforeEach, expect, test, vi } from 'vitest';
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

const sha256_32_hex = (text: string) =>
  crypto.createHash('sha256').update(text, 'utf8').digest('hex').slice(0, 32);

const computeParametersHash = (parameters: Record<string, unknown>) =>
  sha256_32_hex(canonicalJson(parameters));

const computeChildParametersHash = (
  parentParametersHash: string,
  childParameters: Record<string, unknown>,
  interventionTime: unknown,
) => sha256_32_hex(canonicalJson({
  parent_parameters_hash: parentParametersHash,
  parameters: childParameters,
  intervention_time: interventionTime,
}));

const createFixtureModel = () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'registry-endtime-'));
  const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
  const modelDir = path.join(tmpRoot, 'models', 'simulink', 'EndTimeModel');
  fs.mkdirSync(fakeWebRoot, { recursive: true });
  fs.mkdirSync(path.join(modelDir, 'generated'), { recursive: true });

  writeJson(path.join(modelDir, 'generated', 'metadata.json'), {
    simulink_signals_available: ['sig_a'],
  });
  writeJson(path.join(modelDir, 'experiment_config.json'), {
    exposed_variables: {
      initial_state: {},
      parameters: {},
    },
    sampling_rate_hz: 10,
  });
  writeJson(path.join(modelDir, 'model_run_specs.json'), {
    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: {
      baseline_parameters: {
        x: 0.25,
      },
      baseline_parameters_hash: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
      children: {},
    },
  });
  writeJson(path.join(modelDir, 'runs', 'model_record.json'), {
    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: {
      parameters_hash: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
      run_type: 'baseline',
      class_internal: 'baseline',
      class_agent_facing_name: 'baseline',
      status: 'not_run',
    },
  });

  return { fakeWebRoot };
};

beforeEach(() => {
  vi.resetModules();
});

test('GET /api/registry returns error when end_time_input_s is missing from strict specs', async () => {
  const { fakeWebRoot } = createFixtureModel();
  const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

  const { GET } = await import('../app/api/registry/route');
  const res = await GET(new Request('http://localhost/api/registry?model=EndTimeModel'));
  const body = await res.json();

  expect(res.status).toBe(500);
  expect(String(body?.error || '')).toMatch(/end_time_input_s/i);
  cwdSpy.mockRestore();
});

test('GET /api/registry returns error when time0 fields are missing from strict specs', async () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'registry-missing-time0-'));
  const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
  const modelDir = path.join(tmpRoot, 'models', 'simulink', 'MissingTime0Model');
  fs.mkdirSync(fakeWebRoot, { recursive: true });
  fs.mkdirSync(path.join(modelDir, 'generated'), { recursive: true });

  writeJson(path.join(modelDir, 'generated', 'metadata.json'), {
    simulink_signals_available: ['sig_a'],
  });
  writeJson(path.join(modelDir, 'experiment_config.json'), {
    exposed_variables: {
      initial_state: {},
      parameters: {},
    },
    sampling_rate_hz: 10,
    end_time_input_s: 10,
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
      signal_type: { sig_a: { type: 'continuous' } },
    },
  });
  const interventionTime = 3;
  const baselineParameters = { x: 0.25 };
  const baselineParametersHash = computeParametersHash(baselineParameters);
  const childParameters = { x: 1.5 };
  const childParameterHash = computeChildParametersHash(
    baselineParametersHash,
    childParameters,
    interventionTime
  );
  writeJson(path.join(modelDir, 'model_run_specs.json'), {
    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: {
      baseline_parameters: baselineParameters,
      baseline_parameters_hash: baselineParametersHash,
      children: {
        cccccccccccccccccccccccccccccccc: {
          intervention_time: interventionTime,
          parameters: childParameters,
          parameter_hash: childParameterHash,
        },
      },
    },
  });

  const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);
  const { GET } = await import('../app/api/registry/route');
  const res = await GET(new Request('http://localhost/api/registry?model=MissingTime0Model'));
  const body = await res.json();

  expect(res.status).toBe(500);
  expect(String(body?.error || '')).toMatch(/time0_baseline_(uuid|hash)/i);
  cwdSpy.mockRestore();
});
