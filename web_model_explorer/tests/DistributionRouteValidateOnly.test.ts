import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, expect, test, vi } from 'vitest';

vi.mock('../app/api/sharedSchemaAjv', () => ({
  assertValidAgainstSharedSchema: vi.fn(),
}));

const writeJson = (filePath: string, payload: unknown) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2), 'utf8');
};

const createFixture = () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'distribution-route-'));
  const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
  const modelDir = path.join(tmpRoot, 'models', 'simulink', 'BallDrop');
  fs.mkdirSync(fakeWebRoot, { recursive: true });
  fs.mkdirSync(path.join(tmpRoot, 'shared', 'schemas'), { recursive: true });
  fs.copyFileSync(
    path.join(process.cwd(), '..', 'shared', 'schemas', 'experiment_config.schema.json'),
    path.join(tmpRoot, 'shared', 'schemas', 'experiment_config.schema.json'),
  );

  writeJson(path.join(modelDir, 'experiment_config.json'), {
    exposed_variables: {
      initial_state: {},
      parameters: {
        mass: {
          allowed_intervals: [1.0, 2.0],
          min_srd_distance: 0.3,
          min_abs_dist: 0.0,
          sampling_strategy: 'uniform',
        },
      },
    },
    sampling_rate_hz: 20.0,
    end_time_input_s: 10.0,
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
      observable_signals: ['Position', 'Velocity', 'Hard_Stop_f'],
      signal_type: {
        Position: { type: 'continuous' },
        Velocity: { type: 'continuous' },
        Hard_Stop_f: { type: 'impulse_like' },
      },
    },
  });

  return { fakeWebRoot };
};

afterEach(() => {
  vi.restoreAllMocks();
});

test('GET /api/distribution accepts observable signal_type', async () => {
  vi.resetModules();
  const { fakeWebRoot } = createFixture();
  vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

  const { GET } = await import('../app/api/distribution/route');
  const res = await GET(new Request('http://localhost/api/distribution?model=BallDrop'));

  expect(res.status).toBe(200);
  await expect(res.json()).resolves.toMatchObject({
    distribution: {
      observable_signals: {
        signal_type: {
          Position: { type: 'continuous' },
          Velocity: { type: 'continuous' },
          Hard_Stop_f: { type: 'impulse_like' },
        },
      },
    },
  });
});
