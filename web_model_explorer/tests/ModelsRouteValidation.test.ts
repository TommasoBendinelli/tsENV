import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { beforeEach, describe, expect, test, vi } from 'vitest';
import {
  computeChildParametersHash,
  computeParametersHash,
  computeTime0BaselineHash,
} from '../app/api/modelRunSpecHashes';

const writeJson = (filePath: string, payload: unknown) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2), 'utf8');
};

const touch = (filePath: string) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, '', 'utf8');
};

const createFixture = () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'models-route-'));
  const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
  const sharedDir = path.join(tmpRoot, 'shared');
  const modelDir = path.join(tmpRoot, 'models', 'simulink', 'ModelA');
  fs.mkdirSync(fakeWebRoot, { recursive: true });
  fs.mkdirSync(sharedDir, { recursive: true });
  fs.mkdirSync(path.join(modelDir, 'generated'), { recursive: true });
  fs.writeFileSync(
    path.join(sharedDir, 'benchmark_utils.py'),
    'ALLOWED_TSENV_MODELS = ("ModelA", "MissingModel")\n',
    'utf8'
  );
  touch(path.join(modelDir, 'simulink_model_original.mdl'));
  writeJson(path.join(modelDir, 'generated', 'metadata.json'), {});
  writeJson(path.join(modelDir, 'description_levels.json'), {});
  writeJson(path.join(modelDir, 'experiment_config.json'), {});

  const baselineUuid = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
  const childUuid = 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';
  const time0Uuid = 'cccccccccccccccccccccccccccccccc';
  const interventionTime = 2;
  const baselineParameters = { mass: 1 };
  const childParameters = { mass: 4 };
  const baselineHash = computeParametersHash(baselineParameters);
  const childHash = computeChildParametersHash(baselineHash, childParameters, interventionTime);
  writeJson(path.join(modelDir, 'model_run_specs.json'), {
    [baselineUuid]: {
      baseline_parameters: baselineParameters,
      baseline_parameters_hash: baselineHash,
      children: {
        [childUuid]: {
          intervention_time: interventionTime,
          parameters: childParameters,
          parameter_hash: childHash,
          time0_baseline_uuid: time0Uuid,
          time0_baseline_hash: computeTime0BaselineHash(childHash),
        },
      },
    },
  });

  return { fakeWebRoot, modelDir };
};

describe('/api/models validation', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  test('returns stale model_run_specs validation warnings for old hashes', async () => {
    const { fakeWebRoot, modelDir } = createFixture();
    const specsPath = path.join(modelDir, 'model_run_specs.json');
    const specs = JSON.parse(fs.readFileSync(specsPath, 'utf8'));
    specs.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.baseline_parameters.mass = 3;
    writeJson(specsPath, specs);
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/models/route');
    const res = await GET();
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.models).toEqual(['ModelA']);
    expect(body.validation.ModelA.ok).toBe(false);
    expect(body.validation.ModelA.reasons.join('\n')).toContain('model_run_specs.json appears stale');
    cwdSpy.mockRestore();
  });

  test('does not add stale model_run_specs warnings when hashes match', async () => {
    const { fakeWebRoot } = createFixture();
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/models/route');
    const res = await GET();
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.validation.ModelA.ok).toBe(true);
    expect(body.validation.ModelA.reasons).toEqual([]);
    cwdSpy.mockRestore();
  });
});
