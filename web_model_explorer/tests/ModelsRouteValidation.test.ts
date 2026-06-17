import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { beforeEach, describe, expect, test, vi } from 'vitest';

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

  writeJsonl(path.join(modelDir, 'plans', 'policy_a', 'run_nodes.jsonl'), [
    {
      run_id: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      family_id: 'fam_a',
      kind: 'baseline',
      recipe_hash: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      recipe: {
        model: 'ModelA',
        baseline_parameters: { mass: 1 },
        intervention: { parameter: null, time: null, value: null },
      },
    },
  ]);
  writeJsonl(path.join(modelDir, 'plans', 'policy_a', 'run_edges.jsonl'), []);

  return { fakeWebRoot, modelDir };
};

describe('/api/models validation', () => {
  beforeEach(() => {
    vi.resetModules();
  });

  test('returns plan validation warnings when resolved plans are missing', async () => {
    const { fakeWebRoot, modelDir } = createFixture();
    fs.rmSync(path.join(modelDir, 'plans'), { recursive: true, force: true });
    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

    const { GET } = await import('../app/api/models/route');
    const res = await GET();
    const body = await res.json();

    expect(res.status).toBe(200);
    expect(body.models).toEqual(['ModelA']);
    expect(body.validation.ModelA.ok).toBe(false);
    expect(body.validation.ModelA.reasons.join('\n')).toContain('plans/<policy_id>/run_nodes.jsonl');
    cwdSpy.mockRestore();
  });

  test('does not add plan warnings when a resolved plan exists', async () => {
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
