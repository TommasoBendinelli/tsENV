import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { beforeEach, expect, test, vi } from 'vitest';

vi.mock('../app/api/registry/modelSchema', async () => {
  const actual = await vi.importActual<any>('../app/api/registry/modelSchema');
  return {
    ...actual,
    normalizeModelRecord: (payload: unknown) => payload,
  };
});

vi.mock('../app/api/sharedSchemaAjv', () => ({
  assertValidAgainstSharedSchema: (_schema: string, payload: unknown) => payload,
}));

const writeJson = (filePath: string, payload: unknown) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2), 'utf8');
};

const writeJsonl = (filePath: string, rows: unknown[]) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, rows.map((row) => JSON.stringify(row)).join('\n') + '\n', 'utf8');
};

beforeEach(() => {
  vi.resetModules();
});

test('GET /api/registry returns error when end_time_input_s is missing from experiment_config', async () => {
  const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'registry-endtime-'));
  const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
  const modelDir = path.join(tmpRoot, 'models', 'simulink', 'EndTimeModel');
  const planDir = path.join(modelDir, 'plans', 'policy_a');
  fs.mkdirSync(fakeWebRoot, { recursive: true });
  fs.mkdirSync(path.join(modelDir, 'generated'), { recursive: true });
  writeJson(path.join(modelDir, 'generated', 'metadata.json'), {
    simulink_signals_available: ['sig_a'],
  });
  writeJson(path.join(modelDir, 'experiment_config.json'), {
    exposed_variables: { initial_state: {}, parameters: {} },
    sampling_rate_hz: 10,
    observable_signals: { observable_signals: ['sig_a'] },
  });
  writeJsonl(path.join(planDir, 'run_nodes.jsonl'), [
    {
      run_id: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      family_id: 'fam_a',
      kind: 'baseline',
      recipe_hash: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
      recipe: {
        model: 'EndTimeModel',
        baseline_parameters: { x: 1 },
        intervention: { parameter: null, time: null, value: null },
      },
    },
  ]);
  writeJsonl(path.join(planDir, 'run_edges.jsonl'), []);
  const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);

  const { GET } = await import('../app/api/registry/route');
  const res = await GET(new Request('http://localhost/api/registry?model=EndTimeModel&policy=policy_a'));
  const body = await res.json();

  expect(res.status).toBe(500);
  expect(String(body?.error || '')).toMatch(/end_time_input_s/i);
  cwdSpy.mockRestore();
});
