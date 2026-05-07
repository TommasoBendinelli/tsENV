import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, test, vi } from 'vitest';
import {
  listRunDataIdsForModel,
  locateRunModel,
  resolveRunDataFile,
} from '../app/api/runDataResolver';

const touch = (filePath: string) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, '');
};

describe('runDataResolver', () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  test('listRunDataIdsForModel merges canonical runs and tsENV dataframes', () => {
    const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'run-resolver-'));
    const model = 'DampedMassBetweenWalls';
    touch(path.join(repoRoot, 'models', 'simulink', model, 'runs', 'run_a', 'data.parquet'));
    touch(path.join(repoRoot, 'models', 'simulink', model, 'runs', 'run_b', 'data.csv'));
    touch(path.join(repoRoot, 'tsENV_questions', model, 'dataframes', 'tsenv_a.parquet'));
    touch(path.join(repoRoot, 'tsENV_questions', model, 'dataframes', 'run_a.parquet'));

    const runIds = listRunDataIdsForModel({ repoRoot, model });
    expect(runIds).toEqual(['run_a', 'run_b', 'tsenv_a']);
  });

  test('resolveRunDataFile prefers canonical runs over tsENV dataframes', () => {
    const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'run-resolver-'));
    const model = 'DampedMassBetweenWalls';
    const runId = 'same_run';
    touch(path.join(repoRoot, 'models', 'simulink', model, 'runs', runId, 'data.parquet'));
    touch(path.join(repoRoot, 'tsENV_questions', model, 'dataframes', `${runId}.parquet`));

    const resolved = resolveRunDataFile({ repoRoot, model, runId });
    expect(resolved).toBeTruthy();
    expect(resolved?.source).toBe('runs');
    expect(resolved?.format).toBe('parquet');
  });

  test('resolveRunDataFile loads documented tsENV dataframes when no run artifact exists', () => {
    const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'run-resolver-'));
    const model = 'DampedMassBetweenWalls';
    const runId = 'only_tsenv';
    touch(path.join(repoRoot, 'tsENV_questions', model, 'dataframes', `${runId}.parquet`));

    const resolved = resolveRunDataFile({ repoRoot, model, runId });
    expect(resolved).toBeTruthy();
    expect(resolved?.source).toBe('tsenv');
    expect(resolved?.format).toBe('parquet');
  });

  test('locateRunModel discovers models through documented tsENV dataframes', () => {
    const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'run-resolver-'));
    touch(path.join(repoRoot, 'models', 'simulink', 'ModelA', 'model_record.json'));
    touch(path.join(repoRoot, 'models', 'simulink', 'ModelB', 'model_record.json'));
    touch(path.join(repoRoot, 'tsENV_questions', 'ModelB', 'dataframes', 'abc123.parquet'));

    const located = locateRunModel({ repoRoot, runId: 'abc123' });
    expect(located).toEqual({ model: 'ModelB', runId: 'abc123' });
  });

  test('resolveRunDataFile honors WEB_MODEL_EXPLORER_RUNS_DIR_NAME', () => {
    vi.stubEnv('WEB_MODEL_EXPLORER_RUNS_DIR_NAME', 'runs_7161');
    const repoRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'run-resolver-'));
    const model = 'DampedMassBetweenWalls';
    const runId = 'custom_runs_dir';
    touch(path.join(repoRoot, 'models', 'simulink', model, 'runs_7161', runId, 'data.parquet'));

    const resolved = resolveRunDataFile({ repoRoot, model, runId });
    expect(resolved).toBeTruthy();
    expect(resolved?.source).toBe('runs');
    expect(resolved?.filePath).toContain(path.join(model, 'runs_7161', runId, 'data.parquet'));
  });
});
