import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, test, vi } from 'vitest';

const repoPythonPath = () => {
  const candidates = [
    path.resolve(process.cwd(), '..', 'env', 'bin', 'python'),
    path.resolve(process.cwd(), 'env', 'bin', 'python'),
  ];
  const found = candidates.find((candidate) => fs.existsSync(candidate));
  if (!found) throw new Error(`Could not find repo python. Checked: ${candidates.join(', ')}`);
  return found;
};

const writeRepoPythonShim = (tmpRoot: string) => {
  const shimPath = path.join(tmpRoot, 'env', 'bin', 'python');
  fs.mkdirSync(path.dirname(shimPath), { recursive: true });
  fs.writeFileSync(shimPath, `#!/bin/sh\nexec "${repoPythonPath()}" "$@"\n`, 'utf8');
  fs.chmodSync(shimPath, 0o755);
};

const linkSharedPackage = (tmpRoot: string) => {
  const candidates = [
    path.resolve(process.cwd(), '..', 'shared'),
    path.resolve(process.cwd(), 'shared'),
  ];
  const sharedPath = candidates.find((candidate) => fs.existsSync(candidate));
  if (!sharedPath) throw new Error(`Could not find shared package. Checked: ${candidates.join(', ')}`);
  fs.symlinkSync(sharedPath, path.join(tmpRoot, 'shared'), 'dir');
};

const writeCsv = (filePath: string, rows: Array<[number, number]>) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(
    filePath,
    ['time,x', ...rows.map(([time, x]) => `${time},${x}`)].join('\n'),
    'utf8',
  );
};

const writeTwoSignalCsv = (filePath: string, rows: Array<[number, number, number]>) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(
    filePath,
    ['time,x,y', ...rows.map(([time, x, y]) => `${time},${x},${y}`)].join('\n'),
    'utf8',
  );
};

const writeNoiseAdder = (filePath: string) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(
    filePath,
    [
      'NOISE_DICT = {"low": {}, "high": {}}',
      'SNR_THR_DICT = {"low": {"global": [], "local": []}, "high": {"global": [], "local": []}}',
      'def quantify_noise(clean, noisy, reference):',
      '    return {"global": [999.0], "local": [999.0]}',
      'def add_noise(src, seed=0, noise_level="low", ref=None):',
      '    out = src.copy()',
      '    out["x"] = out["x"] + 1.0',
      '    return out, quantify_noise(src, out, ref)',
      '',
    ].join('\n'),
    'utf8',
  );
};

const writeTwoSignalNoiseAdder = (filePath: string) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(
    filePath,
    [
      'NOISE_DICT = {"low": {}, "high": {}}',
      'SNR_THR_DICT = {"low": {"global": [], "local": []}, "high": {"global": [], "local": []}}',
      'def quantify_noise(clean, noisy, reference):',
      '    return {"global": [999.0, 999.0], "local": [999.0, 999.0]}',
      'def add_noise(src, seed=0, noise_level="low", ref=None):',
      '    out = src.copy()',
      '    out["x"] = out["x"] + 1.0',
      '    out["y"] = out["y"] + 1.0',
      '    return out, quantify_noise(src, out, ref)',
      '',
    ].join('\n'),
    'utf8',
  );
};

const writePlanEdge = (modelRoot: string, interventionRunId: string, baselineRunId: string) => {
  fs.mkdirSync(path.join(modelRoot, 'plans', 'policy_a'), { recursive: true });
  fs.writeFileSync(
    path.join(modelRoot, 'plans', 'policy_a', 'run_edges.jsonl'),
    `${JSON.stringify({
      edge_type: 'intervention_to_time0_baseline',
      source_run_id: interventionRunId,
      target_run_id: baselineRunId,
    })}\n`,
    'utf8',
  );
};

describe('/api/data SNR formulation', () => {
  test('reports effect-to-noise SNR against an explicit reference run', async () => {
    vi.resetModules();

    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'data-route-snr-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    const modelRoot = path.join(tmpRoot, 'models', 'simulink', 'ExampleModel');
    writeRepoPythonShim(tmpRoot);
    linkSharedPackage(tmpRoot);
    fs.mkdirSync(fakeWebRoot, { recursive: true });

    writeCsv(path.join(modelRoot, 'runs', 'child123', 'data.csv'), [
      [0, 2],
      [1, 3],
      [2, 4],
      [3, 5],
    ]);
    writeCsv(path.join(modelRoot, 'runs', 'baseline123', 'data.csv'), [
      [0, 1],
      [1, 1],
      [2, 1],
      [3, 1],
    ]);
    writeNoiseAdder(path.join(modelRoot, 'noise_adder.py'));
    writePlanEdge(modelRoot, 'child123', 'baseline123');

    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);
    const { GET } = await import('../app/api/data/route');
    const req = new Request('http://localhost/api/data?model=ExampleModel&run=child123&reference_run=baseline123&noise_profile=low&noise_seed=2');
    const res = await GET(req);
    cwdSpy.mockRestore();

    expect(res.status).toBe(200);
    const body = await res.json();
    const expectedEffectToNoise = 20 * Math.log10(Math.sqrt((1 + 4 + 9 + 16) / 4) / 1);
    const rawSignalToNoise = 20 * Math.log10(Math.sqrt((4 + 9 + 16 + 25) / 4) / 1);

    expect(body.signal_snr).toEqual(body.signal_analysis);
    expect(body.signal_snr.x).toBeCloseTo(expectedEffectToNoise, 10);
    expect(body.signal_snr.x).not.toBeCloseTo(rawSignalToNoise, 4);
    expect(body.signal_snr.x).not.toBe(999);
    expect(body.reference_run).toBe('baseline123');
  });

  test('omits SNR diagnostics when no reference trajectory is available', async () => {
    vi.resetModules();

    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'data-route-snr-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    const modelRoot = path.join(tmpRoot, 'models', 'simulink', 'ExampleModel');
    writeRepoPythonShim(tmpRoot);
    linkSharedPackage(tmpRoot);
    fs.mkdirSync(fakeWebRoot, { recursive: true });

    writeCsv(path.join(modelRoot, 'runs', 'standalone123', 'data.csv'), [
      [0, 2],
      [1, 3],
    ]);
    writeNoiseAdder(path.join(modelRoot, 'noise_adder.py'));

    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);
    const { GET } = await import('../app/api/data/route');
    const req = new Request('http://localhost/api/data?model=ExampleModel&run=standalone123&noise_profile=low&noise_seed=2');
    const res = await GET(req);
    cwdSpy.mockRestore();

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.signal_snr).toBeUndefined();
    expect(body.signal_analysis).toBeUndefined();
  });

  test('reports -inf SNR for zero-effect signals with nonzero noise', async () => {
    vi.resetModules();

    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'data-route-snr-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    const modelRoot = path.join(tmpRoot, 'models', 'simulink', 'ExampleModel');
    writeRepoPythonShim(tmpRoot);
    linkSharedPackage(tmpRoot);
    fs.mkdirSync(fakeWebRoot, { recursive: true });

    writeTwoSignalCsv(path.join(modelRoot, 'runs', 'child123', 'data.csv'), [
      [0, 2, 5],
      [1, 3, 5],
      [2, 4, 5],
      [3, 5, 5],
    ]);
    writeTwoSignalCsv(path.join(modelRoot, 'runs', 'baseline123', 'data.csv'), [
      [0, 1, 5],
      [1, 1, 5],
      [2, 1, 5],
      [3, 1, 5],
    ]);
    writeTwoSignalNoiseAdder(path.join(modelRoot, 'noise_adder.py'));
    writePlanEdge(modelRoot, 'child123', 'baseline123');

    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);
    const { GET } = await import('../app/api/data/route');
    const req = new Request('http://localhost/api/data?model=ExampleModel&run=child123&noise_profile=low&noise_seed=2');
    const res = await GET(req);
    cwdSpy.mockRestore();

    expect(res.status).toBe(200);
    const body = await res.json();

    expect(body.signal_snr).toEqual(body.signal_analysis);
    expect(body.signal_snr.y).toBe('-inf');
  });
});
