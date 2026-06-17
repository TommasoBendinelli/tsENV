import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { EventEmitter } from 'node:events';
import { describe, expect, test, vi } from 'vitest';

type Spawned = EventEmitter & {
  stdout: EventEmitter;
  stderr: EventEmitter;
  pid?: number;
};

const spawnMock = vi.fn();

vi.mock('../app/api/spawnPython', () => ({
  spawnPython: (...args: any[]) => spawnMock(...args),
}));

const touch = (filePath: string) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, '');
};

describe('/api/data tsENV fallback', () => {
  test('loads run data from tsENV when legacy runs data is absent', async () => {
    vi.resetModules();
    spawnMock.mockReset();

    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'data-route-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    fs.mkdirSync(fakeWebRoot, { recursive: true });
    touch(path.join(tmpRoot, 'env', 'bin', 'python'));
    touch(path.join(tmpRoot, 'tsENV_questions', 'DampedMassBetweenWalls', 'dataframes', 'abc123.parquet'));

    spawnMock.mockImplementationOnce((_cmd: string, _args: string[]) => {
      const proc = new EventEmitter() as Spawned;
      proc.stdout = new EventEmitter();
      proc.stderr = new EventEmitter();
      queueMicrotask(() => {
        proc.stdout.emit(
          'data',
          Buffer.from(
            JSON.stringify({
              columns: ['time', 'x'],
              index: [0, 1],
              data: [[0, 1], [1, 2]],
            }),
            'utf8'
          )
        );
        proc.emit('close', 0);
      });
      return proc;
    });

    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);
    const { GET } = await import('../app/api/data/route');
    const req = new Request('http://localhost/api/data?model=DampedMassBetweenWalls&run=abc123&max_rows=100');
    const res = await GET(req);
    cwdSpy.mockRestore();

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.source).toBe('tsenv');
    expect(body.columns).toEqual(['time', 'x']);
    expect(spawnMock).toHaveBeenCalledTimes(1);

    const args = spawnMock.mock.calls[0][1] as string[];
    expect(args[0]).toBe('-c');
    expect(args[2]).toContain(path.join('tsENV_questions', 'DampedMassBetweenWalls', 'dataframes', 'abc123.parquet'));
    expect(args[3]).toBe('100');
    expect(args[4]).toBe('');
    expect(args[5]).toBe('none');
    expect(args[6]).toBe('0');
    expect(args[7]).toBe('');
  });

  test('returns 404 when run file does not exist in either location', async () => {
    vi.resetModules();
    spawnMock.mockReset();

    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'data-route-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    fs.mkdirSync(fakeWebRoot, { recursive: true });
    touch(path.join(tmpRoot, 'env', 'bin', 'python'));

    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);
    const { GET } = await import('../app/api/data/route');
    const req = new Request('http://localhost/api/data?model=DampedMassBetweenWalls&run=missing_run');
    const res = await GET(req);
    cwdSpy.mockRestore();

    expect(res.status).toBe(404);
    const body = await res.json();
    expect(String(body.error || '')).toContain('tsENV_questions');
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test('passes noise profile and seed through to python loader', async () => {
    vi.resetModules();
    spawnMock.mockReset();

    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'data-route-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    fs.mkdirSync(fakeWebRoot, { recursive: true });
    touch(path.join(tmpRoot, 'env', 'bin', 'python'));
    touch(path.join(tmpRoot, 'models', 'simulink', 'DampedMassBetweenWalls', 'runs', 'abc123', 'data.parquet'));
    touch(path.join(tmpRoot, 'models', 'simulink', 'DampedMassBetweenWalls', 'noise_adder.py'));

    spawnMock.mockImplementationOnce((_cmd: string, _args: string[]) => {
      const proc = new EventEmitter() as Spawned;
      proc.stdout = new EventEmitter();
      proc.stderr = new EventEmitter();
      queueMicrotask(() => {
        proc.stdout.emit(
          'data',
          Buffer.from(
            JSON.stringify({
              columns: ['time', 'x'],
              index: [0],
              data: [[0, 1]],
              signal_analysis: {
                x: 2.5,
              },
            }),
            'utf8'
          )
        );
        proc.emit('close', 0);
      });
      return proc;
    });

    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);
    const { GET } = await import('../app/api/data/route');
    const req = new Request('http://localhost/api/data?model=DampedMassBetweenWalls&run=abc123&noise_profile=high&noise_seed=7');
    const res = await GET(req);
    cwdSpy.mockRestore();

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.signal_analysis).toEqual({
      x: 2.5,
    });
    expect(body.signal_snr).toEqual(body.signal_analysis);
    const args = spawnMock.mock.calls[0][1] as string[];
    expect(args[4]).toContain(path.join('models', 'simulink', 'DampedMassBetweenWalls', 'noise_adder.py'));
    expect(args[5]).toBe('high');
    expect(args[6]).toBe('7');
    expect(args[7]).toBe('');
  });

  test('passes documented plan-edge baseline to python loader for child noise analysis', async () => {
    vi.resetModules();
    spawnMock.mockReset();

    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'data-route-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    const modelRoot = path.join(tmpRoot, 'models', 'simulink', 'DampedMassBetweenWalls');
    fs.mkdirSync(fakeWebRoot, { recursive: true });
    touch(path.join(tmpRoot, 'env', 'bin', 'python'));
    touch(path.join(modelRoot, 'runs', 'child123', 'data.parquet'));
    touch(path.join(modelRoot, 'runs', 'baseline123', 'data.parquet'));
    touch(path.join(modelRoot, 'noise_adder.py'));
    fs.mkdirSync(path.join(modelRoot, 'plans', 'policy_a'), { recursive: true });
    fs.writeFileSync(
      path.join(modelRoot, 'plans', 'policy_a', 'run_edges.jsonl'),
      `${JSON.stringify({
        edge_type: 'intervention_to_time0_baseline',
        source_run_id: 'child123',
        target_run_id: 'baseline123',
      })}\n`
    );

    spawnMock.mockImplementationOnce((_cmd: string, _args: string[]) => {
      const proc = new EventEmitter() as Spawned;
      proc.stdout = new EventEmitter();
      proc.stderr = new EventEmitter();
      queueMicrotask(() => {
        proc.stdout.emit(
          'data',
          Buffer.from(
            JSON.stringify({
              columns: ['time', 'x'],
              index: [0],
              data: [[0, 1]],
            }),
            'utf8'
          )
        );
        proc.emit('close', 0);
      });
      return proc;
    });

    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);
    const { GET } = await import('../app/api/data/route');
    const req = new Request('http://localhost/api/data?model=DampedMassBetweenWalls&run=child123&noise_profile=low&noise_seed=3');
    const res = await GET(req);
    cwdSpy.mockRestore();

    expect(res.status).toBe(200);
    const args = spawnMock.mock.calls[0][1] as string[];
    expect(args[7]).toContain(path.join('models', 'simulink', 'DampedMassBetweenWalls', 'runs', 'baseline123', 'data.parquet'));
  });

  test('passes explicit reference_run to python loader when provided', async () => {
    vi.resetModules();
    spawnMock.mockReset();

    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'data-route-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    const modelRoot = path.join(tmpRoot, 'models', 'simulink', 'DampedMassBetweenWalls');
    fs.mkdirSync(fakeWebRoot, { recursive: true });
    touch(path.join(tmpRoot, 'env', 'bin', 'python'));
    touch(path.join(modelRoot, 'runs', 'child123', 'data.parquet'));
    touch(path.join(modelRoot, 'runs', 'explicit_ref', 'data.parquet'));
    touch(path.join(modelRoot, 'noise_adder.py'));

    spawnMock.mockImplementationOnce((_cmd: string, _args: string[]) => {
      const proc = new EventEmitter() as Spawned;
      proc.stdout = new EventEmitter();
      proc.stderr = new EventEmitter();
      queueMicrotask(() => {
        proc.stdout.emit(
          'data',
          Buffer.from(
            JSON.stringify({
              columns: ['time', 'x'],
              index: [0],
              data: [[0, 1]],
            }),
            'utf8'
          )
        );
        proc.emit('close', 0);
      });
      return proc;
    });

    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);
    const { GET } = await import('../app/api/data/route');
    const req = new Request('http://localhost/api/data?model=DampedMassBetweenWalls&run=child123&reference_run=explicit_ref&noise_profile=low&noise_seed=3');
    const res = await GET(req);
    cwdSpy.mockRestore();

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.reference_run).toBe('explicit_ref');
    const args = spawnMock.mock.calls[0][1] as string[];
    expect(args[7]).toContain(path.join('models', 'simulink', 'DampedMassBetweenWalls', 'runs', 'explicit_ref', 'data.parquet'));
  });

  test('uses bundled tsENV noise_adder.py when loading tsENV data with noise', async () => {
    vi.resetModules();
    spawnMock.mockReset();

    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'data-route-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    fs.mkdirSync(fakeWebRoot, { recursive: true });
    touch(path.join(tmpRoot, 'env', 'bin', 'python'));
    touch(path.join(tmpRoot, 'tsENV_questions', 'DampedMassBetweenWalls', 'dataframes', 'abc123.parquet'));
    touch(path.join(tmpRoot, 'tsENV_questions', 'DampedMassBetweenWalls', 'noise_adder.py'));

    spawnMock.mockImplementationOnce((_cmd: string, _args: string[]) => {
      const proc = new EventEmitter() as Spawned;
      proc.stdout = new EventEmitter();
      proc.stderr = new EventEmitter();
      queueMicrotask(() => {
        proc.stdout.emit(
          'data',
          Buffer.from(
            JSON.stringify({
              columns: ['time', 'x'],
              index: [0],
              data: [[0, 1]],
            }),
            'utf8'
          )
        );
        proc.emit('close', 0);
      });
      return proc;
    });

    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);
    const { GET } = await import('../app/api/data/route');
    const req = new Request('http://localhost/api/data?model=DampedMassBetweenWalls&run=abc123&noise_profile=low&noise_seed=3');
    const res = await GET(req);
    cwdSpy.mockRestore();

    expect(res.status).toBe(200);
    const args = spawnMock.mock.calls[0][1] as string[];
    expect(args[4]).toContain(path.join('tsENV_questions', 'DampedMassBetweenWalls', 'noise_adder.py'));
    expect(args[5]).toBe('low');
    expect(args[6]).toBe('3');
    expect(args[7]).toBe('');
  });

  test('returns 400 for invalid noise profile', async () => {
    vi.resetModules();
    spawnMock.mockReset();

    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'data-route-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    fs.mkdirSync(fakeWebRoot, { recursive: true });
    touch(path.join(tmpRoot, 'env', 'bin', 'python'));
    touch(path.join(tmpRoot, 'models', 'simulink', 'DampedMassBetweenWalls', 'runs', 'abc123', 'data.parquet'));

    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);
    const { GET } = await import('../app/api/data/route');
    const req = new Request('http://localhost/api/data?model=DampedMassBetweenWalls&run=abc123&noise_profile=wild');
    const res = await GET(req);
    cwdSpy.mockRestore();

    expect(res.status).toBe(400);
    expect(spawnMock).not.toHaveBeenCalled();
  });

  test('returns 400 when noise profile is requested but noise_adder.py is missing', async () => {
    vi.resetModules();
    spawnMock.mockReset();

    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'data-route-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    fs.mkdirSync(fakeWebRoot, { recursive: true });
    touch(path.join(tmpRoot, 'env', 'bin', 'python'));
    touch(path.join(tmpRoot, 'models', 'simulink', 'DampedMassBetweenWalls', 'runs', 'abc123', 'data.parquet'));

    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);
    const { GET } = await import('../app/api/data/route');
    const req = new Request('http://localhost/api/data?model=DampedMassBetweenWalls&run=abc123&noise_profile=low&noise_seed=3');
    const res = await GET(req);
    cwdSpy.mockRestore();

    expect(res.status).toBe(400);
    expect(spawnMock).not.toHaveBeenCalled();
  });
});
