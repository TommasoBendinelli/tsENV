import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, test, vi } from 'vitest';

const touch = (filePath: string) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, '');
};

describe('/api/locate-run tsENV fallback', () => {
  test('finds run model using tsENV dataframe when legacy runs folder is absent', async () => {
    vi.resetModules();
    const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'locate-run-'));
    const fakeWebRoot = path.join(tmpRoot, 'web_model_explorer');
    fs.mkdirSync(fakeWebRoot, { recursive: true });
    fs.mkdirSync(path.join(tmpRoot, 'models', 'simulink', 'ModelA'), { recursive: true });
    fs.mkdirSync(path.join(tmpRoot, 'models', 'simulink', 'ModelB'), { recursive: true });
    touch(path.join(tmpRoot, 'tsENV_questions', 'ModelB', 'dataframes', 'child123.parquet'));

    const cwdSpy = vi.spyOn(process, 'cwd').mockReturnValue(fakeWebRoot);
    const { GET } = await import('../app/api/locate-run/route');
    const req = new Request('http://localhost/api/locate-run?run=child123');
    const res = await GET(req);
    cwdSpy.mockRestore();

    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body).toEqual({ model: 'ModelB', policy: null, run: 'child123', found: true });
  });
});
