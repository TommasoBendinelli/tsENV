import { describe, expect, test } from 'vitest';

describe('/api/simulate', () => {
  test('GET returns idle status when no simulation status file exists', async () => {
    const { GET } = await import('../app/api/simulate/route');
    const model = `NoStatusModel${Date.now()}`;
    const res = await GET(new Request(`http://localhost/api/simulate?model=${model}`));

    expect(res.status).toBe(200);
    await expect(res.json()).resolves.toEqual({ status: { status: 'idle' } });
  });
});
