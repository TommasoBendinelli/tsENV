import { NextResponse } from 'next/server';
import path from 'path';
import fs from 'fs';
import os from 'os';

type SimulationStatus = {
  status: 'idle' | 'running' | 'success' | 'failed';
  model?: string;
  pid?: number | null;
  started_at?: string;
  finished_at?: string;
  code?: number | null;
  stdout?: string;
  stderr?: string;
  error?: string;
};

function isPidRunning(pid: number) {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err: any) {
    if (err && (err.code === 'ESRCH' || err.code === 'EPERM')) return false;
    return false;
  }
}

function trimOutput(value: string, limit = 8000) {
  if (value.length <= limit) return value;
  return value.slice(-limit);
}

const TMP_DIR = path.join(os.tmpdir(), 'web_model_explorer');

function pidPath(model: string) {
  return path.join(TMP_DIR, `simulate_${model}.pid.json`);
}

function logPath(model: string) {
  return path.join(TMP_DIR, `simulate_${model}.log`);
}

function statusPath(model: string) {
  return path.join(TMP_DIR, `simulate_${model}.status.json`);
}

function readJson(filePath: string): any | null {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return null;
  }
}

function readTail(filePath: string, maxBytes = 64 * 1024) {
  try {
    const st = fs.statSync(filePath);
    const start = Math.max(0, st.size - maxBytes);
    const fd = fs.openSync(filePath, 'r');
    try {
      const buf = Buffer.alloc(st.size - start);
      fs.readSync(fd, buf, 0, buf.length, start);
      return buf.toString('utf8');
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    return '';
  }
}

function readStatus(model: string): SimulationStatus {
  const p = statusPath(model);
  const raw = fs.existsSync(p) ? readJson(p) : null;
  if (!raw || typeof raw !== 'object') return { status: 'idle' };
  return raw as SimulationStatus;
}

export async function GET(request: Request): Promise<NextResponse> {
  const { searchParams } = new URL(request.url);
  const model = searchParams.get('model');
  if (!model) return NextResponse.json({ error: 'Model required' }, { status: 400 });

  const status = readStatus(model);
  if (status.status !== 'running') {
    return NextResponse.json({ status });
  }

  const pidInfo = readJson(pidPath(model));
  const pid = Number(pidInfo?.pid);
  if (Number.isFinite(pid) && pid > 0 && isPidRunning(pid)) {
    return NextResponse.json({
      status: { ...status, model, pid } satisfies SimulationStatus,
    });
  }

  // Stale status/pid: mark as failed with log tail as context.
  const finishedAt = new Date().toISOString();
  const tail = trimOutput(readTail(logPath(model), 64 * 1024), 8000);
  const failed: SimulationStatus = {
    status: 'failed',
    model,
    pid: Number.isFinite(pid) ? pid : null,
    started_at: status.started_at,
    finished_at: finishedAt,
    code: status.code ?? null,
    stderr: tail || undefined,
    error: 'Simulation process is not running (stale pid/status).',
  };
  return NextResponse.json({ status: failed });
}
