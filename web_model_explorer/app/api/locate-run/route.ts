import { NextResponse } from 'next/server';
import path from 'path';
import { locateRunModel } from '../runDataResolver';

export async function GET(request: Request) {
  const url = new URL(request.url);
  const run = String(url.searchParams.get('run') || '').trim();
  if (!run) {
    return NextResponse.json({ error: "Missing required query param 'run'." }, { status: 400 });
  }

  const repoRoot = path.join(process.cwd(), '..');
  try {
    const located = locateRunModel({ repoRoot, runId: run });
    if (located) return NextResponse.json({ model: located.model, run, found: true });
    return NextResponse.json({ model: null, run, found: false });
  } catch (error) {
    return NextResponse.json({ error: (error as Error).message }, { status: 500 });
  }
}
