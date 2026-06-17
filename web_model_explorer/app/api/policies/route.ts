import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { modelDir as resolveModelDir } from '../modelExplorerPaths';

const asString = (value: unknown) => String(value ?? '').trim();

const isDirectory = (filePath: string) => {
  try {
    return fs.statSync(filePath).isDirectory();
  } catch {
    return false;
  }
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const model = asString(searchParams.get('model'));
  if (!model) return NextResponse.json({ error: 'Model required' }, { status: 400 });

  const plansRoot = path.join(resolveModelDir(model), 'plans');
  try {
    if (!isDirectory(plansRoot)) return NextResponse.json({ policies: [] });
    const policies = fs.readdirSync(plansRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
      .filter((entry) => (
        fs.existsSync(path.join(plansRoot, entry.name, 'run_nodes.jsonl'))
        && fs.existsSync(path.join(plansRoot, entry.name, 'run_edges.jsonl'))
      ))
      .map((entry) => entry.name)
      .sort();
    return NextResponse.json({ policies });
  } catch (error) {
    return NextResponse.json({ error: (error as Error).message }, { status: 500 });
  }
}
