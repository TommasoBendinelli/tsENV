import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import {
  modelDir as resolveModelDir,
  modelsRoot as resolveModelsRoot,
} from '../modelExplorerPaths';
import { locateRunModel } from '../runDataResolver';

const isDirectory = (filePath: string) => {
  try {
    return fs.statSync(filePath).isDirectory();
  } catch {
    return false;
  }
};

const planContainsRun = (planDir: string, runId: string) => {
  const nodesPath = path.join(planDir, 'run_nodes.jsonl');
  if (!fs.existsSync(nodesPath)) return false;
  const needle = `"run_id":"${runId}"`;
  return fs.readFileSync(nodesPath, 'utf8').includes(needle);
};

const locateRunPolicy = (repoRoot: string, model: string, runId: string): string | null => {
  const plansRoot = path.join(resolveModelDir(model, repoRoot), 'plans');
  if (!isDirectory(plansRoot)) return null;
  for (const entry of fs.readdirSync(plansRoot, { withFileTypes: true })) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    if (planContainsRun(path.join(plansRoot, entry.name), runId)) return entry.name;
  }
  return null;
};

const locateRunInPlans = (repoRoot: string, runId: string): { model: string; policy: string | null } | null => {
  const modelsRoot = resolveModelsRoot(repoRoot);
  if (!isDirectory(modelsRoot)) return null;
  for (const modelEntry of fs.readdirSync(modelsRoot, { withFileTypes: true })) {
    if (!modelEntry.isDirectory() || modelEntry.name.startsWith('.')) continue;
    const policy = locateRunPolicy(repoRoot, modelEntry.name, runId);
    if (policy) return { model: modelEntry.name, policy };
  }
  return null;
};

export async function GET(request: Request) {
  const url = new URL(request.url);
  const run = String(url.searchParams.get('run') || '').trim();
  if (!run) {
    return NextResponse.json({ error: "Missing required query param 'run'." }, { status: 400 });
  }

  const repoRoot = path.join(process.cwd(), '..');
  try {
    const located = locateRunModel({ repoRoot, runId: run });
    if (located) {
      return NextResponse.json({
        model: located.model,
        policy: locateRunPolicy(repoRoot, located.model, run),
        run,
        found: true,
      });
    }
    const planLocated = locateRunInPlans(repoRoot, run);
    if (planLocated) return NextResponse.json({ ...planLocated, run, found: true });
    return NextResponse.json({ model: null, run, found: false });
  } catch (error) {
    return NextResponse.json({ error: (error as Error).message }, { status: 500 });
  }
}
