import fs from 'fs';
import path from 'path';
import {
  modelDir,
  modelArtifactPath,
  modelRunsDir,
  modelsRoot,
  tsenvModelDir,
} from './modelExplorerPaths';

export type RunDataSource = 'runs' | 'tsenv';
export type RunDataFormat = 'parquet' | 'csv';

export type ResolvedRunDataFile = {
  model: string;
  runId: string;
  source: RunDataSource;
  format: RunDataFormat;
  filePath: string;
};

export const configuredModelArtifactRoot = (repoRoot: string, model: string) =>
  modelRunsDir(model, repoRoot);

export const configuredModelArtifactPath = (
  repoRoot: string,
  model: string,
  filename: string,
) => modelArtifactPath(model, filename, repoRoot);

const isDirectory = (filePath: string) => {
  try {
    return fs.statSync(filePath).isDirectory();
  } catch {
    return false;
  }
};

const readDirNames = (dirPath: string): string[] => {
  try {
    return fs.readdirSync(dirPath, { withFileTypes: true })
      .filter((entry) => !entry.name.startsWith('.'))
      .map((entry) => entry.name);
  } catch {
    return [];
  }
};

const runIdExistsInPolicy = (planDir: string, runId: string): boolean => {
  const nodesPath = path.join(planDir, 'run_nodes.jsonl');
  if (!fs.existsSync(nodesPath)) return false;
  try {
    const lines = fs.readFileSync(nodesPath, 'utf8').split(/\r?\n/);
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      const payload = JSON.parse(trimmed);
      if (String(payload?.run_id || '').trim() === runId) return true;
    }
  } catch {
    return false;
  }
  return false;
};

const locateRunPolicy = (repoRoot: string, model: string, runId: string): string | null => {
  const plansRoot = path.join(modelDir(model, repoRoot), 'plans');
  const policies = readDirNames(plansRoot).filter((name) =>
    isDirectory(path.join(plansRoot, name))
  );
  for (const policy of policies.sort()) {
    if (runIdExistsInPolicy(path.join(plansRoot, policy), runId)) return policy;
  }
  return null;
};

const addRunArtifacts = (runsDir: string, out: Set<string>) => {
  if (!isDirectory(runsDir)) return;
  const entries = fs.readdirSync(runsDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
    .map((entry) => entry.name);
  for (const runId of entries) {
    const runDir = path.join(runsDir, runId);
    const parquetPath = path.join(runDir, 'data.parquet');
    const csvPath = path.join(runDir, 'data.csv');
    if (fs.existsSync(parquetPath) || fs.existsSync(csvPath)) {
      out.add(runId);
    }
  }
};

const addTsenvDataframes = (dataframesDir: string, out: Set<string>) => {
  if (!isDirectory(dataframesDir)) return;
  const files = fs.readdirSync(dataframesDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && !entry.name.startsWith('.'))
    .map((entry) => entry.name);
  for (const filename of files) {
    if (!filename.toLowerCase().endsWith('.parquet')) continue;
    const runId = filename.slice(0, -'.parquet'.length).trim();
    if (runId) out.add(runId);
  }
};

export function listRunDataIdsForModel(params: {
  repoRoot: string;
  model: string;
}): string[] {
  const { repoRoot, model } = params;
  const out = new Set<string>();
  addRunArtifacts(configuredModelArtifactRoot(repoRoot, model), out);
  addTsenvDataframes(path.join(tsenvModelDir(model, repoRoot), 'dataframes'), out);
  return Array.from(out).sort();
}

export function resolveRunDataFile(params: {
  repoRoot: string;
  model: string;
  runId: string;
}): ResolvedRunDataFile | null {
  const { repoRoot, model } = params;
  const runId = String(params.runId || '').trim();
  if (!runId) return null;

  const runsDir = path.join(configuredModelArtifactRoot(repoRoot, model), runId);
  const parquetPath = path.join(runsDir, 'data.parquet');
  const csvPath = path.join(runsDir, 'data.csv');
  if (fs.existsSync(parquetPath)) {
    return { model, runId, source: 'runs', format: 'parquet', filePath: parquetPath };
  }
  if (fs.existsSync(csvPath)) {
    return { model, runId, source: 'runs', format: 'csv', filePath: csvPath };
  }

  const tsenvParquetPath = path.join(tsenvModelDir(model, repoRoot), 'dataframes', `${runId}.parquet`);
  if (fs.existsSync(tsenvParquetPath)) {
    return { model, runId, source: 'tsenv', format: 'parquet', filePath: tsenvParquetPath };
  }

  return null;
}

export function locateRunModel(params: {
  repoRoot: string;
  runId: string;
}): { model: string; policy: string | null; runId: string } | null {
  const { repoRoot } = params;
  const runId = String(params.runId || '').trim();
  if (!runId) return null;
  const root = modelsRoot(repoRoot);
  const modelNames = readDirNames(root).filter((name) =>
    isDirectory(path.join(root, name))
  );
  for (const model of modelNames) {
    const found = resolveRunDataFile({ repoRoot, model, runId });
    if (found) return { model, policy: locateRunPolicy(repoRoot, model, runId), runId };
  }
  return null;
}
