import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import {
  modelDir,
  modelRunsDir,
  repoRoot as resolveRepoRoot,
  tsenvModelDir,
} from '../modelExplorerPaths';
import { spawnPython } from '../spawnPython';
import { resolveRunDataFile, type RunDataSource } from '../runDataResolver';

type DataPayload = {
  columns: string[];
  index: number[];
  data: Array<Array<string | number | boolean | null>>;
  signal_snr?: Record<string, string | number | boolean | null>;
  signal_analysis?: Record<string, string | number | boolean | null>;
};

const DEFAULT_MAX_ROWS = 50_000;
const HARD_MAX_ROWS = 200_000;

const parsePositiveInt = (value: string | null, fallback: number): number => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  const asInt = Math.trunc(parsed);
  if (asInt <= 0) return fallback;
  return Math.min(asInt, HARD_MAX_ROWS);
};

const parseNoiseProfile = (value: string | null): 'none' | 'low' | 'high' | null => {
  const normalized = String(value || '').trim().toLowerCase();
  if (!normalized) return 'none';
  if (normalized === 'none' || normalized === 'low' || normalized === 'high') return normalized;
  return null;
};

const parseNoiseSeed = (value: string | null): number => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 0;
  return Math.max(0, Math.trunc(parsed));
};

const resolveNoiseAdderPath = (params: {
  repoRoot: string;
  model: string;
  source: 'runs' | 'tsenv';
}): string | null => {
  const { model, source } = params;
  const candidates =
    source === 'tsenv'
      ? [path.join(tsenvModelDir(model), 'noise_adder.py')]
      : [path.join(modelDir(model), 'noise_adder.py')];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return null;
};

const readJsonObject = (filePath: string): Record<string, unknown> => {
  try {
    const parsed = JSON.parse(fs.readFileSync(filePath, 'utf8')) as unknown;
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : {};
  } catch {
    return {};
  }
};

const resolveRunDataFileForSource = (params: {
  repoRoot: string;
  model: string;
  runId: string;
  source: RunDataSource;
}) => {
  const { repoRoot, model, runId, source } = params;
  if (source === 'tsenv') {
    const tsenvPath = path.join(tsenvModelDir(model), 'dataframes', `${runId}.parquet`);
    if (fs.existsSync(tsenvPath)) {
      return { model, runId, source, format: 'parquet' as const, filePath: tsenvPath };
    }
  } else {
    const runDir = path.join(modelRunsDir(model), runId);
    const parquetPath = path.join(runDir, 'data.parquet');
    const csvPath = path.join(runDir, 'data.csv');
    if (fs.existsSync(parquetPath)) {
      return { model, runId, source, format: 'parquet' as const, filePath: parquetPath };
    }
    if (fs.existsSync(csvPath)) {
      return { model, runId, source, format: 'csv' as const, filePath: csvPath };
    }
  }
  return resolveRunDataFile({ repoRoot, model, runId });
};

const readJsonlObjects = (filePath: string): Array<Record<string, unknown>> => {
  try {
    return fs.readFileSync(filePath, 'utf8')
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line) as unknown)
      .filter((value): value is Record<string, unknown> =>
        Boolean(value) && typeof value === 'object' && !Array.isArray(value)
      );
  } catch {
    return [];
  }
};

const baselineUuidFromPlanEdges = (params: {
  repoRoot: string;
  model: string;
  runId: string;
}): string | null => {
  const plansRoot = path.join(modelDir(params.model), 'plans');
  let policies: string[];
  try {
    policies = fs.readdirSync(plansRoot, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
      .map((entry) => entry.name)
      .sort();
  } catch {
    return null;
  }

  for (const policy of policies) {
    const edges = readJsonlObjects(path.join(plansRoot, policy, 'run_edges.jsonl'));
    for (const edge of edges) {
      const edgeType = String(edge.edge_type || '').trim();
      const sourceRunId = String(edge.source_run_id || '').trim();
      const targetRunId = String(edge.target_run_id || '').trim();
      if (
        edgeType === 'intervention_to_time0_baseline'
        && sourceRunId === params.runId
        && targetRunId
      ) {
        return targetRunId;
      }
    }
  }
  return null;
};

const baselineUuidFromSampleManifest = (params: {
  repoRoot: string;
  model: string;
  runId: string;
}): string | null => {
  const payload = readJsonObject(path.join(tsenvModelDir(params.model), 'sample_manifest.json'));
  for (const rawEntries of Object.values(payload)) {
    if (!Array.isArray(rawEntries)) continue;
    for (const rawEntry of rawEntries) {
      if (!rawEntry || typeof rawEntry !== 'object' || Array.isArray(rawEntry)) continue;
      const entry = rawEntry as Record<string, unknown>;
      for (const [samplesKey, baselinesKey] of [
        ['train_samples', 'train_samples_baselines'],
        ['test_samples', 'test_samples_baselines'],
      ] as const) {
        const samples = entry[samplesKey];
        const baselines = entry[baselinesKey];
        if (!Array.isArray(samples) || !Array.isArray(baselines)) continue;
        for (let idx = 0; idx < samples.length; idx += 1) {
          if (String(samples[idx] || '').trim() !== params.runId || idx >= baselines.length) continue;
          const baseline = String(baselines[idx] || '').trim();
          return baseline || null;
        }
      }
    }
  }
  return null;
};

const resolveBaselineDataFile = (params: {
  repoRoot: string;
  model: string;
  runId: string;
  source: RunDataSource;
}) => {
  const baselineId =
    params.source === 'tsenv'
      ? baselineUuidFromSampleManifest(params) || baselineUuidFromPlanEdges(params)
      : baselineUuidFromPlanEdges(params);
  if (!baselineId || baselineId === params.runId) return null;
  return resolveRunDataFileForSource({
    repoRoot: params.repoRoot,
    model: params.model,
    runId: baselineId,
    source: params.source,
  });
};

const resolveReferenceDataFile = (params: {
  repoRoot: string;
  model: string;
  runId: string;
  referenceRunId: string;
  source: RunDataSource;
}) => {
  const referenceRunId = String(params.referenceRunId || '').trim();
  if (!referenceRunId || referenceRunId === params.runId) return null;
  return resolveRunDataFileForSource({
    repoRoot: params.repoRoot,
    model: params.model,
    runId: referenceRunId,
    source: params.source,
  });
};

const PY_READ_SCRIPT = [
  'import json',
  'import math',
  'import sys',
  'from pathlib import Path',
  'import importlib.util',
  '',
  'import numpy as np',
  'import pandas as pd',
  'from shared.noise_analysis import quantify_analysis',
  '',
  'file_path = Path(sys.argv[1])',
  'max_rows = int(sys.argv[2])',
  'noise_adder_path = Path(sys.argv[3]) if sys.argv[3] else None',
  'noise_profile = str(sys.argv[4] or "none").strip().lower()',
  'noise_seed = int(sys.argv[5])',
  'baseline_path = Path(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None',
  'baseline_df = None',
  '',
  "suffix = file_path.suffix.lower()",
  "if suffix == '.parquet':",
  '    df = pd.read_parquet(file_path)',
  "elif suffix == '.csv':",
  '    df = pd.read_csv(file_path)',
  'else:',
  "    raise RuntimeError(f'Unsupported data format: {file_path}')",
  'clean_df = df.copy()',
  '',
  'if noise_profile != "none":',
  '    if noise_adder_path is None or not noise_adder_path.exists():',
  "        raise RuntimeError(f'Noise profile {noise_profile!r} requested but noise_adder.py was not found.')",
  '    spec = importlib.util.spec_from_file_location("web_model_explorer_noise_adder", noise_adder_path)',
  '    if spec is None or spec.loader is None:',
  "        raise RuntimeError(f'Failed to load noise adder from {noise_adder_path}.')",
  '    module = importlib.util.module_from_spec(spec)',
  '    spec.loader.exec_module(module)',
  '    add_noise = getattr(module, "add_noise", None)',
  '    if not callable(add_noise):',
  "        raise RuntimeError(f'noise_adder.py at {noise_adder_path} does not export add_noise(src, seed, noise_level, ref).')",
  '    if baseline_path is not None:',
  "        baseline_suffix = baseline_path.suffix.lower()",
  "        if baseline_suffix == '.parquet':",
  '            baseline_df = pd.read_parquet(baseline_path)',
  "        elif baseline_suffix == '.csv':",
  '            baseline_df = pd.read_csv(baseline_path)',
  '        else:',
  "            raise RuntimeError(f'Unsupported baseline data format: {baseline_path}')",
  '    result = add_noise(clean_df.copy(), seed=noise_seed, noise_level=noise_profile, ref=baseline_df)',
  '    if not isinstance(result, tuple) or len(result) != 2:',
  "        raise RuntimeError('noise_adder.add_noise must return (pandas.DataFrame, noise_analysis).')",
  '    df, _noise_adder_analysis = result',
  '    if not isinstance(df, pd.DataFrame):',
  "        raise RuntimeError('noise_adder.add_noise first return value must be a pandas DataFrame.')",
  '    if not isinstance(_noise_adder_analysis, dict):',
  "        raise RuntimeError('noise_adder.add_noise second return value must be a noise_analysis dict.')",
  '',
  'row_index = np.arange(len(df), dtype=int)',
  'if max_rows > 0 and len(df) > max_rows:',
  '    keep = np.linspace(0, len(df) - 1, num=max_rows, dtype=int)',
  '    keep = np.unique(keep)',
  '    df = df.iloc[keep]',
  '    clean_df = clean_df.iloc[keep]',
  '    if baseline_df is not None:',
  '        baseline_df = baseline_df.iloc[keep]',
  '    row_index = row_index[keep]',
  '',
  'def to_jsonable(value):',
  '    if value is None:',
  '        return None',
  '    try:',
  '        if pd.isna(value):',
  '            return None',
  '    except Exception:',
  '        pass',
  '',
  '    if isinstance(value, np.integer):',
  '        return int(value)',
  '    if isinstance(value, np.floating):',
  '        numeric = float(value)',
  '        if not math.isfinite(numeric):',
  '            return None',
  '        return numeric',
  '    if isinstance(value, np.bool_):',
  '        return bool(value)',
  '    if isinstance(value, pd.Timestamp):',
  '        return value.isoformat()',
  '',
  '    if isinstance(value, float):',
  '        return value if math.isfinite(value) else None',
  '    if isinstance(value, (str, int, bool)):',
  '        return value',
  '    if hasattr(value, "isoformat"):',
  '        try:',
  '            return value.isoformat()',
  '        except Exception:',
  '            pass',
  '    if isinstance(value, (bytes, bytearray)):',
  '        return value.decode("utf-8", errors="replace")',
  '    return str(value)',
  '',
  'columns = [str(col) for col in list(df.columns)]',
  'signal_columns = [col for col in columns if col.strip().lower() not in {"time", "timestamp", "index", "idx"}]',
  '',
  'def time_column_name(frame):',
  '    if frame is None or not hasattr(frame, "columns"):',
  '        return None',
  '    for col in [str(c).strip() for c in list(frame.columns)]:',
  '        if col.lower() in {"time", "timestamp"}:',
  '            return col',
  '    return None',
  '',
  'def build_signal_analysis():',
  '    out = {}',
  '    if noise_profile == "none" or baseline_df is None:',
  '        return out',
  '    analysis = quantify_analysis(clean_df, df, baseline_df)',
  '    global_values = analysis.get("global", []) if isinstance(analysis, dict) else []',
  '    def put(signal, value):',
  '        signal_key = str(signal or "").strip()',
  '        if not signal_key:',
  '            return',
  '        converted = to_jsonable(value)',
  '        if converted is None:',
  '            return',
  '        out[signal_key] = converted',
  '    for signal, value in zip(signal_columns, global_values):',
  '        put(signal, value)',
  '    return out',
  '',
  'rows = []',
  'for row in df.itertuples(index=False, name=None):',
  '    rows.append([to_jsonable(value) for value in row])',
  'signal_analysis = build_signal_analysis()',
  '',
  'payload = {',
  '    "columns": columns,',
  '    "index": [int(i) for i in row_index.tolist()],',
  '    "data": rows,',
  '}',
  'if signal_analysis:',
  '    payload["signal_snr"] = signal_analysis',
  '    payload["signal_analysis"] = signal_analysis',
  'sys.stdout.write(json.dumps(payload))',
  '',
].join('\n');

const loadDataWithPython = async (params: {
  repoRoot: string;
  filePath: string;
  maxRows: number;
  noiseAdderPath: string | null;
  noiseProfile: 'none' | 'low' | 'high';
  noiseSeed: number;
  baselinePath: string | null;
}): Promise<DataPayload> => {
  const { repoRoot, filePath, maxRows, noiseAdderPath, noiseProfile, noiseSeed, baselinePath } = params;
  const pythonPath = path.join(repoRoot, 'env', 'bin', 'python');
  if (!fs.existsSync(pythonPath)) {
    throw new Error(`Python executable not found at ${pythonPath}`);
  }

  return new Promise((resolve, reject) => {
    const pyProc = spawnPython(
      pythonPath,
      [
        '-c',
        PY_READ_SCRIPT,
        filePath,
        String(maxRows),
        noiseAdderPath || '',
        noiseProfile,
        String(noiseSeed),
        baselinePath || '',
      ],
      {
        cwd: repoRoot,
        env: {
          ...process.env,
          PYTHONPATH: [repoRoot, process.env.PYTHONPATH].filter(Boolean).join(path.delimiter),
        },
      }
    );

    let stdout = '';
    let stderr = '';
    pyProc.stdout.on('data', (chunk) => {
      stdout += String(chunk);
    });
    pyProc.stderr.on('data', (chunk) => {
      stderr += String(chunk);
    });
    pyProc.on('error', (error) => reject(error));
    pyProc.on('close', (code) => {
      if (code !== 0) {
        reject(
          new Error(
            `Python data load failed (exit ${String(code)}): ${stderr.trim() || 'no stderr output'}`
          )
        );
        return;
      }
      try {
        const parsed = JSON.parse(stdout) as DataPayload;
        if (!Array.isArray(parsed?.columns) || !Array.isArray(parsed?.index) || !Array.isArray(parsed?.data)) {
          throw new Error('Invalid data payload shape from python loader.');
        }
        resolve(parsed);
      } catch (error) {
        reject(new Error(`Failed to parse python data output: ${(error as Error).message}`));
      }
    });
  });
};

export async function GET(request: Request): Promise<NextResponse> {
  const { searchParams } = new URL(request.url);
  const model = String(searchParams.get('model') || '').trim();
  const run = String(searchParams.get('run') || '').trim();
  if (!model) {
    return NextResponse.json({ error: "Missing required query param 'model'." }, { status: 400 });
  }
  if (!run) {
    return NextResponse.json({ error: "Missing required query param 'run'." }, { status: 400 });
  }

  const maxRows = parsePositiveInt(searchParams.get('max_rows'), DEFAULT_MAX_ROWS);
  const noiseProfile = parseNoiseProfile(searchParams.get('noise_profile'));
  if (!noiseProfile) {
    return NextResponse.json({ error: "Invalid 'noise_profile'. Expected 'none', 'low', or 'high'." }, { status: 400 });
  }
  const noiseSeed = parseNoiseSeed(searchParams.get('noise_seed'));
  const referenceRun = String(searchParams.get('reference_run') || '').trim();
  const repoRoot = resolveRepoRoot();
  const resolved = resolveRunDataFile({ repoRoot, model, runId: run });
  if (!resolved) {
    const legacyPath = path.join(modelRunsDir(model), run);
    const tsenvPath = path.join(tsenvModelDir(model), 'dataframes', `${run}.parquet`);
    return NextResponse.json(
      {
        error: `Run '${run}' not found for model '${model}'. Checked '${legacyPath}' (data.parquet/data.csv) and '${tsenvPath}'.`,
      },
      { status: 404 }
    );
  }
  const noiseAdderPath = resolveNoiseAdderPath({ repoRoot, model, source: resolved.source });
  const explicitReferenceDataFile = referenceRun
    ? resolveReferenceDataFile({
        repoRoot,
        model,
        runId: run,
        referenceRunId: referenceRun,
        source: resolved.source,
      })
    : null;
  if (noiseProfile !== 'none' && referenceRun && !explicitReferenceDataFile) {
    return NextResponse.json(
      { error: `Reference run '${referenceRun}' not found for model '${model}'.` },
      { status: 404 }
    );
  }
  const baselineDataFile = explicitReferenceDataFile ?? resolveBaselineDataFile({
    repoRoot,
    model,
    runId: run,
    source: resolved.source,
  });
  if (noiseProfile !== 'none' && !noiseAdderPath) {
    return NextResponse.json(
      { error: `Noise profile '${noiseProfile}' is not available for model '${model}' (missing noise_adder.py).` },
      { status: 400 }
    );
  }

  try {
    const loaded = await loadDataWithPython({
      repoRoot,
      filePath: resolved.filePath,
      maxRows,
      noiseAdderPath: noiseProfile === 'none' ? null : noiseAdderPath,
      noiseProfile,
      noiseSeed,
      baselinePath: noiseProfile === 'none' ? null : baselineDataFile?.filePath ?? null,
    });
    const signalSnr = loaded.signal_snr ?? loaded.signal_analysis;
    return NextResponse.json({
      ...loaded,
      ...(signalSnr ? { signal_snr: signalSnr, signal_analysis: signalSnr } : {}),
      source: resolved.source,
      format: resolved.format,
      noise_profile: noiseProfile,
      noise_seed: noiseSeed,
      reference_run: noiseProfile === 'none' ? null : explicitReferenceDataFile?.runId ?? null,
      run,
      model,
    });
  } catch (error) {
    return NextResponse.json({ error: (error as Error).message }, { status: 500 });
  }
}
