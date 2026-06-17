import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import {
  modelDir as resolveModelDir,
  modelsRoot as resolveModelsRoot,
  repoModelDir,
  repoRoot as resolveRepoRoot,
} from '../modelExplorerPaths';

type ModelValidationResult = {
  ok: boolean;
  reasons: string[];
};

const readAllowedModels = () => {
  const configPath = path.join(process.cwd(), '..', 'shared', 'benchmark_utils.py');
  const source = fs.readFileSync(configPath, 'utf8');
  const match = source.match(/ALLOWED_TSENV_MODELS\s*=\s*\(([\s\S]*?)\)/m);
  if (!match) {
    throw new Error('ALLOWED_TSENV_MODELS not found in shared/benchmark_utils.py');
  }
  return Array.from(match[1].matchAll(/"([^"]+)"/g)).map((item) => item[1]);
};

export async function GET() {
  const repoRoot = resolveRepoRoot();
  const modelsRoot = resolveModelsRoot(repoRoot);
  try {
    const categories = readAllowedModels().filter((model) => {
      const modelDir = repoModelDir(model, repoRoot);
      return fs.existsSync(modelDir) && fs.statSync(modelDir).isDirectory();
    });

    const validation: Record<string, ModelValidationResult> = {};
    for (const model of categories) {
      const modelDir = resolveModelDir(model, repoRoot);
      const hasExperimentConfig = fs.existsSync(path.join(modelDir, 'experiment_config.json'));
      const plansRoot = path.join(modelDir, 'plans');
      const requiredPaths: Array<{ rel: string; reason: string }> = [
        {
          rel: 'simulink_model_original.mdl',
          reason: 'Missing simulink_model_original.mdl (required by metadata/simulation workflows).',
        },
        {
          rel: path.join('generated', 'metadata.json'),
          reason: 'Missing generated/metadata.json (run workflows/simulate/build_metadata.py to generate it).',
        },
        {
          rel: 'description_levels.json',
          reason: 'Missing description_levels.json (signal display metadata).',
        },
      ];

      const reasons: string[] = [];
      if (!hasExperimentConfig) {
        reasons.push('Missing experiment_config.json (experiment configuration).');
      }
      if (!fs.existsSync(plansRoot) || !fs.statSync(plansRoot).isDirectory()) {
        reasons.push('Missing plans/<policy_id>/run_nodes.jsonl and run_edges.jsonl (resolved policy topology).');
      } else {
        const planIds = fs.readdirSync(plansRoot, { withFileTypes: true })
          .filter((entry) => entry.isDirectory() && !entry.name.startsWith('.'))
          .map((entry) => entry.name);
        const hasResolvedPlan = planIds.some((policyId) => (
          fs.existsSync(path.join(plansRoot, policyId, 'run_nodes.jsonl'))
          && fs.existsSync(path.join(plansRoot, policyId, 'run_edges.jsonl'))
        ));
        if (!hasResolvedPlan) {
          reasons.push('Missing plans/<policy_id>/run_nodes.jsonl and run_edges.jsonl (resolved policy topology).');
        }
      }
      for (const item of requiredPaths) {
        const fullPath = path.join(modelDir, item.rel);
        if (!fs.existsSync(fullPath)) reasons.push(item.reason);
      }

      validation[model] = { ok: reasons.length === 0, reasons };
    }

    return NextResponse.json({ models: categories, validation });
  } catch (error) {
    return NextResponse.json({ error: (error as Error).message }, { status: 500 });
  }
}
