import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { collectStaleModelRunSpecReasons } from '../modelRunSpecHashes';

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
  const modelsRoot = path.join(process.cwd(), '..', 'models', 'simulink');
  try {
    const categories = readAllowedModels().filter((model) => {
      const modelDir = path.join(modelsRoot, model);
      return fs.existsSync(modelDir) && fs.statSync(modelDir).isDirectory();
    });

    const validation: Record<string, ModelValidationResult> = {};
    for (const model of categories) {
      const modelDir = path.join(modelsRoot, model);
      const hasExperimentConfig = fs.existsSync(path.join(modelDir, 'experiment_config.json'));
      const specsPath = path.join(modelDir, 'model_run_specs.json');
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
      if (!fs.existsSync(specsPath)) {
        reasons.push('Missing model_run_specs.json (planned baseline and intervention topology).');
      } else {
        try {
          const specsPayload = JSON.parse(fs.readFileSync(specsPath, 'utf8'));
          reasons.push(...collectStaleModelRunSpecReasons(specsPayload));
        } catch (error) {
          reasons.push(`Failed to validate model_run_specs.json: ${(error as Error).message}`);
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
