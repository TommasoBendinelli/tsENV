import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getExperimentConfigPath } from './experimentConfigFile';
import { assertValidAgainstSharedSchema } from '../sharedSchemaAjv';
import { modelDir as resolveModelDir } from '../modelExplorerPaths';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const model = searchParams.get('model');
  if (!model) {
    return NextResponse.json({ error: 'Model required' }, { status: 400 });
  }

  const modelDir = resolveModelDir(model);
  const experimentConfigPath = getExperimentConfigPath(modelDir);
  const metadataPath = path.join(modelDir, 'generated', 'metadata.json');
  try {
    let distribution: any = null;
    if (fs.existsSync(experimentConfigPath)) {
      distribution = JSON.parse(fs.readFileSync(experimentConfigPath, 'utf8'));
      assertValidAgainstSharedSchema('experiment_config.schema.json', distribution);
    }
    const metadata = fs.existsSync(metadataPath)
      ? JSON.parse(fs.readFileSync(metadataPath, 'utf8'))
      : null;
    if (metadata) assertValidAgainstSharedSchema('simulink_generated_metadata.schema.json', metadata);
    return NextResponse.json({ distribution, metadata });
  } catch (error) {
    return NextResponse.json({ error: (error as Error).message }, { status: 500 });
  }
}
