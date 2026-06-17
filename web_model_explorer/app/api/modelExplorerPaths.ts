import path from 'path';

const cleanSegment = (value: unknown) => String(value ?? '').trim();

export const repoRoot = () => path.join(process.cwd(), '..');

export const runsDirName = () => cleanSegment(process.env.WEB_MODEL_EXPLORER_RUNS_DIR_NAME) || 'runs';

export const modelArtifactDir = () => cleanSegment(process.env.WEB_MODEL_EXPLORER_MODEL_ARTIFACT_DIR);

export const tsenvQuestionsRoot = (root = repoRoot()) => path.join(root, 'tsENV_questions');

export const modelsRoot = (root = repoRoot()) => path.join(root, 'models', 'simulink');

export const repoModelDir = (model: string, root = repoRoot()) => path.join(modelsRoot(root), model);

export const modelDir = (model: string, root = repoRoot()) => modelArtifactDir() || repoModelDir(model, root);

export const modelRunsDir = (model: string, root = repoRoot()) => {
  return path.join(modelDir(model, root), runsDirName());
};

export const modelArtifactPath = (model: string, filename: string, root = repoRoot()) =>
  path.join(modelRunsDir(model, root), filename);

export const modelPlanDir = (model: string, policy: string, root = repoRoot()) =>
  path.join(modelDir(model, root), 'plans', policy);

export const tsenvModelDir = (model: string, root = repoRoot()) =>
  path.join(tsenvQuestionsRoot(root), model);
