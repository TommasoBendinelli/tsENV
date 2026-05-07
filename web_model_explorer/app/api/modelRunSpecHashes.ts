import crypto from 'crypto';
import { canonicalJson } from './pythonCanonicalJson';

type AnyRecord = Record<string, any>;

const PERSISTED_PARENT_PARAMETERS_FIELD = 'baseline_parameters';
const PERSISTED_PARENT_HASH_FIELD = 'baseline_parameters_hash';

const isRecord = (value: unknown): value is AnyRecord => (
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const asString = (value: unknown) => String(value ?? '').trim();

const normalizeIdentityNumbers = (value: unknown): unknown => {
  if (value === null || value === undefined) return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return value;
    if (Object.is(value, -0)) return 0;
    if (Number.isInteger(value)) return Math.trunc(value);
    return value;
  }
  if (Array.isArray(value)) return value.map((item) => normalizeIdentityNumbers(item));
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[String(k)] = normalizeIdentityNumbers(v);
    }
    return out;
  }
  return value;
};

export const sha256_32_hex = (text: string) => (
  crypto.createHash('sha256').update(text, 'utf8').digest('hex').slice(0, 32)
);

export const computeParametersHash = (parameters: Record<string, unknown>) => (
  sha256_32_hex(canonicalJson(normalizeIdentityNumbers(isRecord(parameters) ? parameters : {})))
);

export const computeChildParametersHash = (
  parentParametersHash: string,
  childParameters: Record<string, unknown>,
  interventionTime: unknown
) => (
  sha256_32_hex(canonicalJson({
    parent_parameters_hash: String(parentParametersHash || ''),
    parameters: normalizeIdentityNumbers(isRecord(childParameters) ? childParameters : {}),
    intervention_time: normalizeIdentityNumbers(interventionTime),
  }))
);

export const computeTime0BaselineHash = (childParameterHash: string) => (
  sha256_32_hex(canonicalJson({
    parameter_hash: String(childParameterHash || ''),
    kind: 'time0',
  }))
);

export function collectStaleModelRunSpecReasons(specsPayload: unknown): string[] {
  if (!isRecord(specsPayload)) return ['model_run_specs.json must be a JSON object.'];

  const reasons: string[] = [];
  for (const [baselineUuid, rawParent] of Object.entries(specsPayload)) {
    if (!isRecord(rawParent)) {
      reasons.push(`model_run_specs.json baseline '${baselineUuid}' must be an object.`);
      continue;
    }

    const baselineParameters = rawParent[PERSISTED_PARENT_PARAMETERS_FIELD];
    if (!isRecord(baselineParameters)) {
      reasons.push(`model_run_specs.json baseline '${baselineUuid}' is missing baseline_parameters.`);
      continue;
    }

    const expectedParentHash = computeParametersHash(baselineParameters);
    if (asString(rawParent[PERSISTED_PARENT_HASH_FIELD]).toLowerCase() !== expectedParentHash) {
      reasons.push(`model_run_specs.json appears stale: baseline '${baselineUuid}' baseline_parameters_hash does not match current baseline_parameters.`);
    }

    const children = rawParent.children;
    if (!isRecord(children)) continue;
    for (const [childUuid, rawChild] of Object.entries(children)) {
      if (!isRecord(rawChild)) continue;
      const childParameters = rawChild.parameters;
      if (!isRecord(childParameters)) continue;
      if (Object.values(childParameters).some((value) => value === null)) continue;

      const expectedChildHash = computeChildParametersHash(expectedParentHash, childParameters, rawChild.intervention_time);
      if (asString(rawChild.parameter_hash).toLowerCase() !== expectedChildHash) {
        reasons.push(`model_run_specs.json appears stale: child '${childUuid}' parameter_hash does not match current parameters.`);
      }

      const expectedTime0Hash = computeTime0BaselineHash(expectedChildHash);
      if (asString(rawChild.time0_baseline_hash).toLowerCase() !== expectedTime0Hash) {
        reasons.push(`model_run_specs.json appears stale: child '${childUuid}' time0_baseline_hash does not match current parameters.`);
      }
    }
  }

  return reasons;
}
