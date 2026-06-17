import { assertValidAgainstSharedSchema } from '../sharedSchemaAjv';

type AnyRecord = Record<string, any>;

const isRecord = (value: unknown): value is AnyRecord => (
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const asString = (value: unknown) => String(value ?? '').trim();

export type RuntimeHashField =
  | 'parameters_hash'
  | 'recipe_hash';

const RUNTIME_HASH_FIELDS: RuntimeHashField[] = [
  'recipe_hash',
  'parameters_hash',
];

export type NormalizedModelRecord = AnyRecord;
export type NormalizedRuntimeModelRecord = Record<string, {
  parameters_hash?: string;
  recipe_hash?: string;
  run_type: 'baseline' | 'intervention' | 'time0_baseline';
  class_internal?: string;
  class_agent_facing_name?: string;
  status: 'not_run' | 'success' | 'failed';
  timestamp?: string;
  end_time_simulation?: number;
  error?: string;
  [key: string]: any;
}>;

export const normalizeModelRecord = (
  payload: unknown,
  opts: { modelId: string; defaultSamplingRateHz?: number | null },
): NormalizedModelRecord => {
  if (!isRecord(payload)) {
    throw new Error('model_record.json must be a JSON object (ModelRecord)');
  }

  const record = assertValidAgainstSharedSchema('model_record.schema.json', payload);
  const modelId = asString((record as any).model_id);
  if (modelId !== asString(opts.modelId)) {
    throw new Error(`model_id mismatch: expected '${opts.modelId}' got '${modelId}'`);
  }
  return record;
};

export const normalizeRuntimeModelRecord = (
  payload: unknown,
  opts?: { expectedHashFields?: Record<string, RuntimeHashField> },
): NormalizedRuntimeModelRecord => {
  if (!isRecord(payload)) {
    throw new Error('model_record.json must be a JSON object');
  }
  const out: NormalizedRuntimeModelRecord = {};
  const expectedHashFields = isRecord(opts?.expectedHashFields)
    ? Object.fromEntries(
        Object.entries(opts?.expectedHashFields ?? {}).map(([runId, field]) => [
          asString(runId).toLowerCase(),
          field,
        ])
      )
    : {};
  for (const [runId, raw] of Object.entries(payload)) {
    const normalizedRunId = asString(runId).toLowerCase();
    if (!/^[a-f0-9]{32}$/i.test(normalizedRunId)) {
      throw new Error(`Invalid run id '${runId}' in model_record.json`);
    }
    if (!isRecord(raw)) {
      throw new Error(`model_record.json entry '${runId}' must be an object`);
    }
    const status = asString((raw as any).status);
    if (Object.prototype.hasOwnProperty.call(raw, 'runtime_values')) {
      throw new Error(`model_record.json entry '${runId}' must not contain runtime_values`);
    }
    const presentHashFields = RUNTIME_HASH_FIELDS.filter((field) => asString((raw as any)[field]));
    const expectedHashField = expectedHashFields[normalizedRunId];
    if (presentHashFields.length === 0) {
      const missingField = expectedHashField ?? 'hash field';
      throw new Error(`model_record.json entry '${runId}' is missing ${missingField}`);
    }
    const hashField = presentHashFields[0];
    if (expectedHashField && hashField !== expectedHashField) {
      throw new Error(`model_record.json entry '${runId}' must use ${expectedHashField}`);
    }
    if (status !== 'not_run' && status !== 'success' && status !== 'failed') {
      throw new Error(`model_record.json entry '${runId}' has invalid status`);
    }
    const runType = asString((raw as any).run_type);
    if (runType !== 'baseline' && runType !== 'intervention' && runType !== 'time0_baseline') {
      throw new Error(`model_record.json entry '${runId}' has invalid run_type`);
    }
    out[normalizedRunId] = {
      [hashField]: asString((raw as any)[hashField]).toLowerCase(),
      run_type: runType as 'baseline' | 'intervention' | 'time0_baseline',
      class_internal: asString((raw as any).class_internal),
      class_agent_facing_name: asString((raw as any).class_agent_facing_name),
      status: status as 'not_run' | 'success' | 'failed',
      timestamp: asString((raw as any).timestamp),
      end_time_simulation: Number.isFinite(Number((raw as any).end_time_simulation))
        ? Number((raw as any).end_time_simulation)
        : undefined,
      error: asString((raw as any).error) || undefined,
    };
  }
  return out;
};
