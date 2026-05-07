import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import * as modelSchema from './modelSchema';
import { assertValidAgainstSharedSchema } from '../sharedSchemaAjv';
import {
  extractConfiguredObservableSignals,
  getExperimentConfigPath,
  readExperimentConfigFile,
} from '../distribution/experimentConfigFile';
import {
  configuredModelArtifactPath,
  listRunDataIdsForModel,
} from '../runDataResolver';
import {
  computeChildParametersHash,
  computeParametersHash,
  computeTime0BaselineHash,
} from '../modelRunSpecHashes';

type AnyRecord = Record<string, any>;
type RunStatus = 'not_run' | 'success' | 'failed';
type RuntimeHashField = modelSchema.RuntimeHashField;
type RuntimeRecordEntry = {
  parameters_hash?: string;
  run_type: 'baseline' | 'intervention' | 'time0_baseline';
  class_internal?: string;
  class_agent_facing_name?: string;
  status: RunStatus;
  timestamp?: string;
  end_time_simulation?: number;
  error?: string;
  [key: string]: any;
};
type NormalizedSpecChild = {
  interventionUuid: string;
  parameter: string;
  setValue: unknown;
  interventionTime: number;
  parameterHash: string;
  time0BaselineHash: string;
  time0BaselineUuid: string;
};
type NormalizedSpecParent = {
  baselineUuid: string;
  parameters: AnyRecord;
  rawParameters: AnyRecord;
  parametersHash: string;
  interventionTime: number | null;
  endTimeInputS: number;
  samplingRateHz: number;
  children: NormalizedSpecChild[];
};
type ValidatedRuntimeStatus = {
  status: RunStatus;
  stale_reason?: 'hash_mismatch' | 'missing_data';
};
type DetectabilityStatus = 'yes' | 'no' | 'error';
type ChildDetectabilitySummary = {
  vsBaseline: DetectabilityStatus;
  vsTime0Baseline: DetectabilityStatus;
  failed: boolean;
};
type MetricsByBaseline = Record<string, {
  eligible?: boolean;
  children: Record<string, ChildDetectabilitySummary>;
}>;

const isRecord = (value: unknown): value is AnyRecord => (
  Boolean(value) && typeof value === 'object' && !Array.isArray(value)
);

const asString = (value: unknown) => String(value ?? '').trim();
const isFiniteNumber = (value: unknown) => Number.isFinite(Number(value));
const isHex32 = (value: unknown) => /^[a-f0-9]{32}$/.test(asString(value));
const isNullValue = (value: unknown) => value === null;
const finiteNumberOrNull = (value: unknown): number | null => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};
const positiveFiniteNumberOrNull = (value: unknown): number | null => {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
};

const PERSISTED_PARENT_PARAMETERS_FIELD = 'baseline_parameters';
const PERSISTED_PARENT_HASH_FIELD = 'baseline_parameters_hash';
const readJsonFile = (filePath: string): unknown => JSON.parse(fs.readFileSync(filePath, 'utf8'));

const readMetricsByBaseline = (repoRoot: string, model: string): MetricsByBaseline => {
  const metricsPath = configuredModelArtifactPath(repoRoot, model, 'eligibility_metrics.json');
  if (!fs.existsSync(metricsPath)) return {};
  const payload = readJsonFile(metricsPath);
  assertValidAgainstSharedSchema('similarity_metrics.schema.json', payload);
  const baselines = isRecord((payload as any)?.baselines) ? (payload as any).baselines : {};
  const out: MetricsByBaseline = {};
  for (const [baselineUuid, summary] of Object.entries(baselines)) {
    if (!isRecord(summary)) continue;
    const baselineMetrics: MetricsByBaseline[string] = { children: {} };
    if (isRecord(summary) && typeof (summary as any).eligible === 'boolean') {
      baselineMetrics.eligible = Boolean((summary as any).eligible);
    }
    const children = isRecord((summary as any).children) ? (summary as any).children : {};
    for (const [childUuid, childSummary] of Object.entries(children)) {
      if (!isRecord(childSummary)) continue;
      const detectability = isRecord((childSummary as any).detectability)
        ? (childSummary as any).detectability
        : null;
      if (!detectability) continue;
      const vsBaselineEntry = (detectability as any).vs_baseline;
      const vsTime0BaselineEntry = (detectability as any).vs_time0_baseline;
      const vsBaseline = asString(vsBaselineEntry?.detectability ?? vsBaselineEntry?.detectable).toLowerCase();
      const vsTime0Baseline = asString(vsTime0BaselineEntry?.detectability ?? vsTime0BaselineEntry?.detectable).toLowerCase();
      if (
        (vsBaseline !== 'yes' && vsBaseline !== 'no' && vsBaseline !== 'error')
        || (vsTime0Baseline !== 'yes' && vsTime0Baseline !== 'no' && vsTime0Baseline !== 'error')
      ) {
        continue;
      }
      baselineMetrics.children[asString(childUuid)] = {
        vsBaseline: vsBaseline as DetectabilityStatus,
        vsTime0Baseline: vsTime0Baseline as DetectabilityStatus,
        failed: vsBaseline !== 'yes' || vsTime0Baseline !== 'yes',
      };
    }
    out[asString(baselineUuid)] = baselineMetrics;
  }
  return out;
};

const readAvailableNoiseProfiles = (modelDir: string): string[] => {
  const noiseAdderPath = path.join(modelDir, 'noise_adder.py');
  if (!fs.existsSync(noiseAdderPath)) return ['none'];
  return ['none', 'low', 'high'];
};

const readSignalDisplayMapping = (modelDir: string): Record<string, string> => {
  const levelsPath = path.join(modelDir, 'description_levels.json');
  if (!fs.existsSync(levelsPath)) return {};
  try {
    const payload = JSON.parse(fs.readFileSync(levelsPath, 'utf8'));
    let rawMapping: Record<string, unknown> = {};
    if (isRecord(payload?.internal_naming_to_agent_facing_signal)) {
      rawMapping = payload.internal_naming_to_agent_facing_signal;
    }
    const out: Record<string, string> = {};
    for (const [rawKey, rawValue] of Object.entries(rawMapping)) {
      const key = asString(rawKey);
      const value = asString(rawValue);
      if (!key || !value) continue;
      out[key] = value;
    }
    return out;
  } catch {
    return {};
  }
};

const readConfiguredSignals = (modelDir: string): { configuredSignals: string[]; metadata: any } => {
  const experimentConfig = readExperimentConfigFile(modelDir);
  if (experimentConfig) {
    assertValidAgainstSharedSchema('experiment_config.schema.json', experimentConfig);
  }
  const metadataPath = path.join(modelDir, 'generated', 'metadata.json');
  if (!fs.existsSync(metadataPath)) {
    throw new Error(`metadata.json not found for model '${path.basename(modelDir)}'. Regenerate model metadata.`);
  }

  let metadata: any = null;
  try {
    metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
    assertValidAgainstSharedSchema('simulink_generated_metadata.schema.json', metadata);
  } catch {
    metadata = null;
  }

  const configuredSignalsFromExperiment = extractConfiguredObservableSignals(experimentConfig);
  const orderedSignals = [
    ...(Array.isArray(metadata?.simulink_signals_available)
      ? metadata.simulink_signals_available.map((s: any) => asString(s)).filter(Boolean)
      : []),
    ...(Array.isArray(metadata?.simscape_signals_available)
      ? metadata.simscape_signals_available.map((s: any) => asString(s)).filter(Boolean)
      : []),
  ];
  const configuredSignals = configuredSignalsFromExperiment.length > 0
    ? Array.from(new Set(configuredSignalsFromExperiment))
    : Array.from(new Set(orderedSignals));
  if (configuredSignals.length === 0) {
    throw new Error(
      `No configured observable signals found for model '${path.basename(modelDir)}'. Add observable_signals to experiment_config.json or regenerate metadata.`
    );
  }

  return { configuredSignals, metadata };
};

const splitParentParameters = (raw: unknown) => {
  const rawParameters = isRecord(raw) ? ({ ...raw } as AnyRecord) : {};
  const parameters = { ...rawParameters };
  return {
    rawParameters,
    parameters,
  };
};

const commonChildInterventionTime = (children: NormalizedSpecChild[]): number | null => {
  if (children.length === 0) return null;
  const first = children[0].interventionTime;
  return children.every((child) => child.interventionTime === first) ? first : null;
};

const normalizeSpecsPayload = (opts: {
  specsPayload: unknown;
  defaultSamplingRateHz: number | null;
  defaultEndTimeInputS: number | null;
}) => {
  if (!isRecord(opts.specsPayload)) {
    throw new Error('model_run_specs.json must be a JSON object');
  }
  return Object.entries(opts.specsPayload).map(([baselineUuid, rawParent]) => {
    if (!isRecord(rawParent)) {
      throw new Error(`Baseline '${baselineUuid}' must be an object.`);
    }
    const parent = rawParent;
    const persistedParentParameters = (parent as any)[PERSISTED_PARENT_PARAMETERS_FIELD];
    if (!isRecord(persistedParentParameters)) {
      throw new Error(`Baseline '${baselineUuid}' is missing ${PERSISTED_PARENT_PARAMETERS_FIELD}.`);
    }
    const { rawParameters, parameters } = splitParentParameters(persistedParentParameters);
    if (
      Object.prototype.hasOwnProperty.call(parameters, 'intervention_time')
      || Object.prototype.hasOwnProperty.call(parameters, 'end_time_input_s')
      || Object.prototype.hasOwnProperty.call(parameters, 'sampling_rate_hz')
    ) {
      throw new Error(`Baseline '${baselineUuid}' must not store intervention_time, end_time_input_s, or sampling_rate_hz in baseline_parameters.`);
    }
    if (opts.defaultEndTimeInputS === null) {
      throw new Error(`Missing end_time_input_s in experiment_config.json for baseline '${baselineUuid}'.`);
    }
    if (opts.defaultSamplingRateHz === null) {
      throw new Error(`Missing sampling_rate_hz in experiment_config.json for baseline '${baselineUuid}'.`);
    }
    const defaultEndTimeInputS = opts.defaultEndTimeInputS;
    const defaultSamplingRateHz = opts.defaultSamplingRateHz;
    const parametersHash = computeParametersHash(rawParameters);
    const storedParentHash = asString((parent as any)[PERSISTED_PARENT_HASH_FIELD]);
    if (!storedParentHash) {
      throw new Error(`Missing ${PERSISTED_PARENT_HASH_FIELD} for baseline '${baselineUuid}'.`);
    }
    if (storedParentHash !== parametersHash) {
      throw new Error(`${PERSISTED_PARENT_HASH_FIELD} mismatch for baseline '${baselineUuid}'.`);
    }
    const childrenRaw = (parent as any).children;
    if (!isRecord(childrenRaw)) {
      throw new Error(`Baseline '${baselineUuid}' is missing children object.`);
    }
    const children = Object.entries(childrenRaw).flatMap(([interventionUuid, rawChild]) => {
      if (!isRecord(rawChild)) {
        throw new Error(`Child '${interventionUuid}' under baseline '${baselineUuid}' must be an object.`);
      }
      const child = rawChild;
      if (!isRecord(child.parameters)) {
        throw new Error(`Child '${interventionUuid}' under baseline '${baselineUuid}' is missing parameters object.`);
      }
      const childParameters = { ...child.parameters } as AnyRecord;
      const childEntries = Object.entries(childParameters);
      if (childEntries.length !== 1) {
        throw new Error(`Child '${interventionUuid}' under baseline '${baselineUuid}' must contain exactly one intervened parameter.`);
      }
      const [parameter, setValue] = childEntries[0];
      const interventionTime = Number((child as any).intervention_time);
      if (!Number.isFinite(interventionTime)) {
        throw new Error(`Child '${interventionUuid}' under baseline '${baselineUuid}' must define finite intervention_time.`);
      }
      if (!(interventionTime > 0 && interventionTime < defaultEndTimeInputS)) {
        throw new Error(`Child '${interventionUuid}' under baseline '${baselineUuid}' intervention_time must satisfy 0 < intervention_time < end_time_input_s.`);
      }
      if (setValue === null) {
        if (
          !isNullValue((child as any).parameter_hash)
          || !isNullValue((child as any).time0_baseline_hash)
          || !isNullValue((child as any).time0_baseline_uuid)
        ) {
          throw new Error(
            `Skipped child '${interventionUuid}' under baseline '${baselineUuid}' must set parameter_hash, time0_baseline_hash, and time0_baseline_uuid to null.`
          );
        }
        return [];
      }
      const parameterHash = computeChildParametersHash(parametersHash, childParameters, interventionTime);
      const storedChildHash = asString((child as any).parameter_hash);
      if (!storedChildHash) {
        throw new Error(`Missing parameter_hash for child '${interventionUuid}' under baseline '${baselineUuid}'.`);
      }
      if (storedChildHash !== parameterHash) {
        throw new Error(`parameter_hash mismatch for child '${interventionUuid}' under baseline '${baselineUuid}'.`);
      }
      const time0BaselineHash = computeTime0BaselineHash(parameterHash);
      const storedTime0Hash = asString((child as any).time0_baseline_hash);
      if (!storedTime0Hash) {
        throw new Error(`Missing time0_baseline_hash for child '${interventionUuid}' under baseline '${baselineUuid}'.`);
      }
      if (storedTime0Hash !== time0BaselineHash) {
        throw new Error(`time0_baseline_hash mismatch for child '${interventionUuid}' under baseline '${baselineUuid}'.`);
      }
      const time0BaselineUuid = asString((child as any).time0_baseline_uuid);
      if (!isHex32(time0BaselineUuid)) {
        throw new Error(`Invalid time0_baseline_uuid for child '${interventionUuid}' under baseline '${baselineUuid}'.`);
      }
      return {
        interventionUuid,
        parameter,
        setValue,
        interventionTime,
        parameterHash,
        time0BaselineHash,
        time0BaselineUuid,
      } as NormalizedSpecChild;
    });
    return {
      baselineUuid,
      parameters,
      rawParameters,
      parametersHash,
      interventionTime: commonChildInterventionTime(children),
      endTimeInputS: defaultEndTimeInputS,
      samplingRateHz: defaultSamplingRateHz,
      children,
    } as NormalizedSpecParent;
  });
};

const validateRuntimeStatus = (opts: {
  runId: string;
  expectedParametersHash: string;
  runtimeRun: RuntimeRecordEntry | null;
  availableRunIds: Set<string>;
}): ValidatedRuntimeStatus => {
  const runtimeStatus = opts.runtimeRun?.status ?? 'not_run';
  if (!opts.runtimeRun) return { status: 'not_run' };

  const actualHash = asString(opts.runtimeRun.parameters_hash).toLowerCase();
  const expectedHash = asString(opts.expectedParametersHash).toLowerCase();
  if (!actualHash || actualHash !== expectedHash) {
    return { status: 'not_run', stale_reason: 'hash_mismatch' };
  }

  if (runtimeStatus === 'failed') return { status: 'failed' };
  if (runtimeStatus === 'not_run') return { status: 'not_run' };
  if (runtimeStatus === 'success' && !opts.availableRunIds.has(opts.runId)) {
    return { status: 'not_run', stale_reason: 'missing_data' };
  }
  return { status: runtimeStatus };
};

const buildMergedModelRecord = (opts: {
  modelId: string;
  specsPayload: unknown;
  runtimeModelRecord: Record<string, RuntimeRecordEntry>;
  availableRunIds: Set<string>;
  metricsByBaseline: MetricsByBaseline;
  defaultSamplingRateHz: number | null;
  defaultEndTimeInputS: number | null;
}) => {
  const specs = normalizeSpecsPayload({
    specsPayload: opts.specsPayload,
    defaultSamplingRateHz: opts.defaultSamplingRateHz,
    defaultEndTimeInputS: opts.defaultEndTimeInputS,
  });

  const mergedBaselines = specs.map((spec) => {
    const runtimeRun = opts.runtimeModelRecord[spec.baselineUuid] ?? null;
    const baselineMetrics = opts.metricsByBaseline[spec.baselineUuid];
    const interventions = spec.children.map((child) => {
      const runtimeIv = opts.runtimeModelRecord[child.interventionUuid] ?? null;
      const runtimeTime0 = opts.runtimeModelRecord[child.time0BaselineUuid] ?? null;
      const validatedIv = validateRuntimeStatus({
        runId: child.interventionUuid,
        expectedParametersHash: child.parameterHash,
        runtimeRun: runtimeIv,
        availableRunIds: opts.availableRunIds,
      });
      const validatedTime0 = validateRuntimeStatus({
        runId: child.time0BaselineUuid,
        expectedParametersHash: child.time0BaselineHash,
        runtimeRun: runtimeTime0,
        availableRunIds: opts.availableRunIds,
      });
      return {
        name: child.interventionUuid,
        parent_id: spec.baselineUuid,
        depth: 1,
        intervention_time: child.interventionTime,
        intervention_uuid: child.interventionUuid,
        parameter: child.parameter,
        set_value: child.setValue,
        variable: child.parameter,
        value: child.setValue,
        time0_baseline_uuid: child.time0BaselineUuid,
        time0_status: validatedTime0.status,
        time0_timestamp: asString(runtimeTime0?.timestamp),
        time0_stale_reason: validatedTime0.stale_reason ?? null,
        end_time_input_s: spec.endTimeInputS,
        end_time_simulation: finiteNumberOrNull(runtimeIv?.end_time_simulation),
        time0_end_time_simulation: finiteNumberOrNull(runtimeTime0?.end_time_simulation),
        display_name: runtimeIv?.display_name ?? null,
        is_classification_few_shot: runtimeIv?.is_classification_few_shot ?? null,
        detectability_passed: runtimeIv?.detectability_passed ?? null,
        distance_baseline: runtimeIv?.distance_baseline ?? null,
        distance_parent: runtimeIv?.distance_parent ?? null,
        threshold: runtimeIv?.threshold ?? null,
        threshold_parent: runtimeIv?.threshold_parent ?? null,
        failure_reason: runtimeIv?.failure_reason ?? null,
        time0_detectability_passed: runtimeIv?.time0_detectability_passed ?? null,
        distance_time0_baseline: runtimeIv?.distance_time0_baseline ?? null,
        threshold_time0_baseline: runtimeIv?.threshold_time0_baseline ?? null,
        time0_failure_reason: runtimeIv?.time0_failure_reason ?? null,
        parent_similarity_corr: runtimeIv?.parent_similarity_corr ?? null,
        parent_similarity_euclid: runtimeIv?.parent_similarity_euclid ?? null,
        parent_similarity_rmse: runtimeIv?.parent_similarity_rmse ?? null,
        parent_similarity_overlays: runtimeIv?.parent_similarity_overlays ?? null,
        question: runtimeIv?.question ?? null,
        detection_name: runtimeIv?.detection_name ?? null,
        detection_feedback: runtimeIv?.detection_feedback ?? null,
        detection_few_shot_ids: runtimeIv?.detection_few_shot_ids ?? null,
        detection_dataset: runtimeIv?.detection_dataset ?? null,
        detection_signals: runtimeIv?.detection_signals ?? null,
        detectability_failed: baselineMetrics?.children?.[child.interventionUuid]?.failed ?? false,
        status: validatedIv.status,
        stale_reason: validatedIv.stale_reason ?? null,
        timestamp: asString(runtimeIv?.timestamp),
      };
    });

    const validatedBaseline = validateRuntimeStatus({
      runId: spec.baselineUuid,
      expectedParametersHash: spec.parametersHash,
      runtimeRun,
      availableRunIds: opts.availableRunIds,
    });
    const baseline: AnyRecord = {
      run_id: spec.baselineUuid,
      baseline_uuid: spec.baselineUuid,
      parent_id: null,
      parameters: spec.parameters,
      intervention_time: spec.interventionTime,
      sampling_rate_hz: spec.samplingRateHz,
      end_time_input_s: spec.endTimeInputS,
      end_time_simulation: finiteNumberOrNull(runtimeRun?.end_time_simulation),
      error: runtimeRun?.error ?? null,
      noise: runtimeRun?.noise ?? null,
      classification: runtimeRun?.classification ?? null,
      interventions,
      status: validatedBaseline.status,
      stale_reason: validatedBaseline.stale_reason ?? null,
      timestamp: asString(runtimeRun?.timestamp),
    };
    if (baselineMetrics && Object.prototype.hasOwnProperty.call(baselineMetrics, 'eligible')) {
      baseline.eligible = baselineMetrics.eligible;
    }
    if (baseline.status === 'not_run') {
      baseline.interventions = baseline.interventions.map((iv: AnyRecord) => ({
        ...iv,
        status: 'not_run',
        timestamp: '',
        time0_status: 'not_run',
        time0_timestamp: '',
      }));
    }
    if (baseline.status === 'failed') {
      baseline.interventions = baseline.interventions.map((iv: AnyRecord) => ({
        ...iv,
        status: 'failed',
        time0_status: 'failed',
      }));
    }
    return baseline;
  });

  return {
    version: 1,
    model_id: opts.modelId,
    metadata: {},
    baselines: mergedBaselines,
  };
};

const buildRuntimeModelRecordFromAvailableRuns = (opts: {
  specsPayload: unknown;
  defaultSamplingRateHz: number | null;
  defaultEndTimeInputS: number | null;
  availableRunIds: Set<string>;
}): Record<string, RuntimeRecordEntry> => {
  const specs = normalizeSpecsPayload({
    specsPayload: opts.specsPayload,
    defaultSamplingRateHz: opts.defaultSamplingRateHz,
    defaultEndTimeInputS: opts.defaultEndTimeInputS,
  });
  const runtimeModelRecord: Record<string, RuntimeRecordEntry> = {};
  for (const spec of specs) {
    runtimeModelRecord[spec.baselineUuid] = {
      parameters_hash: spec.parametersHash,
      run_type: 'baseline',
      class_internal: 'baseline',
      class_agent_facing_name: 'baseline',
      status: opts.availableRunIds.has(spec.baselineUuid) ? 'success' : 'not_run',
    };
    for (const child of spec.children) {
      runtimeModelRecord[child.interventionUuid] = {
        parameters_hash: child.parameterHash,
        run_type: 'intervention',
        class_internal: child.parameter,
        class_agent_facing_name: child.parameter,
        status: opts.availableRunIds.has(child.interventionUuid) ? 'success' : 'not_run',
      };
      if (isHex32(child.time0BaselineUuid)) {
        runtimeModelRecord[child.time0BaselineUuid] = {
          parameters_hash: child.time0BaselineHash,
          run_type: 'time0_baseline',
          class_internal: 'nothing_happened',
          class_agent_facing_name: 'Nothing happened',
          status: opts.availableRunIds.has(child.time0BaselineUuid) ? 'success' : 'not_run',
        };
      }
    }
  }
  return runtimeModelRecord;
};

const buildExpectedRuntimeHashFields = (opts: {
  specsPayload: unknown;
  defaultSamplingRateHz: number | null;
  defaultEndTimeInputS: number | null;
}): Record<string, RuntimeHashField> => {
  const specs = normalizeSpecsPayload(opts);
  const out: Record<string, RuntimeHashField> = {};
  for (const spec of specs) {
    out[spec.baselineUuid] = 'parameters_hash';
    for (const child of spec.children) {
      out[child.interventionUuid] = 'parameters_hash';
      out[child.time0BaselineUuid] = 'parameters_hash';
    }
  }
  return out;
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const model = asString(searchParams.get('model'));
  if (!model) return NextResponse.json({ error: 'Model required' }, { status: 400 });

  const repoRoot = path.join(process.cwd(), '..');
  const modelDir = path.join(repoRoot, 'models', 'simulink', model);
  const specsPath = path.join(modelDir, 'model_run_specs.json');
  const registryPath = configuredModelArtifactPath(repoRoot, model, 'model_record.json');
  const experimentConfigPath = getExperimentConfigPath(modelDir);

  try {
    if (!fs.existsSync(specsPath)) {
      return NextResponse.json({ error: 'model_run_specs.json not found for model' }, { status: 404 });
    }

    const specsPayload = readJsonFile(specsPath);
    const experimentConfig = fs.existsSync(experimentConfigPath)
      ? readJsonFile(experimentConfigPath)
      : null;
    if (experimentConfig) {
      assertValidAgainstSharedSchema('experiment_config.schema.json', experimentConfig);
    }
    const defaultSamplingRateHz: number | null = positiveFiniteNumberOrNull((experimentConfig as any)?.sampling_rate_hz);
    const defaultEndTimeInputS: number | null = positiveFiniteNumberOrNull((experimentConfig as any)?.end_time_input_s);
    const expectedHashFields = buildExpectedRuntimeHashFields({
      specsPayload,
      defaultSamplingRateHz,
      defaultEndTimeInputS,
    });
    const diskRuns: string[] = listRunDataIdsForModel({
      repoRoot,
      model,
    });
    const runtimeModelRecord = fs.existsSync(registryPath)
      ? (() => {
          const runtimeRaw = readJsonFile(registryPath);
          if (!isRecord(runtimeRaw) || Array.isArray(runtimeRaw)) {
            throw new Error('model_record.json must be a flat JSON object runtime map.');
          }
          if (Array.isArray((runtimeRaw as any).baselines)) {
            throw new Error('model_record.json must be a flat runtime map, not a baselines list.');
          }
          return modelSchema.normalizeRuntimeModelRecord(runtimeRaw, { expectedHashFields });
        })()
      : buildRuntimeModelRecordFromAvailableRuns({
          specsPayload,
          defaultSamplingRateHz,
          defaultEndTimeInputS,
          availableRunIds: new Set(diskRuns),
        });
    const metricsByBaseline = readMetricsByBaseline(repoRoot, model);

    const merged = buildMergedModelRecord({
      modelId: model,
      specsPayload,
      runtimeModelRecord,
      availableRunIds: new Set(diskRuns),
      metricsByBaseline,
      defaultSamplingRateHz,
      defaultEndTimeInputS,
    });
    let modelRecord: any;
    try {
      modelRecord = modelSchema.normalizeModelRecord(merged, { modelId: model, defaultSamplingRateHz });
    } catch {
      modelRecord = merged;
    }

    const { configuredSignals, metadata } = readConfiguredSignals(modelDir);
    const configuredSet = new Set(configuredSignals);
    const signalDisplayNames = readSignalDisplayMapping(modelDir);
    const runtimeMetadata = isRecord((runtimeModelRecord as any)?.metadata)
      ? ((runtimeModelRecord as any).metadata as AnyRecord)
      : {};
    const signalSpecs = Array.isArray((runtimeMetadata as any).signals)
      ? (runtimeMetadata as any).signals
      : [];
    for (const spec of signalSpecs) {
      if (!isRecord(spec)) continue;
      const key = asString((spec as any).key);
      if (!key || !configuredSet.has(key)) continue;
      const displayName = asString((spec as any).display_name);
      if (displayName) signalDisplayNames[key] = displayName;
    }
    const availableNoiseProfiles = readAvailableNoiseProfiles(modelDir);
    const availableSignals = configuredSignals.slice();
    return NextResponse.json({
      modelRecord,
      diskRuns,
      signalDisplayNames,
      availableNoiseProfiles,
      availableSignals,
      metadata,
    });
  } catch (error) {
    return NextResponse.json({ error: (error as Error).message }, { status: 500 });
  }
}
