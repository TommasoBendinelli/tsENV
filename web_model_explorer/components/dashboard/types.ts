import type * as ModelRecordSchema from '../../generated/schemas/model_record.schema';

type WithRequired<T, K extends keyof T> = T & {
  [P in K]-?: Exclude<T[P], undefined>;
};

export type SignalSpec = ModelRecordSchema.SignalSpec;
export type ModelMetadata = ModelRecordSchema.ModelMetadata;

// Raw contents of `experiment_config.json` (experiment-level sampling and observable-signal specs).
export type DistributionsFile = Record<string, any>;

export type InterventionRun = WithRequired<
  ModelRecordSchema.SimulatedInterventionRecord,
  | 'depth'
  | 'intervention_time'
  | 'variable'
  | 'value'
  | 'status'
  | 'timestamp'
>;

type _BaselineRunBase = WithRequired<
  ModelRecordSchema.SimulatedRunRecord,
  | 'parent_id'
  | 'parameters'
  | 'intervention_time'
  | 'interventions'
  | 'status'
  | 'timestamp'
>;
export type BaselineRun = Omit<_BaselineRunBase, 'interventions'> & {
  interventions: InterventionRun[];
};

export type ModelRecord = Omit<ModelRecordSchema.ModelRecord, 'baselines'> & {
  baselines: BaselineRun[];
};

export type RunRecord = BaselineRun;
export type Intervention = InterventionRun;

export interface SimulationStatus {
  status: 'idle' | 'running' | 'success' | 'failed';
  started_at?: string;
  finished_at?: string;
  code?: number | null;
  stdout?: string;
  stderr?: string;
  error?: string;
}

export interface SamplingSpec {
  min: number;
  max: number;
  type: 'uniform' | 'loguniform';
}

export interface RegistryPageInfo {
  mode: 'page' | 'family' | 'full';
  page: number;
  page_size: number;
  total_families: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
  family_id?: string;
  run?: string;
}
