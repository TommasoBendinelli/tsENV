/* This file is generated. Do not edit by hand. */
/* Source: shared/schemas/model_record.schema.json */

export interface ClassificationConfig {
  dataset?: string | null;
  entries?: Record<string, ClassificationEntry> | null;
  feedback?: string | null;
  name?: string | null;
  one_shot_examples?: (string)[] | null;
  question?: string | null;
  signal_text?: string | null;
  signals?: (string)[] | null;
  status?: string | null;
}

export interface ClassificationEntry {
  choices?: string | null;
  few_shot_ids?: (string)[] | null;
  label?: string | null;
  one_shot_examples?: (string)[] | null;
}

export interface ModelMetadata {
  parameter_display_mapping?: Record<string, string>;
  signals?: (SignalSpec)[] | null;
}

export interface SignalSpec {
  display_name?: string | null;
  key: string;
  noise_multiplier?: number | null;
}

export interface SimulatedInterventionRecord {
  depth?: number;
  end_time_input_s?: number | null;
  end_time_simulation?: number | null;
  error?: string | null;
  intervention_time?: number | null;
  intervention_uuid?: string;
  name: string;
  parameter?: string;
  parent_id: string;
  set_value?: number | string | null;
  status?: "not_run" | "success" | "failed";
  time0_baseline_end_time_simulation?: number | null;
  time0_baseline_error?: string | null;
  time0_baseline_status?: "not_run" | "success" | "failed";
  time0_baseline_timestamp?: string;
  time0_baseline_uuid?: string | null;
  timestamp?: string;
  value?: number | string | null;
  variable?: string;
}

export interface SimulatedRunRecord {
  baseline_uuid?: string;
  classification?: ClassificationConfig | null;
  end_time_input_s?: number | null;
  end_time_simulation?: number | null;
  error?: string | null;
  intervention_time?: number;
  interventions?: (SimulatedInterventionRecord)[];
  noise?: number | null;
  parameters?: Record<string, any>;
  parent_id?: string | null;
  run_id: string;
  sampling_rate_hz: number;
  status?: "not_run" | "success" | "failed";
  timestamp?: string;
}

export interface ModelRecord {
  baselines: (SimulatedRunRecord)[];
  metadata?: ModelMetadata;
  model_id: string;
  version: 1;
}
