/* This file is generated. Do not edit by hand. */
/* Source: shared/schemas/experiment_config.schema.json */

export type SamplingType = "uniform" | "loguniform";

export type ObservableSignalType = "continuous" | "impulse_like";

export interface ObservableSignalSpecObject {
  type: ObservableSignalType;
}

export type ObservableSignalsList = (string)[];

export interface ObservableSignalsConfig {
  observable_signals: ObservableSignalsList;
  signal_type: Record<string, ObservableSignalSpecObject>;
}

export interface ContinuousDetectabilityConfig {
  min_srd_distance: number;
  epsilon_SRD: number;
  minimum_consecurive_below_SRD: number;
}

export interface ImpulseLikeDetectabilityConfig {
  min_srd_distance: number;
  epsilon_SRD: number;
}

export interface DetectabilityConfig {
  continuous: ContinuousDetectabilityConfig;
  impulse_like: ImpulseLikeDetectabilityConfig;
  signal_to_noise_ratio_db_thresholds?: Record<string, SignalToNoiseRatioThresholdPair>;
}

export type SignalToNoiseRatioThresholdPair = (number)[];

export type IntervalRange = (number)[];

export interface VariableIntervalSpecObject {
  allowed_intervals: IntervalRange;
  min_srd_distance: number;
  min_abs_dist: number;
  sampling_strategy: SamplingType;
}

export interface ParameterIntervalSpecObject {
  allowed_intervals: IntervalRange;
  min_srd_distance: number;
  min_abs_dist: number;
  sampling_strategy: SamplingType;
}

export type NamedIntervalMap = Record<string, VariableIntervalSpecObject>;

export type NamedParameterIntervalMap = Record<string, ParameterIntervalSpecObject>;

export interface ExposedVariablesConfig {
  initial_state?: NamedIntervalMap;
  parameters?: NamedParameterIntervalMap;
}

export interface SchemaRoot {
  exposed_variables: ExposedVariablesConfig;
  sampling_rate_hz: number;
  end_time_input_s: number;
  detectability: DetectabilityConfig;
  observable_signals: ObservableSignalsConfig;
}
