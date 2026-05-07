/* This file is generated. Do not edit by hand. */
/* Source: shared/schemas/similarity_metrics.schema.json */

export interface DetectabilityOutput {
  mean_euclidean_distance_clean_dirty: (number)[];
  mean_euclidean_distance_clean_baseline: (number)[];
  mean_SNR: (any)[];
  first_diff: (any)[];
}

export interface BaselineDetectabilityStatus {
  environment_specific_detectability: "yes" | "no" | "error";
  detectable: "yes" | "no" | "error";
  detectability_output: DetectabilityOutput;
}

export interface Time0DetectabilityStatus {
  detectable: "yes" | "no" | "error";
  detectability_output: DetectabilityOutput;
}

export interface DetectabilitySummary {
  vs_baseline: BaselineDetectabilityStatus;
  vs_time0_baseline: Time0DetectabilityStatus;
}

export interface ChildSimilarity {
  url: string;
  detectability: DetectabilitySummary;
  eligible: boolean;
}

export interface BaselineSimilarity {
  url: string;
  family_eligible: boolean;
  eligible: boolean;
  children: Record<string, ChildSimilarity>;
}

export interface SchemaRoot {
  timestamp: string;
  noise_adder_md5: any;
  eligible_baselines: number;
  total_baselines: number;
  baselines: Record<string, BaselineSimilarity>;
}
