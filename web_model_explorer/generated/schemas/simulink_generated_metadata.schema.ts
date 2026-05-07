/* This file is generated. Do not edit by hand. */
/* Source: shared/schemas/simulink_generated_metadata.schema.json */

export interface InterventionBlockMapEntry {
  parameters: (InterventionBlockParameter)[];
}

export interface InterventionBlockParameter {
  expression: string;
  name: string;
  path: string;
  runtime_type: boolean;
}

export interface SimulinkGeneratedMetadata {
  default_values: { [k: string]: number };
  intervention_block_map: Record<string, InterventionBlockMapEntry>;
  parameter_set: (string)[];
  simscape_signals_available: (string)[];
  simulink_signals_available: (string)[];
}
