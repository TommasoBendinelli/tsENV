/* This file is generated. Do not edit by hand. */
/* Source: shared/schemas/sample_manifest.schema.json */

export interface Accuracy {
  correlation_nn: number;
  euclidean_centroid: number;
  euclidean_knn: number;
}

export interface ManifestShotSelection {
  is_adversarial: boolean | null;
  number_test_samples: number;
  number_train_samples_per_class: number;
}

export interface ManifestItem {
  accuracy_with_baselines: Accuracy;
  accuracy_with_baselines_all_samples: Accuracy;
  seed: number;
  test_set_slug: string;
  shot_slug_recipe: ManifestShotSelection;
  test_samples: (string)[];
  test_samples_baselines: (string)[];
  test_samples_hashes: (string)[];
  other_samples: (string)[];
  train_samples: (string)[];
  train_samples_baselines: (string)[];
  train_samples_hashes: (string)[];
  train_test_sample_hash: string;
}

export type SchemaRoot = Record<string, (ManifestItem)[]>;
