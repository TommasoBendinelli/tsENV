import { expect, test } from 'vitest';
import { migrateLegacyParameterSpecsIntoExperimentConfig } from '../app/api/distribution/experimentConfigFile';

test('migrates legacy metadata.parameters sampling into exposed_variables.parameters', () => {
  const { next, changed } = migrateLegacyParameterSpecsIntoExperimentConfig(
    {},
    [
      { key: 'alpha', sampling: { min: 1, max: 2, type: 'uniform' } },
      { key: 'beta', sampling: null },
    ],
  );

  expect(changed).toBe(true);
  expect((next as any).exposed_variables.parameters.alpha).toEqual({
    allowed_intervals: [1, 2],
    sampling_strategy: 'uniform',
  });
  expect((next as any).exposed_variables.parameters.beta).toBeUndefined();
});

test('does not override existing exposed_variables.parameters entries', () => {
  const { next, changed } = migrateLegacyParameterSpecsIntoExperimentConfig(
    {
      exposed_variables: {
        initial_state: {},
        parameters: {
          alpha: {
            allowed_intervals: [0, 0],
            sampling_strategy: 'uniform',
          },
        },
      },
    },
    [
      { key: 'alpha', sampling: { min: 1, max: 2, type: 'uniform' } },
      { key: 'gamma', sampling: { min: 3, max: 4, type: 'loguniform' } },
    ],
  );

  expect(changed).toBe(true);
  expect((next as any).exposed_variables.parameters.alpha).toEqual({
    allowed_intervals: [0, 0],
    sampling_strategy: 'uniform',
  });
  expect((next as any).exposed_variables.parameters.gamma).toEqual({
    allowed_intervals: [3, 4],
    sampling_strategy: 'loguniform',
  });
});
