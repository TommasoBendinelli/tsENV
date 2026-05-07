import { expect, test } from 'vitest';
import { normalizeRuntimeModelRecord } from '../app/api/registry/modelSchema';

test('strict runtime model_record payload validates against the documented parameters_hash contract', () => {
  const raw = {
    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: {
      parameters_hash: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
      run_type: 'baseline',
      class_internal: 'baseline',
      class_agent_facing_name: 'baseline',
      status: 'success',
      timestamp: '',
    },
    cccccccccccccccccccccccccccccccc: {
      parameters_hash: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
      run_type: 'intervention',
      class_internal: 'mass',
      class_agent_facing_name: 'mass',
      status: 'success',
      timestamp: '',
    },
    dddddddddddddddddddddddddddddddd: {
      parameters_hash: 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
      run_type: 'time0_baseline',
      class_internal: 'nothing_happened',
      class_agent_facing_name: 'Nothing happened',
      status: 'failed',
      error: 'boom',
    },
  };
  expect(() => normalizeRuntimeModelRecord(raw, {
    expectedHashFields: {
      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: 'parameters_hash',
      cccccccccccccccccccccccccccccccc: 'parameters_hash',
      dddddddddddddddddddddddddddddddd: 'parameters_hash',
    },
  })).not.toThrow();
});

test('runtime payloads with non-run-id top-level keys are rejected', () => {
  const raw = {
    version: 1,
  };
  expect(() => normalizeRuntimeModelRecord(raw)).toThrow(/Invalid run id 'version'/);
});

test('legacy baselines model_record payload is rejected', () => {
  expect(() => normalizeRuntimeModelRecord({
    baselines: [],
  })).toThrow(/Invalid run id|entry 'baselines' must be an object/);
});

test('runtime entries missing parameters_hash are rejected', () => {
  expect(() => normalizeRuntimeModelRecord({
    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: {
      run_type: 'baseline',
      status: 'success',
    },
  }, {
    expectedHashFields: {
      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: 'parameters_hash',
    },
  })).toThrow(/missing parameters_hash/);
});

test('runtime entries reject unexpected hash field names', () => {
  expect(() => normalizeRuntimeModelRecord({
    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: {
      baseline_parameters_hash: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',
      run_type: 'time0_baseline',
      status: 'success',
    },
  }, {
    expectedHashFields: {
      aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa: 'parameters_hash',
    },
  })).toThrow(/missing parameters_hash/);
});
