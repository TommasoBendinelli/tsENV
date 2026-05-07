import { expect, test } from 'vitest';
import {
  collectBaselineSubtreeRunIds,
  formatNumericSignificantDigits,
  formatVariableLabel,
  resolveNumericSignificantDigits,
  roundToSignificantDigits,
} from '../components/dashboard/domains/runs';

test('formatNumericSignificantDigits uses 4 significant digits by default', () => {
  expect(formatNumericSignificantDigits(1342)).toBe('1342');
  expect(formatNumericSignificantDigits(12.2222)).toBe('12.22');
  expect(formatNumericSignificantDigits(0.01434123)).toBe('0.01434');
});

test('roundToSignificantDigits rounds finite numbers and preserves zero', () => {
  expect(roundToSignificantDigits(9999.9, 4)).toBe(10000);
  expect(roundToSignificantDigits(-0.000987654, 4)).toBeCloseTo(-0.0009877, 12);
  expect(roundToSignificantDigits(0, 4)).toBe(0);
});

test('resolveNumericSignificantDigits clamps to a safe integer range', () => {
  expect(resolveNumericSignificantDigits(undefined)).toBe(4);
  expect(resolveNumericSignificantDigits(-5)).toBe(1);
  expect(resolveNumericSignificantDigits(99)).toBe(15);
  expect(resolveNumericSignificantDigits(6.9)).toBe(6);
});

test('formatVariableLabel maps display names and falls back to the raw key', () => {
  expect(formatVariableLabel('Position', { Position: 'Ball position' })).toBe('Ball position');
  expect(formatVariableLabel('Velocity', null)).toBe('Velocity');
});

test('collectBaselineSubtreeRunIds includes baseline, children, and linked time0 ids', () => {
  expect(
    collectBaselineSubtreeRunIds({
      run_id: 'baseline1',
      interventions: [
        { name: 'child1', time0_baseline_uuid: 'time0_1' },
        { name: 'child2', time0_baseline_uuid: 'time0_2' },
      ],
    })
  ).toEqual(['baseline1', 'child1', 'time0_1', 'child2', 'time0_2']);
});

test('collectBaselineSubtreeRunIds ignores empty and duplicate ids', () => {
  expect(
    collectBaselineSubtreeRunIds({
      run_id: 'baseline1',
      interventions: [
        { name: 'child1', time0_baseline_uuid: 'baseline1' },
        { name: ' ', time0_baseline_uuid: 'child1' },
      ],
    })
  ).toEqual(['baseline1', 'child1']);
});
