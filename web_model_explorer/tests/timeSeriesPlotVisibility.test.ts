import { expect, test } from 'vitest';
import { computeVisibleSignalsFromPlotlyData } from '../components/timeSeriesPlotVisibility';

test('returns visible signal keys in available signal order', () => {
  expect(computeVisibleSignalsFromPlotlyData([
    { visible: true, meta: { signalKey: 'sig_b', runId: 'run1' } },
    { visible: 'legendonly', meta: { signalKey: 'sig_a', runId: 'run1' } },
    { visible: true, meta: { signalKey: 'sig_a', runId: 'run2' } },
  ], ['sig_a', 'sig_b', 'sig_c'])).toEqual(['sig_a', 'sig_b']);
});

test('returns an empty list when all traces are hidden', () => {
  expect(computeVisibleSignalsFromPlotlyData([
    { visible: 'legendonly', meta: { signalKey: 'sig_a' } },
    { visible: false, meta: { signalKey: 'sig_b' } },
  ], ['sig_a', 'sig_b'])).toEqual([]);
});

test('falls back to all signals when plot data is unavailable', () => {
  expect(computeVisibleSignalsFromPlotlyData(null, ['sig_a', 'sig_b'])).toEqual(['sig_a', 'sig_b']);
});
