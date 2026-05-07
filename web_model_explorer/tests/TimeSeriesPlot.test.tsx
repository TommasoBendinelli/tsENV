import React from 'react';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, expect, test, vi } from 'vitest';
import TimeSeriesPlot from '../components/TimeSeriesPlot';

const plotMock = vi.hoisted(() => ({
  props: [] as any[],
}));

vi.mock('next/dynamic', () => ({
  default: () => {
    const ReactRuntime = require('react') as typeof import('react');
    return function MockPlot(props: any) {
      plotMock.props.push(props);
      ReactRuntime.useEffect(() => {
        props.onInitialized?.({}, { data: props.data });
        props.onUpdate?.({}, { data: props.data });
      }, [props]);
      return (
        <>
          <button
            data-testid="plot-restyle"
            onClick={() => props.onRestyle?.({}, { data: props.data })}
          >
            restyle
          </button>
          <button
            data-testid="plot-restyle-event-only"
            onClick={() => props.onRestyle?.({})}
          >
            restyle event only
          </button>
        </>
      );
    };
  },
}));

const runData = {
  columns: ['time', 'sig_a', 'sig_b'],
  index: [0, 1],
  data: [
    [0, 10, 20],
    [1, 11, 21],
  ],
};

beforeEach(() => {
  plotMock.props = [];
});

test('initialization and update events do not overwrite selected signals', async () => {
  const onVisibleSignalsChange = vi.fn();

  render(
    <TimeSeriesPlot
      allRunsData={{ run1: runData }}
      selectedRunIds={['run1']}
      availableSignals={['sig_a', 'sig_b']}
      selectedSignals={['sig_a']}
      onVisibleSignalsChange={onVisibleSignalsChange}
    />
  );

  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

  expect(onVisibleSignalsChange).not.toHaveBeenCalled();
});

test('restyle events are the only plot events that update visible signals', async () => {
  const onVisibleSignalsChange = vi.fn();

  render(
    <TimeSeriesPlot
      allRunsData={{ run1: runData }}
      selectedRunIds={['run1']}
      availableSignals={['sig_a', 'sig_b']}
      selectedSignals={['sig_a']}
      onVisibleSignalsChange={onVisibleSignalsChange}
    />
  );

  fireEvent.click(screen.getByTestId('plot-restyle'));

  await waitFor(() => {
    expect(onVisibleSignalsChange).toHaveBeenCalledWith(['sig_a']);
  });
});

test('restyle events without a graph div argument still read the initialized graph data', async () => {
  const onVisibleSignalsChange = vi.fn();

  render(
    <TimeSeriesPlot
      allRunsData={{ run1: runData }}
      selectedRunIds={['run1']}
      availableSignals={['sig_a', 'sig_b']}
      selectedSignals={['sig_a']}
      onVisibleSignalsChange={onVisibleSignalsChange}
    />
  );

  fireEvent.click(screen.getByTestId('plot-restyle-event-only'));

  await waitFor(() => {
    expect(onVisibleSignalsChange).toHaveBeenCalledWith(['sig_a']);
  });
});

test('newly added runs inherit the current selected signal set', () => {
  const { rerender } = render(
    <TimeSeriesPlot
      allRunsData={{ run1: runData, run2: runData }}
      selectedRunIds={['run1']}
      availableSignals={['sig_a', 'sig_b']}
      selectedSignals={['sig_a']}
    />
  );

  rerender(
    <TimeSeriesPlot
      allRunsData={{ run1: runData, run2: runData }}
      selectedRunIds={['run1', 'run2']}
      availableSignals={['sig_a', 'sig_b']}
      selectedSignals={['sig_a']}
    />
  );

  const latestData = plotMock.props.at(-1)?.data ?? [];
  expect(latestData.map((trace: any) => ({
    runId: trace.meta.runId,
    signalKey: trace.meta.signalKey,
    visible: trace.visible,
  }))).toEqual([
    { runId: 'run1', signalKey: 'sig_a', visible: true },
    { runId: 'run1', signalKey: 'sig_b', visible: 'legendonly' },
    { runId: 'run2', signalKey: 'sig_a', visible: true },
    { runId: 'run2', signalKey: 'sig_b', visible: 'legendonly' },
  ]);
});
