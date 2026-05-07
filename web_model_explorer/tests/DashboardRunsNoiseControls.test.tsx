import React from 'react';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, expect, test, vi } from 'vitest';
import { DashboardControllerProvider } from '../components/dashboard/DashboardControllerContext';
import { DashboardMainRuns } from '../components/dashboard/view/main/DashboardMainRuns';
import { useDashboardStore } from '../components/dashboard/useDashboardStore';
import { makeRunsUIController } from './fixtures/controllers';

const plotPropsSpy = vi.fn();

vi.mock('../components/TimeSeriesPlot', () => {
  const Mock = (props: any) => {
    plotPropsSpy(props);
    return <div data-testid="plot" />;
  };
  return { default: Mock };
});

beforeEach(() => {
  plotPropsSpy.mockClear();
  useDashboardStore.setState({
    selectedModel: 'ModelA',
    loading: false,
    simulating: false,
    runsData: {},
    selectedRunIds: [],
    selectedSignals: ['Ball_Center_Position', 'Ball_Center_Speed'],
    signalDisplayNames: {
      Ball_Center_Position: 'Ball Position',
      Ball_Center_Speed: 'Ball Speed',
    },
    availableNoiseProfiles: ['none', 'low', 'high'],
    selectedNoiseProfile: 'none',
    noiseSeed: 0,
    availableSignals: ['Ball_Center_Position', 'Ball_Center_Speed'],
    expandedInterventions: [],
    bulkTimeByRunId: {},
    selectedRunRowIds: [],
    selectedParamKeys: [],
    samplingRate: 50,
    samplingRateDraft: '50',
    samplingRateDirty: false,
    samplingRateError: '',
    samplingRatePlotOverride: null,
    samplingDetails: {},
    exposedParameterKeys: [],
    exposedInitialStateKeys: [],
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [],
    } as any,
  });
});

test('plot controls show only documented noise profile and seed controls', () => {
  render(
    <DashboardControllerProvider controller={makeRunsUIController({ currentTimeSeriesSamplingRate: 50.125 })}>
      <DashboardMainRuns />
    </DashboardControllerProvider>
  );

  expect(screen.getByLabelText(/plot noise profile/i)).toBeTruthy();
  expect(screen.getByLabelText(/plot noise seed/i)).toBeTruthy();
  expect(screen.queryByText('Signals')).toBeNull();
  expect(screen.queryByLabelText(/global local-noise multiplier/i)).toBeNull();
  expect(screen.queryByLabelText(/global global-noise multiplier/i)).toBeNull();
  expect(screen.queryByLabelText(/global absolute-noise sigma/i)).toBeNull();
  expect(screen.queryByText(/plot sampling/i)).toBeNull();
  expect(screen.queryByText(/data:\s*50\.125 hz/i)).toBeNull();
  expect(screen.queryByText(/decimate to/i)).toBeNull();
  expect(screen.queryByRole('button', { name: /clear/i })).toBeNull();
});

test('plot header renders first selected run SNR values in dB when comparing two noisy runs', () => {
  useDashboardStore.setState({
    selectedRunIds: ['run1', 'run2'],
    selectedNoiseProfile: 'low',
    runsData: {
      run1: {
        columns: ['time', 'Ball_Center_Position'],
        index: [0],
        data: [[0, 1]],
        signal_analysis: {
          Ball_Center_Position: 12.3456,
          Ball_Center_Speed: '-inf',
        },
      },
      run2: {
        columns: ['time', 'Ball_Center_Position'],
        index: [0],
        data: [[0, 0]],
        signal_analysis: {
          Ball_Center_Position: 99,
        },
      },
    },
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [
        {
          run_id: 'run1',
          parent_id: null,
          parameters: {},
          intervention_time: 0,
          interventions: [],
          status: 'success',
          timestamp: 'r1',
        } as any,
        {
          run_id: 'run2',
          parent_id: null,
          parameters: {},
          intervention_time: 0,
          interventions: [],
          status: 'success',
          timestamp: 'r2',
        } as any,
      ],
    } as any,
  });

  render(
    <DashboardControllerProvider controller={makeRunsUIController({ currentTimeSeriesSamplingRate: 50.125 })}>
      <DashboardMainRuns />
    </DashboardControllerProvider>
  );

  expect(screen.getByText('Plot')).toBeInTheDocument();
  expect(screen.getAllByText('Ball Position')).toHaveLength(1);
  const snrRow = screen.getByLabelText('SNR by signal');
  const rows = screen.getAllByLabelText(/SNR by signal/i);
  expect(rows).toHaveLength(1);
  expect(rows[0]).toBe(snrRow);
  expect(within(snrRow).getByText('SNR')).toBeInTheDocument();
  expect(within(snrRow).getByText('12.35 dB')).toBeInTheDocument();
  expect(within(snrRow).getByText('-inf dB')).toBeInTheDocument();
  expect(screen.queryByText('99 dB')).toBeNull();
  expect(screen.queryByLabelText('GLOBAL SNR by signal')).toBeNull();
  expect(screen.queryByLabelText('LOCAL SNR by signal')).toBeNull();
});

test('plot header renders all first-run SNR entries', () => {
  const signalNames = Array.from({ length: 7 }, (_, idx) => `sig_${idx + 1}`);
  useDashboardStore.setState({
    selectedRunIds: ['run1', 'run2'],
    selectedNoiseProfile: 'low',
    selectedSignals: signalNames,
    availableSignals: signalNames,
    signalDisplayNames: Object.fromEntries(
      signalNames.map((signal, idx) => [signal, `Signal ${idx + 1}`])
    ),
    runsData: {
      run1: {
        columns: ['time', ...signalNames],
        index: [0],
        data: [[0, ...signalNames.map((_, idx) => idx)]],
        signal_analysis: Object.fromEntries(
          signalNames.map((signal, idx) => [
            signal,
            idx + 11,
          ])
        ),
      },
      run2: {
        columns: ['time', ...signalNames],
        index: [0],
        data: [[0, ...signalNames.map((_, idx) => idx)]],
        signal_analysis: Object.fromEntries(
          signalNames.map((signal, idx) => [
            signal,
            idx + 101,
          ])
        ),
      },
    },
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [
        {
          run_id: 'run1',
          parent_id: null,
          parameters: {},
          intervention_time: 0,
          interventions: [],
          status: 'success',
          timestamp: 'r1',
        } as any,
        {
          run_id: 'run2',
          parent_id: null,
          parameters: {},
          intervention_time: 0,
          interventions: [],
          status: 'success',
          timestamp: 'r2',
        } as any,
      ],
    } as any,
  });

  render(
    <DashboardControllerProvider controller={makeRunsUIController({ currentTimeSeriesSamplingRate: 50.125 })}>
      <DashboardMainRuns />
    </DashboardControllerProvider>
  );

  expect(screen.getByText('17 dB')).toBeInTheDocument();
  expect(screen.queryByText('107 dB')).toBeNull();
  expect(screen.queryByText('7 dB')).toBeNull();
  expect(screen.getByLabelText('SNR by signal')).toBeInTheDocument();
  expect(screen.queryByLabelText('GLOBAL SNR by signal')).toBeNull();
  expect(screen.queryByLabelText('LOCAL SNR by signal')).toBeNull();
});

test('plot header hides SNR unless noise is active and exactly two runs are selected', () => {
  useDashboardStore.setState({
    selectedRunIds: ['run1', 'run2'],
    selectedNoiseProfile: 'none',
    runsData: {
      run1: {
        columns: ['time', 'Ball_Center_Position'],
        index: [0],
        data: [[0, 1]],
        signal_analysis: {
          Ball_Center_Position: 12,
        },
      },
    },
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [
        { run_id: 'run1', parameters: {}, interventions: [], status: 'success', timestamp: 'r1' } as any,
        { run_id: 'run2', parameters: {}, interventions: [], status: 'success', timestamp: 'r2' } as any,
      ],
    } as any,
  });

  const { rerender } = render(
    <DashboardControllerProvider controller={makeRunsUIController({ currentTimeSeriesSamplingRate: 50.125 })}>
      <DashboardMainRuns />
    </DashboardControllerProvider>
  );

  expect(screen.queryByLabelText('SNR by signal')).toBeNull();

  useDashboardStore.setState({
    selectedNoiseProfile: 'low',
    selectedRunIds: ['run1', 'run2', 'run3'],
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [
        { run_id: 'run1', parameters: {}, interventions: [], status: 'success', timestamp: 'r1' } as any,
        { run_id: 'run2', parameters: {}, interventions: [], status: 'success', timestamp: 'r2' } as any,
        { run_id: 'run3', parameters: {}, interventions: [], status: 'success', timestamp: 'r3' } as any,
      ],
    } as any,
  });
  rerender(
    <DashboardControllerProvider controller={makeRunsUIController({ currentTimeSeriesSamplingRate: 50.125 })}>
      <DashboardMainRuns />
    </DashboardControllerProvider>
  );

  expect(screen.queryByLabelText('SNR by signal')).toBeNull();
});

test('noise profile and seed controls update store', async () => {
  render(
    <DashboardControllerProvider controller={makeRunsUIController({ currentTimeSeriesSamplingRate: 50.125 })}>
      <DashboardMainRuns />
    </DashboardControllerProvider>
  );

  fireEvent.change(screen.getByLabelText(/plot noise profile/i), {
    target: { value: 'high' },
  });
  fireEvent.change(screen.getByLabelText(/plot noise seed/i), {
    target: { value: '7' },
  });

  await waitFor(() => {
    expect(useDashboardStore.getState().selectedNoiseProfile).toBe('high');
    expect(useDashboardStore.getState().noiseSeed).toBe(7);
  });

  const latestProps = plotPropsSpy.mock.calls.at(-1)?.[0];
  expect(latestProps?.noiseSeed).toBeUndefined();
  expect(latestProps?.signalNoiseMapping).toBeUndefined();
});

test('plot legend visibility callback updates selected signals', async () => {
  render(
    <DashboardControllerProvider controller={makeRunsUIController({ currentTimeSeriesSamplingRate: 50.125 })}>
      <DashboardMainRuns />
    </DashboardControllerProvider>
  );

  const latestProps = plotPropsSpy.mock.calls.at(-1)?.[0];
  expect(latestProps?.availableSignals).toEqual([
    'Ball_Center_Position',
    'Ball_Center_Speed',
  ]);

  await act(async () => {
    latestProps?.onVisibleSignalsChange?.(['Ball_Center_Position']);
  });

  await waitFor(() => {
    expect(useDashboardStore.getState().selectedSignals).toEqual(['Ball_Center_Position']);
  });

  await act(async () => {
    latestProps?.onVisibleSignalsChange?.([]);
  });

  await waitFor(() => {
    expect(useDashboardStore.getState().selectedSignals).toEqual([]);
  });
});
