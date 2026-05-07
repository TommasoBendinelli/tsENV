import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, expect, test, vi } from 'vitest';
import { DashboardControllerProvider } from '../components/dashboard/DashboardControllerContext';
import { DashboardMainRuns } from '../components/dashboard/view/main/DashboardMainRuns';
import { useDashboardStore } from '../components/dashboard/useDashboardStore';
import { makeRunsUIController } from './fixtures/controllers';

vi.mock('../components/TimeSeriesPlot', () => {
  return {
    default: () => null,
  };
});

beforeEach(() => {
  useDashboardStore.setState({
    selectedModel: 'ModelA',
    loading: false,
    simulating: false,
    runsData: {},
    selectedRunIds: [],
    selectedSignals: [],
    signalDisplayNames: {},
    availableNoiseProfiles: ['none'],
    selectedNoiseProfile: 'none',
    availableSignals: [],
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
    exposedParameterKeys: ['x'],
    exposedInitialStateKeys: ['initial_height'],
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [
        {
          run_id: '1',
          parent_id: null,
          parameters: { x: 0, initial_height: 2 },
          intervention_time: 7,
          end_time_input_s: 10,
          end_time_simulation: 9.990050315856934,
          interventions: [
            {
              name: 'x_001',
              parent_id: '1',
              depth: 1,
              intervention_time: 7,
              variable: 'x',
              value: 1,
              end_time_input_s: 10,
              end_time_simulation: 9.990050315856934,
              time0_end_time_simulation: 9.990050315856934,
              status: 'not_run',
              timestamp: 't1',
            },
          ],
          status: 'not_run',
          timestamp: 'r1',
          sampling_rate_hz: 50,
        } as any,
      ],
    } as any,
  });
});

const renderRuns = (controllerOverrides: Record<string, unknown> = {}) => render(
  <DashboardControllerProvider controller={makeRunsUIController(controllerOverrides)}>
    <DashboardMainRuns />
  </DashboardControllerProvider>
);

test('clicking a baseline row expands interventions', async () => {
  const user = userEvent.setup();
  const nonActConsoleErrors: unknown[][] = [];
  const isActWarning = (args: unknown[]) => String(args[0] ?? '').includes('not wrapped in act');
  const errorSpy = vi.spyOn(console, 'error').mockImplementation((...args: unknown[]) => {
    if (isActWarning(args)) return;
    nonActConsoleErrors.push(args);
  });

  renderRuns();

  expect(screen.queryByText(/Simulation Steps & Interventions/i)).not.toBeInTheDocument();

  const rows = screen.getAllByRole('row');
  const baselineRow = rows.find((row) => {
    try {
      return Boolean(within(row).queryByText('1'));
    } catch {
      return false;
    }
  });
  expect(baselineRow).toBeTruthy();

  await user.click(baselineRow!);

  expect(screen.getByText(/Simulation Steps & Interventions/i)).toBeInTheDocument();
  expect(nonActConsoleErrors).toEqual([]);

  errorSpy.mockRestore();
});

test('collapsing a baseline row removes selected child and time0 plots', async () => {
  const user = userEvent.setup();
  useDashboardStore.setState({
    expandedInterventions: ['1'],
    selectedRunIds: ['1', 'x_001', 'time0_001'],
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [
        {
          run_id: '1',
          parent_id: null,
          parameters: { x: 0 },
          intervention_time: 7,
          interventions: [
            {
              name: 'x_001',
              parent_id: '1',
              depth: 1,
              intervention_time: 7,
              variable: 'x',
              value: 1,
              time0_baseline_uuid: 'time0_001',
              status: 'success',
              timestamp: 't1',
            },
          ],
          status: 'success',
          timestamp: 'r1',
        } as any,
      ],
    } as any,
  });

  renderRuns();
  expect(screen.getByText(/Simulation Steps & Interventions/i)).toBeInTheDocument();

  await act(async () => {
    await user.click(screen.getByTitle('1').closest('tr')!);
  });

  expect(useDashboardStore.getState().expandedInterventions).toEqual([]);
  expect(useDashboardStore.getState().selectedRunIds).toEqual(['1']);
});

test('collapsing with the interventions button removes selected child and time0 plots', async () => {
  const user = userEvent.setup();
  useDashboardStore.setState({
    expandedInterventions: ['1'],
    selectedRunIds: ['1', 'x_001', 'time0_001'],
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [
        {
          run_id: '1',
          parent_id: null,
          parameters: { x: 0 },
          intervention_time: 7,
          interventions: [
            {
              name: 'x_001',
              parent_id: '1',
              depth: 1,
              intervention_time: 7,
              variable: 'x',
              value: 1,
              time0_baseline_uuid: 'time0_001',
              status: 'success',
              timestamp: 't1',
            },
          ],
          status: 'success',
          timestamp: 'r1',
        } as any,
      ],
    } as any,
  });

  renderRuns();

  await act(async () => {
    await user.click(screen.getByTitle('Show interventions'));
  });

  expect(useDashboardStore.getState().expandedInterventions).toEqual([]);
  expect(useDashboardStore.getState().selectedRunIds).toEqual(['1']);
});

test('runs toolbar omits configuration header and write controls', () => {
  renderRuns();

  expect(screen.queryByText(/Configuration & Runs/i)).not.toBeInTheDocument();
  expect(screen.queryByLabelText(/Include Unit Test Cases/i)).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /New Run/i })).not.toBeInTheDocument();
});

test('baseline subtree plot button is enabled when only linked time0 data exists', async () => {
  const user = userEvent.setup();
  const toggleRunVisualizationGroup = vi.fn();
  useDashboardStore.setState({
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [
        {
          run_id: '1',
          parent_id: null,
          parameters: { x: 0 },
          intervention_time: 7,
          interventions: [
            {
              name: 'x_001',
              parent_id: '1',
              depth: 1,
              intervention_time: 7,
              variable: 'x',
              value: 1,
              time0_baseline_uuid: 'time0_001',
              status: 'success',
              timestamp: 't1',
            },
          ],
          status: 'success',
          timestamp: 'r1',
        } as any,
      ],
    } as any,
  });

  renderRuns({
    hasRunData: vi.fn((runId: string) => runId === 'time0_001'),
    toggleRunVisualizationGroup,
  });

  const button = screen.getByTitle('Toggle baseline, child, and time0 plots');
  expect(button).not.toBeDisabled();
  await user.click(button);

  expect(toggleRunVisualizationGroup).toHaveBeenCalledTimes(1);
});

test('clicking Run ID text does not expand interventions', async () => {
  const user = userEvent.setup();

  renderRuns();

  expect(screen.queryByText(/Simulation Steps & Interventions/i)).not.toBeInTheDocument();

  await user.click(screen.getByTitle('1'));

  expect(screen.queryByText(/Simulation Steps & Interventions/i)).not.toBeInTheDocument();
});

test('runs table orders columns as Status Eligible Run ID parameters initial state Actions', () => {
  useDashboardStore.setState({
    exposedParameterKeys: ['mass', 'drag_coeff'],
    exposedInitialStateKeys: ['initial_height', 'initial_velocity'],
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [
        {
          run_id: '1',
          parent_id: null,
          parameters: {
            mass: 1.2,
            drag_coeff: 0.5,
            initial_height: 4,
            initial_velocity: -2,
          },
          intervention_time: 7,
          interventions: [],
          status: 'not_run',
          timestamp: 'r1',
        } as any,
      ],
    } as any,
  });

  renderRuns();

  const headers = screen.getAllByRole('columnheader').map((h) => (h.textContent || '').trim());
  expect(headers).toEqual([
    'Status',
    'Eligible',
    'Run ID',
    'mass',
    'drag_coeff',
    'initial_height',
    'initial_velocity',
    'Actions',
  ]);
  expect(headers).not.toContain('Hz');
  expect(headers).not.toContain('Ending Time');
  expect(headers).not.toContain('Measured');
});

test('runs table renders Eligible from baseline eligibility metrics', () => {
  useDashboardStore.setState({
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [
        {
          run_id: '1',
          parent_id: null,
          parameters: { x: 0 },
          intervention_time: 7,
          eligible: true,
          interventions: [],
          status: 'success',
          timestamp: 'r1',
        } as any,
      ],
    } as any,
  });

  renderRuns();

  expect(screen.getByLabelText('Eligible')).toBeInTheDocument();
  expect(screen.queryByText('Eligible')).toBeInTheDocument();
});

test('runs table renders Not Eligible and Unknown from baseline eligibility metrics', () => {
  useDashboardStore.setState({
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [
        {
          run_id: '1',
          parent_id: null,
          parameters: { x: 0 },
          intervention_time: 7,
          eligible: false,
          interventions: [],
          status: 'success',
          timestamp: 'r1',
        } as any,
        {
          run_id: '2',
          parent_id: null,
          parameters: { x: 1 },
          intervention_time: 7,
          interventions: [],
          status: 'success',
          timestamp: 'r2',
        } as any,
      ],
    } as any,
  });

  renderRuns();

  expect(screen.getByLabelText('Not Eligible')).toBeInTheDocument();
  expect(screen.getByLabelText('Eligibility Unknown')).toBeInTheDocument();
  expect(screen.queryByText('Not Eligible')).not.toBeInTheDocument();
  expect(screen.queryByText('Unknown')).not.toBeInTheDocument();
});

test('expanded child row shows detectability cross marker', () => {
  useDashboardStore.setState({
    expandedInterventions: ['1'],
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [
        {
          run_id: '1',
          parent_id: null,
          parameters: { x: 0 },
          intervention_time: 7,
          interventions: [
            {
              name: 'x_001',
              parent_id: '1',
              depth: 1,
              intervention_time: 7,
              variable: 'x',
              value: 1,
              status: 'success',
              detectability_failed: true,
              timestamp: 't1',
            },
          ],
          status: 'success',
          timestamp: 'r1',
        } as any,
      ],
    } as any,
  });

  renderRuns();

  expect(screen.getByLabelText('Child not detectable over baseline or time0 baseline')).toBeInTheDocument();
});

test('expanded child row omits detectability cross marker for runtime failure alone', () => {
  useDashboardStore.setState({
    expandedInterventions: ['1'],
    modelRecord: {
      version: 1,
      model_id: 'ModelA',
      metadata: {},
      baselines: [
        {
          run_id: '1',
          parent_id: null,
          parameters: { x: 0 },
          intervention_time: 7,
          interventions: [
            {
              name: 'x_001',
              parent_id: '1',
              depth: 1,
              intervention_time: 7,
              variable: 'x',
              value: 1,
              time0_baseline_uuid: 'time0_001',
              status: 'failed',
              time0_status: 'failed',
              timestamp: 't1',
            },
          ],
          status: 'failed',
          timestamp: 'r1',
        } as any,
      ],
    } as any,
  });

  renderRuns();

  expect(screen.queryByLabelText('Child not detectable over baseline or time0 baseline')).not.toBeInTheDocument();
});

test('expanded detectable child row omits detectability cross marker', () => {
  useDashboardStore.setState({ expandedInterventions: ['1'] });

  renderRuns();

  expect(screen.queryByLabelText('Child not detectable over baseline or time0 baseline')).not.toBeInTheDocument();
});

test('run id column keeps width classes and text selection behavior', () => {
  const runId = '1';
  renderRuns();

  const header = screen.getByRole('columnheader', { name: /run id/i });
  expect(header.className).toContain('w-56');
  expect(header.className).toContain('min-w-[220px]');

  const runSpan = screen.getByTitle(runId);
  const cell = runSpan.closest('td');
  expect(cell).toBeTruthy();
  expect(cell!.className).toContain('w-56');
  expect(cell!.className).toContain('min-w-[220px]');
  expect(cell!.className).toContain('cursor-text');
  expect(runSpan.className).toContain('cursor-text');
});

test('runs table uses Status as first column and shows child count', () => {
  renderRuns();

  const headers = screen.getAllByRole('columnheader');
  const headerTexts = headers.map((h) => (h.textContent || '').trim());
  expect(headerTexts[0]).toBe('Status');
  expect(headerTexts).not.toContain('Sel');
  expect(headerTexts).not.toContain('Viz');
  expect(screen.getByText('1 child')).toBeInTheDocument();
  expect(screen.queryByText(/^\d+\s+runs$/i)).not.toBeInTheDocument();
});

test('baseline parameter value 0 is shown and not rendered as empty', () => {
  renderRuns();

  const runSpan = screen.getByTitle('1');
  const row = runSpan.closest('tr');
  expect(row).toBeTruthy();

  expect(row!.textContent).toContain('0');
});
