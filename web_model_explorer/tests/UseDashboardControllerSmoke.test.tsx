import { act, renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, test, vi } from 'vitest';
import axios from 'axios';
import { useDashboardController } from '../components/dashboard/useDashboardController';
import { useDashboardStore } from '../components/dashboard/useDashboardStore';

vi.mock('axios');

const mockedAxios = vi.mocked(axios, true);

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

describe('useDashboardController', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(window, 'alert').mockImplementation(() => {});
    useDashboardStore.setState({
      models: [],
      modelValidation: {},
      selectedModel: '',
      modelRecord: null,
      originalModelRecord: null,
      diskRuns: [],
      loading: false,
      simulating: false,
      runsData: {},
      selectedRunIds: [],
      selectedSignals: [],
      signalDisplayNames: {},
      availableNoiseProfiles: ['none'],
      selectedNoiseProfile: 'none',
      noiseSeed: 0,
      selectedSignalCases: [],
      availableSignals: [],
      expandedInterventions: [],
      history: [],
      historyIndex: -1,
      saveNotice: '',
      simulationNotice: null,
      bulkTimeByRunId: {},
      signalsFetchInFlight: false,
      selectedRunRowIds: [],
      selectedParamKeys: [],
      samplingRate: null,
      samplingRateDraft: '',
      samplingRateDirty: false,
      samplingRateError: '',
      samplingRatePlotOverride: null,
      samplingIntervals: {},
      samplingPerturbationIntervals: {},
      samplingDetails: {},
      exposedParameterKeys: [],
      exposedInitialStateKeys: [],
    });
    mockedAxios.get.mockResolvedValue({ data: { models: [] } } as any);
  });

  test('renders without stale runtime references', () => {
    mockedAxios.get.mockReturnValue(new Promise(() => {}) as any);
    expect(() => renderHook(() => useDashboardController())).not.toThrow();
  });

  test('registry load initializes all available signals as selected for visible plots', async () => {
    mockedAxios.get.mockImplementation((url: any) => {
      const rawUrl = String(url);
      if (rawUrl.startsWith('/api/models')) {
        return Promise.resolve({ data: { models: ['ModelA'] } } as any);
      }
      if (rawUrl.startsWith('/api/registry')) {
        return Promise.resolve({
          data: {
            modelRecord: { metadata: {}, baselines: [] },
            diskRuns: [],
            availableNoiseProfiles: ['none'],
            availableSignals: ['sig_a', 'sig_b'],
          },
        } as any);
      }
      if (rawUrl.startsWith('/api/distribution')) {
        return Promise.resolve({ data: { distribution: {} } } as any);
      }
      return Promise.resolve({ data: {} } as any);
    });

    useDashboardStore.setState({
      selectedModel: 'ModelA',
      selectedSignals: [],
      availableSignals: [],
    });

    renderHook(() => useDashboardController());

    await waitFor(() => {
      expect(useDashboardStore.getState().availableSignals).toEqual(['sig_a', 'sig_b']);
      expect(useDashboardStore.getState().selectedSignals).toEqual(['sig_a', 'sig_b']);
    });
  });

  test('does not reselect a run when its delayed data load completes after user toggles it off', async () => {
    const dataLoad = deferred<any>();
    mockedAxios.get.mockImplementation((url: any) => {
      if (String(url).startsWith('/api/data')) return dataLoad.promise;
      return Promise.resolve({ data: { models: [] } } as any);
    });
    const { result } = renderHook(() => useDashboardController());

    act(() => {
      useDashboardStore.setState({ diskRuns: ['run1'] });
    });

    let firstToggle!: Promise<void>;
    act(() => {
      firstToggle = result.current.toggleRunVisualization('run1') as Promise<void>;
    });
    expect(useDashboardStore.getState().selectedRunIds).toEqual(['run1']);

    await act(async () => {
      await result.current.toggleRunVisualization('run1');
    });
    expect(useDashboardStore.getState().selectedRunIds).toEqual([]);

    await act(async () => {
      dataLoad.resolve({
        status: 200,
        data: { columns: ['time', 'sig_a'], index: [0], data: [[0, 1]] },
      });
      await firstToggle;
    });

    expect(useDashboardStore.getState().selectedRunIds).toEqual([]);
  });

  test('does not reselect a run group when delayed data loads complete after user toggles it off', async () => {
    const dataLoads: Array<ReturnType<typeof deferred<any>>> = [];
    mockedAxios.get.mockImplementation((url: any) => {
      if (String(url).startsWith('/api/data')) {
        const load = deferred<any>();
        dataLoads.push(load);
        return load.promise;
      }
      return Promise.resolve({ data: { models: [] } } as any);
    });
    const { result } = renderHook(() => useDashboardController());

    act(() => {
      useDashboardStore.setState({ diskRuns: ['baseline1', 'child1'] });
    });

    const run = {
      run_id: 'baseline1',
      interventions: [{ name: 'child1' }],
    } as any;

    let firstToggle!: Promise<void>;
    act(() => {
      firstToggle = result.current.toggleRunVisualizationGroup(run) as Promise<void>;
    });
    expect(useDashboardStore.getState().selectedRunIds).toEqual(['baseline1', 'child1']);

    await act(async () => {
      await result.current.toggleRunVisualizationGroup(run);
    });
    expect(useDashboardStore.getState().selectedRunIds).toEqual([]);

    await act(async () => {
      for (const load of dataLoads) {
        load.resolve({
          status: 200,
          data: { columns: ['time', 'sig_a'], index: [0], data: [[0, 1]] },
        });
      }
      await firstToggle;
    });

    expect(useDashboardStore.getState().selectedRunIds).toEqual([]);
  });

  test('baseline group toggle includes linked time0 run ids', async () => {
    mockedAxios.get.mockImplementation((url: any) => {
      if (String(url).startsWith('/api/data')) {
        return Promise.resolve({
          status: 200,
          data: { columns: ['time', 'sig_a'], index: [0], data: [[0, 1]] },
        } as any);
      }
      return Promise.resolve({ data: { models: [] } } as any);
    });
    const { result } = renderHook(() => useDashboardController());

    act(() => {
      useDashboardStore.setState({ diskRuns: ['baseline1', 'child1', 'time0_1'] });
    });

    const run = {
      run_id: 'baseline1',
      interventions: [{ name: 'child1', time0_baseline_uuid: 'time0_1' }],
    } as any;

    await act(async () => {
      await result.current.toggleRunVisualizationGroup(run);
    });
    expect(useDashboardStore.getState().selectedRunIds).toEqual(['baseline1', 'child1', 'time0_1']);

    await act(async () => {
      await result.current.toggleRunVisualizationGroup(run);
    });
    expect(useDashboardStore.getState().selectedRunIds).toEqual([]);
  });

  test('fetches the first selected noisy run with the second selected run as reference', async () => {
    const dataUrls: string[] = [];
    mockedAxios.get.mockImplementation((url: any) => {
      const rawUrl = String(url);
      if (rawUrl.startsWith('/api/data')) {
        dataUrls.push(rawUrl);
        return Promise.resolve({
          status: 200,
          data: { columns: ['time', 'sig_a'], index: [0], data: [[0, 1]] },
        } as any);
      }
      return Promise.resolve({ data: { models: [] } } as any);
    });

    useDashboardStore.setState({
      selectedModel: 'ModelA',
      diskRuns: ['run1', 'run2'],
      selectedRunIds: ['run1', 'run2'],
      availableNoiseProfiles: ['none', 'low'],
      selectedNoiseProfile: 'low',
      noiseSeed: 7,
    });

    renderHook(() => useDashboardController());

    await waitFor(() => {
      expect(dataUrls.filter((url) => url.startsWith('/api/data')).length).toBeGreaterThanOrEqual(2);
    });

    const runParam = (url: string) => new URL(url, 'http://localhost').searchParams.get('run');
    const firstRunUrl = dataUrls.find((url) => runParam(url) === 'run1') || '';
    const secondRunUrl = dataUrls.find((url) => runParam(url) === 'run2') || '';
    expect(firstRunUrl).toContain('reference_run=run2');
    expect(firstRunUrl).toContain('noise_profile=low');
    expect(firstRunUrl).toContain('noise_seed=7');
    expect(secondRunUrl).not.toContain('reference_run=');
  });

  test('removes only failed run IDs after an optimistic group selection load failure', async () => {
    mockedAxios.get.mockImplementation((url: any) => {
      const rawUrl = String(url);
      if (rawUrl.startsWith('/api/data') && rawUrl.includes('run=child1')) {
        return Promise.resolve({ status: 400, data: { error: 'missing child' } } as any);
      }
      if (rawUrl.startsWith('/api/data')) {
        return Promise.resolve({
          status: 200,
          data: { columns: ['time', 'sig_a'], index: [0], data: [[0, 1]] },
        } as any);
      }
      return Promise.resolve({ data: { models: [] } } as any);
    });
    const { result } = renderHook(() => useDashboardController());

    act(() => {
      useDashboardStore.setState({ diskRuns: ['baseline1', 'child1'] });
    });

    await act(async () => {
      await result.current.toggleRunVisualizationGroup({
        run_id: 'baseline1',
        interventions: [{ name: 'child1' }],
      } as any);
    });

    expect(useDashboardStore.getState().selectedRunIds).toEqual(['baseline1']);
    expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('child1: missing child'));
  });

  test('run data API JSON error is preferred over a generic axios message', async () => {
    mockedAxios.get.mockImplementation((url: any) => {
      const rawUrl = String(url);
      if (rawUrl.startsWith('/api/data')) {
        return Promise.reject({
          message: 'Request failed with status code 500',
          response: {
            status: 500,
            data: { error: 'noise_adder failed for run1' },
          },
          config: { url: rawUrl },
        });
      }
      return Promise.resolve({ data: { models: [] } } as any);
    });
    const { result } = renderHook(() => useDashboardController());

    act(() => {
      useDashboardStore.setState({ diskRuns: ['run1'] });
    });

    await act(async () => {
      await result.current.toggleRunVisualization('run1');
    });

    expect(window.alert).toHaveBeenCalledWith('noise_adder failed for run1');
    expect(useDashboardStore.getState().selectedRunIds).toEqual([]);
  });

  test('run data XHR failure is shown with endpoint context instead of raw network error', async () => {
    mockedAxios.get.mockImplementation((url: any) => {
      const rawUrl = String(url);
      if (rawUrl.startsWith('/api/data')) {
        return Promise.reject({
          message: 'Network Error',
          request: {},
          config: { url: rawUrl },
        });
      }
      return Promise.resolve({ data: { models: [] } } as any);
    });
    const { result } = renderHook(() => useDashboardController());

    act(() => {
      useDashboardStore.setState({ diskRuns: ['run1'] });
    });

    await act(async () => {
      await result.current.toggleRunVisualization('run1');
    });

    expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('Failed to load run data'));
    expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('/api/data?model='));
    expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('documented host and port'));
    expect(window.alert).not.toHaveBeenCalledWith('Network Error');
    expect(useDashboardStore.getState().selectedRunIds).toEqual([]);
  });

  test('registry load failure alerts and leaves registry-bound state empty', async () => {
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    mockedAxios.get.mockImplementation((url: any) => {
      const rawUrl = String(url);
      if (rawUrl.startsWith('/api/models')) {
        return Promise.resolve({ data: { models: ['ModelA'] } } as any);
      }
      if (rawUrl.startsWith('/api/registry')) {
        return Promise.reject({
          response: {
            status: 500,
            data: { error: 'model_run_specs.json is invalid' },
          },
          config: { url: rawUrl },
        });
      }
      if (rawUrl.startsWith('/api/distribution')) {
        return Promise.resolve({ data: { distribution: {} } } as any);
      }
      return Promise.resolve({ data: {} } as any);
    });

    useDashboardStore.setState({
      selectedModel: 'ModelA',
      modelRecord: { metadata: {}, baselines: [{ run_id: 'old' }] } as any,
      diskRuns: ['old'],
      selectedRunIds: [],
      availableSignals: ['sig_old'],
      selectedSignals: ['sig_old'],
      runsData: {
        old: { columns: ['time', 'sig_old'], index: [0], data: [[0, 1]] },
      },
    });

    renderHook(() => useDashboardController());

    await waitFor(() => {
      expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('Failed to load registry for model'));
      expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('model_run_specs.json is invalid'));
    });
    expect(useDashboardStore.getState().modelRecord).toBeNull();
    expect(useDashboardStore.getState().diskRuns).toEqual([]);
    expect(useDashboardStore.getState().selectedRunIds).toEqual([]);
    expect(useDashboardStore.getState().availableSignals).toEqual([]);
    expect(useDashboardStore.getState().runsData).toEqual({});
    consoleErrorSpy.mockRestore();
  });

  test('refetches selected noisy runs when the seed changes', async () => {
    const dataUrls: string[] = [];
    mockedAxios.get.mockImplementation((url: any) => {
      const rawUrl = String(url);
      if (rawUrl.startsWith('/api/models')) {
        return Promise.resolve({ data: { models: ['ModelA'] } } as any);
      }
      if (rawUrl.startsWith('/api/registry')) {
        return Promise.resolve({
          data: {
            modelRecord: { metadata: {}, baselines: [] },
            diskRuns: ['run1'],
            availableNoiseProfiles: ['none', 'low'],
            availableSignals: ['sig_a'],
          },
        } as any);
      }
      if (rawUrl.startsWith('/api/distribution')) {
        return Promise.resolve({ data: { distribution: {} } } as any);
      }
      if (rawUrl.startsWith('/api/data')) {
        dataUrls.push(rawUrl);
        return Promise.resolve({
          status: 200,
          data: { columns: ['time', 'sig_a'], index: [0], data: [[0, 1]] },
        } as any);
      }
      return Promise.resolve({ data: {} } as any);
    });

    useDashboardStore.setState({ selectedModel: 'ModelA' });
    const { result } = renderHook(() => useDashboardController());

    await waitFor(() => {
      expect(useDashboardStore.getState().diskRuns).toEqual(['run1']);
    });

    act(() => {
      useDashboardStore.getState().setSelectedNoiseProfile('low');
    });

    await waitFor(() => {
      expect(useDashboardStore.getState().selectedNoiseProfile).toBe('low');
    });

    await act(async () => {
      await result.current.toggleRunVisualization('run1');
    });

    await waitFor(() => {
      expect(dataUrls.some((url) => url.includes('noise_seed=0'))).toBe(true);
    });

    act(() => {
      useDashboardStore.getState().setNoiseSeed(7);
    });

    await waitFor(() => {
      expect(dataUrls.some((url) => url.includes('noise_seed=7'))).toBe(true);
    });
  });
});
