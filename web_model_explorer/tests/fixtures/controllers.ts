import { vi } from 'vitest';
import { useDashboardStore } from '../../components/dashboard/useDashboardStore';

export const makeRunsUIController = (overrides: Record<string, any> = {}) =>
  ({
    currentTimeSeriesSamplingRate: null,
    canVisualizeSamplingRate: true,
    handleVisibleSignalsChange: vi.fn((visibleSignals: string[]) => {
      useDashboardStore.getState().setSelectedSignals(visibleSignals);
    }),
    updateRegistry: vi.fn(),
    updateRunRole: vi.fn(),
    handleRunDragStart: vi.fn(),
    handleRunDragOver: vi.fn(),
    handleRunDrop: vi.fn(),
    handleRunDragEnd: vi.fn(),
    hasRunData: vi.fn(() => false),
    formatTimeLabel: (value: any) => String(value ?? ''),
    formatVariableLabel: (value: any) => String(value ?? ''),
    getInterventionKey: vi.fn((runId: string, ivName: string) => `${runId}::${ivName}`),
    isValidNumberInput: vi.fn(() => true),
    getRunRole: vi.fn(() => null),
    isFewShotRun: vi.fn(() => false),
    ensureRunsLoaded: vi.fn(async () => {}),
    handleQuestionSelect: vi.fn(),
    runNoiseById: {},
    getPreviousValue: vi.fn(() => 0),
    toggleRunVisualization: vi.fn(),
    toggleRunVisualizationGroup: vi.fn(),
    handleClone: vi.fn(),
    toggleRunRowSelection: vi.fn(),
    handleAddRun: vi.fn(),
    handleResetSelectedRuns: vi.fn(),
    openParamEditor: vi.fn(),
    toggleParamSelection: vi.fn(),
    handleSaveSamplingSpec: vi.fn(),
    handleResetSamplingEditor: vi.fn(),
    handleSaveSamplingRate: vi.fn(),
    handleResetSamplingRate: vi.fn(),
    handleVisualizeSamplingRate: vi.fn(),
    handleDelete: vi.fn(),
    handleParamChange: vi.fn(),
    handleAddIntervention: vi.fn(),
    handleInterventionChange: vi.fn(),
    applyBulkTimeToRun: vi.fn(),
    handleRemoveIntervention: vi.fn(),
    paramKeys: ['x'],
    toggleInterventionSelection: vi.fn(),
    openClassificationModal: vi.fn(),
    shouldIgnoreRowClick: (target: EventTarget | null) => {
      if (!(target instanceof Element)) return false;
      return Boolean(target.closest('button, input, select, textarea, a, svg'));
    },
    ...overrides,
  }) as any;
