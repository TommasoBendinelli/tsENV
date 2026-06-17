'use client';

import { create } from 'zustand';
import type {
  ModelRecord,
  RegistryPageInfo,
  SamplingSpec,
} from './types';

type SamplingDetail = {
  initial?: SamplingSpec;
  perturbation?: SamplingSpec;
  interval?: SamplingSpec;
};

type Setter<T> = T | ((prev: T) => T);

function resolveSetter<T>(next: Setter<T>, prev: T): T {
  if (typeof next === 'function') {
    return (next as (value: T) => T)(prev);
  }
  return next;
}

type DashboardState = {
  models: string[];
  modelValidation: Record<string, { ok: boolean; reasons: string[] }>;
  selectedModel: string;
  policies: string[];
  selectedPolicy: string;
  modelRecord: ModelRecord | null;
  originalModelRecord: ModelRecord | null;
  registryPage: RegistryPageInfo | null;
  loadingFamilyIds: string[];
  diskRuns: string[];
  loading: boolean;
  simulating: boolean;
  runsData: Record<string, any>;
  selectedRunIds: string[];
  selectedSignals: string[];
  signalDisplayNames: Record<string, string>;
  availableNoiseProfiles: string[];
  selectedNoiseProfile: string;
  noiseSeed: number;
  selectedSignalCases: string[][];
  availableSignals: string[];
  expandedInterventions: string[];
  history: ModelRecord[];
  historyIndex: number;
  saveNotice: string;
  simulationNotice: { done: number; missing: number; total?: number; running?: boolean } | null;
  bulkTimeByRunId: Record<string, string>;
  signalsFetchInFlight: boolean;
  selectedRunRowIds: string[];
  selectedParamKeys: string[];
  samplingRate: number | null;
  samplingRateDraft: string;
  samplingRateDirty: boolean;
  samplingRateError: string;
  samplingRatePlotOverride: number | null;
  samplingIntervals: Record<string, SamplingSpec>;
  samplingPerturbationIntervals: Record<string, SamplingSpec>;
  samplingDetails: Record<string, SamplingDetail>;
  exposedParameterKeys: string[];
  exposedInitialStateKeys: string[];
};

type DashboardActions = {
  setModels: (next: Setter<string[]>) => void;
  setModelValidation: (next: Setter<DashboardState['modelValidation']>) => void;
  setSelectedModel: (next: Setter<string>) => void;
  setPolicies: (next: Setter<string[]>) => void;
  setSelectedPolicy: (next: Setter<string>) => void;
  setModelRecord: (next: Setter<ModelRecord | null>) => void;
  setOriginalModelRecord: (next: Setter<ModelRecord | null>) => void;
  setRegistryPage: (next: Setter<RegistryPageInfo | null>) => void;
  setLoadingFamilyIds: (next: Setter<string[]>) => void;
  setDiskRuns: (next: Setter<string[]>) => void;
  setLoading: (next: Setter<boolean>) => void;
  setSimulating: (next: Setter<boolean>) => void;
  setRunsData: (next: Setter<Record<string, any>>) => void;
  setSelectedRunIds: (next: Setter<string[]>) => void;
  setSelectedSignals: (next: Setter<string[]>) => void;
  setSignalDisplayNames: (next: Setter<Record<string, string>>) => void;
  setAvailableNoiseProfiles: (next: Setter<string[]>) => void;
  setSelectedNoiseProfile: (next: Setter<string>) => void;
  setNoiseSeed: (next: Setter<number>) => void;
  setSelectedSignalCases: (next: Setter<string[][]>) => void;
  setAvailableSignals: (next: Setter<string[]>) => void;
  setExpandedInterventions: (next: Setter<string[]>) => void;
  setHistory: (next: Setter<ModelRecord[]>) => void;
  setHistoryIndex: (next: Setter<number>) => void;
  setSaveNotice: (next: Setter<string>) => void;
  setSimulationNotice: (next: Setter<DashboardState['simulationNotice']>) => void;
  setBulkTimeByRunId: (next: Setter<Record<string, string>>) => void;
  setSignalsFetchInFlight: (next: Setter<boolean>) => void;
  setSelectedRunRowIds: (next: Setter<string[]>) => void;
  setSelectedParamKeys: (next: Setter<string[]>) => void;
  setSamplingRate: (next: Setter<number | null>) => void;
  setSamplingRateDraft: (next: Setter<string>) => void;
  setSamplingRateDirty: (next: Setter<boolean>) => void;
  setSamplingRateError: (next: Setter<string>) => void;
  setSamplingRatePlotOverride: (next: Setter<number | null>) => void;
  setSamplingIntervals: (next: Setter<Record<string, SamplingSpec>>) => void;
  setSamplingPerturbationIntervals: (next: Setter<Record<string, SamplingSpec>>) => void;
  setSamplingDetails: (next: Setter<Record<string, SamplingDetail>>) => void;
  setExposedParameterKeys: (next: Setter<string[]>) => void;
  setExposedInitialStateKeys: (next: Setter<string[]>) => void;
};

export const useDashboardStore = create<DashboardState & DashboardActions>((set) => ({
  models: [],
  modelValidation: {},
  selectedModel: '',
  policies: [],
  selectedPolicy: '',
  modelRecord: null,
  originalModelRecord: null,
  registryPage: null,
  loadingFamilyIds: [],
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

  setModels: (next) => set((state) => ({ models: resolveSetter(next, state.models) })),
  setModelValidation: (next) =>
    set((state) => ({ modelValidation: resolveSetter(next, state.modelValidation) })),
  setSelectedModel: (next) =>
    set((state) => ({ selectedModel: resolveSetter(next, state.selectedModel) })),
  setPolicies: (next) =>
    set((state) => ({ policies: resolveSetter(next, state.policies) })),
  setSelectedPolicy: (next) =>
    set((state) => ({ selectedPolicy: resolveSetter(next, state.selectedPolicy) })),
  setModelRecord: (next) => set((state) => ({ modelRecord: resolveSetter(next, state.modelRecord) })),
  setOriginalModelRecord: (next) =>
    set((state) => ({ originalModelRecord: resolveSetter(next, state.originalModelRecord) })),
  setRegistryPage: (next) =>
    set((state) => ({ registryPage: resolveSetter(next, state.registryPage) })),
  setLoadingFamilyIds: (next) =>
    set((state) => ({ loadingFamilyIds: resolveSetter(next, state.loadingFamilyIds) })),
  setDiskRuns: (next) => set((state) => ({ diskRuns: resolveSetter(next, state.diskRuns) })),
  setLoading: (next) => set((state) => ({ loading: resolveSetter(next, state.loading) })),
  setSimulating: (next) => set((state) => ({ simulating: resolveSetter(next, state.simulating) })),
  setRunsData: (next) => set((state) => ({ runsData: resolveSetter(next, state.runsData) })),
  setSelectedRunIds: (next) =>
    set((state) => ({ selectedRunIds: resolveSetter(next, state.selectedRunIds) })),
  setSelectedSignals: (next) =>
    set((state) => ({ selectedSignals: resolveSetter(next, state.selectedSignals) })),
  setSignalDisplayNames: (next) =>
    set((state) => ({ signalDisplayNames: resolveSetter(next, state.signalDisplayNames) })),
  setAvailableNoiseProfiles: (next) =>
    set((state) => ({ availableNoiseProfiles: resolveSetter(next, state.availableNoiseProfiles) })),
  setSelectedNoiseProfile: (next) =>
    set((state) => ({ selectedNoiseProfile: resolveSetter(next, state.selectedNoiseProfile) })),
  setNoiseSeed: (next) =>
    set((state) => ({ noiseSeed: resolveSetter(next, state.noiseSeed) })),
  setSelectedSignalCases: (next) =>
    set((state) => ({ selectedSignalCases: resolveSetter(next, state.selectedSignalCases) })),
  setAvailableSignals: (next) =>
    set((state) => ({ availableSignals: resolveSetter(next, state.availableSignals) })),
  setExpandedInterventions: (next) =>
    set((state) => ({ expandedInterventions: resolveSetter(next, state.expandedInterventions) })),
  setHistory: (next) => set((state) => ({ history: resolveSetter(next, state.history) })),
  setHistoryIndex: (next) => set((state) => ({ historyIndex: resolveSetter(next, state.historyIndex) })),
  setSaveNotice: (next) => set((state) => ({ saveNotice: resolveSetter(next, state.saveNotice) })),
  setSimulationNotice: (next) =>
    set((state) => ({ simulationNotice: resolveSetter(next, state.simulationNotice) })),
  setBulkTimeByRunId: (next) =>
    set((state) => ({ bulkTimeByRunId: resolveSetter(next, state.bulkTimeByRunId) })),
  setSignalsFetchInFlight: (next) =>
    set((state) => ({ signalsFetchInFlight: resolveSetter(next, state.signalsFetchInFlight) })),
  setSelectedRunRowIds: (next) =>
    set((state) => ({ selectedRunRowIds: resolveSetter(next, state.selectedRunRowIds) })),
  setSelectedParamKeys: (next) =>
    set((state) => ({ selectedParamKeys: resolveSetter(next, state.selectedParamKeys) })),
  setSamplingRate: (next) =>
    set((state) => ({ samplingRate: resolveSetter(next, state.samplingRate) })),
  setSamplingRateDraft: (next) =>
    set((state) => ({ samplingRateDraft: resolveSetter(next, state.samplingRateDraft) })),
  setSamplingRateDirty: (next) =>
    set((state) => ({ samplingRateDirty: resolveSetter(next, state.samplingRateDirty) })),
  setSamplingRateError: (next) =>
    set((state) => ({ samplingRateError: resolveSetter(next, state.samplingRateError) })),
  setSamplingRatePlotOverride: (next) =>
    set((state) => ({ samplingRatePlotOverride: resolveSetter(next, state.samplingRatePlotOverride) })),
  setSamplingIntervals: (next) =>
    set((state) => ({ samplingIntervals: resolveSetter(next, state.samplingIntervals) })),
  setSamplingPerturbationIntervals: (next) =>
    set((state) => ({ samplingPerturbationIntervals: resolveSetter(next, state.samplingPerturbationIntervals) })),
  setSamplingDetails: (next) =>
    set((state) => ({ samplingDetails: resolveSetter(next, state.samplingDetails) })),
  setExposedParameterKeys: (next) =>
    set((state) => ({ exposedParameterKeys: resolveSetter(next, state.exposedParameterKeys) })),
  setExposedInitialStateKeys: (next) =>
    set((state) => ({ exposedInitialStateKeys: resolveSetter(next, state.exposedInitialStateKeys) })),
}));
