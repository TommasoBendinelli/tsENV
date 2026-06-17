'use client';

import React from 'react';
import { Check, Clock, Eye, Minus, X } from 'lucide-react';
import TimeSeriesPlot from '../../../TimeSeriesPlot';
import { cn } from '../../utils';
import {
  collectBaselineSubtreeRunIds,
  formatSignalLabel,
  formatNumericSignificantDigits,
  resolveNumericSignificantDigits,
} from '../../domains/runs';
import { useDashboardControllerContext } from '@/components/dashboard/DashboardControllerContext';
import { useDashboardStore } from '../../useDashboardStore';

const formatNoiseSeedDraft = (value: number) => String(Math.trunc(Number.isFinite(value) ? value : 0));

const normalizeStatus = (value: unknown): 'success' | 'failed' | 'not_run' => {
  const normalized = String(value || '').trim();
  if (normalized === 'success' || normalized === 'failed') return normalized;
  return 'not_run';
};

const resolveBaselineEligibility = (baseline: any): 'eligible' | 'not_eligible' | 'unknown' => {
  if (baseline?.eligible === true) return 'eligible';
  if (baseline?.eligible === false) return 'not_eligible';
  return 'unknown';
};

const StatusPill = (props: { status: 'success' | 'failed' | 'not_run' }) => (
  <span className={cn(
    "px-2 py-1 rounded-md text-[9px] font-black uppercase tracking-tight border",
    props.status === 'failed'
      ? "bg-red-100 text-red-700 border-red-200"
      : props.status === 'success'
        ? "bg-green-100 text-green-700 border-green-200"
        : "bg-orange-100 text-orange-700 border-orange-200"
  )}>
    {props.status === 'failed' ? 'Failed' : props.status === 'success' ? 'Success' : 'Not Run'}
  </span>
);

const EligibilityPill = (props: { eligibility: 'eligible' | 'not_eligible' | 'unknown' }) => (
  <span className={cn(
    "inline-flex h-7 w-7 items-center justify-center rounded-md border",
    props.eligibility === 'eligible'
      ? "bg-green-100 text-green-700 border-green-200"
      : props.eligibility === 'not_eligible'
        ? "bg-red-100 text-red-700 border-red-200"
        : "bg-gray-100 text-gray-500 border-gray-200"
  )}
    title={
      props.eligibility === 'eligible'
        ? 'Eligible'
        : props.eligibility === 'not_eligible'
          ? 'Not Eligible'
          : 'Eligibility Unknown'
    }
    aria-label={
      props.eligibility === 'eligible'
        ? 'Eligible'
        : props.eligibility === 'not_eligible'
          ? 'Not Eligible'
          : 'Eligibility Unknown'
    }
  >
    {props.eligibility === 'eligible'
      ? <Check size={16} strokeWidth={3} aria-hidden="true" />
      : props.eligibility === 'not_eligible'
        ? <X size={16} strokeWidth={3} aria-hidden="true" />
        : <Minus size={16} strokeWidth={3} aria-hidden="true" />}
  </span>
);

const ChildFailureMarker = () => (
  <span
    className="inline-flex h-6 w-6 items-center justify-center rounded-md border border-red-200 bg-red-100 text-red-700"
    title="Child not detectable over baseline or time0 baseline"
    aria-label="Child not detectable over baseline or time0 baseline"
  >
    <X size={14} strokeWidth={3} aria-hidden="true" />
  </span>
);

type SignalSnrEntry = {
  runId: string;
  signal: string;
  value: string;
};

const SignalSnrRow = (props: { entries: SignalSnrEntry[] }) => {
  if (props.entries.length === 0) return null;

  const label = 'SNR';

  return (
    <div className="flex min-w-0 items-start gap-2" aria-label={`${label} by signal`}>
      <span
        className={cn(
          "mt-1 w-16 shrink-0 text-[10px] font-black uppercase tracking-wider",
          "text-emerald-600"
        )}
      >
        {label}
      </span>
      <div className="flex min-w-0 flex-wrap items-center gap-1">
        {props.entries.map((entry, idx) => (
          <span
            key={`${entry.runId}-${entry.signal}-${idx}`}
            className={cn(
              "inline-flex max-w-[18rem] items-center gap-1 truncate rounded-md border px-2 py-1 text-[10px] font-semibold",
              "border-emerald-200 bg-emerald-50 text-emerald-700"
            )}
            title={`${entry.runId} ${entry.signal} ${label}: ${entry.value}`}
          >
            <span className="truncate">{entry.signal}</span>
            <span className="font-mono">{entry.value}</span>
          </span>
        ))}
      </div>
    </div>
  );
};

const formatSnrDbValue = (value: unknown): string | null => {
  if (value === null || value === undefined) return null;
  const rawText = String(value).trim().toLowerCase();
  if (rawText === '-inf' || rawText === '-infinity') return '-inf dB';
  const numeric = typeof value === 'number' ? value : Number(String(value).trim());
  if (!Number.isFinite(numeric)) return null;
  return `${formatNumericSignificantDigits(numeric, 4)} dB`;
};

const collectSignalSnrEntries = (params: {
  runId: string;
  source: unknown;
  availableSignals: string[];
  signalDisplayNames: Record<string, string>;
}): SignalSnrEntry[] => {
  const payload = (params.source as any)?.signal_snr
    ?? (params.source as any)?.signal_analysis
    ?? (params.source as any)?.signal_analayis;
  if (!payload || typeof payload !== 'object') return [];
  const availableSet = new Set(params.availableSignals);
  const entries: SignalSnrEntry[] = [];
  const pushEntry = (signal: unknown, value: unknown) => {
    const signalKey = String(signal ?? '').trim();
    if (!signalKey || !availableSet.has(signalKey)) return;
    const formatted = formatSnrDbValue(value);
    if (!formatted) return;
    entries.push({
      runId: params.runId,
      signal: formatSignalLabel(signalKey, params.signalDisplayNames),
      value: formatted,
    });
  };

  if (Array.isArray(payload)) {
    for (const item of payload) {
      if (!item || typeof item !== 'object') continue;
      const signal = (item as any).signal ?? (item as any).channel ?? (item as any).name ?? (item as any).key;
      pushEntry(signal, (item as any).value ?? (item as any).snr ?? (item as any).global);
    }
    return entries;
  }

  const record = payload as Record<string, any>;
  const scopedGlobal = record.global;
  if (scopedGlobal && typeof scopedGlobal === 'object' && !Array.isArray(scopedGlobal)) {
    for (const [signal, value] of Object.entries(scopedGlobal)) {
      pushEntry(signal, value);
    }
  }

  for (const [signal, analysis] of Object.entries(record)) {
    if (signal === 'local' || signal === 'global') continue;
    if (!analysis || typeof analysis !== 'object' || Array.isArray(analysis)) {
      pushEntry(signal, analysis);
      continue;
    }
    pushEntry(signal, (analysis as any).value ?? (analysis as any).snr ?? (analysis as any).global);
  }

  return entries;
};

export function DashboardMainRuns() {
  const selectedModel = useDashboardStore((state) => state.selectedModel);
  const modelRecord = useDashboardStore((state) => state.modelRecord);
  const registry = modelRecord?.baselines ?? [];
  const registryPage = useDashboardStore((state) => state.registryPage);
  const loadingFamilyIds = useDashboardStore((state) => state.loadingFamilyIds);
  const runsData = useDashboardStore((state) => state.runsData);
  const selectedRunIds = useDashboardStore((state) => state.selectedRunIds);
  const selectedSignals = useDashboardStore((state) => state.selectedSignals);
  const signalDisplayNames = useDashboardStore((state) => state.signalDisplayNames);
  const availableSignals = useDashboardStore((state) => state.availableSignals);
  const availableNoiseProfiles = useDashboardStore((state) => state.availableNoiseProfiles);
  const selectedNoiseProfile = useDashboardStore((state) => state.selectedNoiseProfile);
  const noiseSeed = useDashboardStore((state) => state.noiseSeed);
  const setSelectedRunIds = useDashboardStore((state) => state.setSelectedRunIds);
  const setSelectedNoiseProfile = useDashboardStore((state) => state.setSelectedNoiseProfile);
  const setNoiseSeed = useDashboardStore((state) => state.setNoiseSeed);
  const expandedInterventions = useDashboardStore((state) => state.expandedInterventions);
  const setExpandedInterventions = useDashboardStore((state) => state.setExpandedInterventions);
  const samplingRate = useDashboardStore((state) => state.samplingRate);
  const exposedParameterKeys = useDashboardStore((state) => state.exposedParameterKeys);
  const exposedInitialStateKeys = useDashboardStore((state) => state.exposedInitialStateKeys);
  const numericSignificantDigits = React.useMemo(
    () => resolveNumericSignificantDigits((modelRecord as any)?.metadata?.ui?.numeric_significant_digits),
    [modelRecord],
  );

  const {
    handleVisibleSignalsChange,
    handleRegistryPageChange,
    ensureFamilyDetailsLoaded,
    hasRunData,
    toggleRunVisualization,
    toggleRunVisualizationGroup,
    paramKeys,
    shouldIgnoreRowClick,
  } = useDashboardControllerContext();

  const [noiseSeedDraft, setNoiseSeedDraft] = React.useState<string>(() => formatNoiseSeedDraft(noiseSeed));

  React.useEffect(() => {
    setNoiseSeedDraft(formatNoiseSeedDraft(noiseSeed));
  }, [noiseSeed]);

  const selectedRunIdsInRegistry = React.useMemo(() => {
    const knownIds = new Set<string>();
    for (const base of registry ?? []) {
      const baseId = String((base as any)?.run_id || '').trim();
      if (baseId) knownIds.add(baseId);
      for (const iv of (base as any)?.interventions ?? []) {
        const ivId = String((iv as any)?.name || '').trim();
        if (ivId) knownIds.add(ivId);
        const time0 = String((iv as any)?.time0_baseline_uuid || '').trim();
        if (time0) knownIds.add(time0);
      }
    }
    return selectedRunIds.filter((id) => knownIds.has(String(id || '').trim()));
  }, [registry, selectedRunIds]);
  const selectedRunIdSet = React.useMemo(
    () => new Set(selectedRunIdsInRegistry.map((id) => String(id || '').trim()).filter(Boolean)),
    [selectedRunIdsInRegistry],
  );
  const selectedSignalSnrEntries = React.useMemo(() => {
    if (selectedNoiseProfile === 'none' || selectedRunIdsInRegistry.length !== 2) return [];
    const runId = selectedRunIdsInRegistry[0];
    return collectSignalSnrEntries({
      runId,
      source: runsData[runId],
      availableSignals,
      signalDisplayNames,
    });
  }, [availableSignals, runsData, selectedNoiseProfile, selectedRunIdsInRegistry, signalDisplayNames]);
  const collapseBaselineSubtree = React.useCallback((baseline: any) => {
    const baselineId = String(baseline?.run_id || '').trim();
    if (!baselineId) return;
    const childIds = new Set<string>();
    for (const iv of baseline?.interventions ?? []) {
      const childId = String(iv?.name || '').trim();
      const time0BaselineRunId = String(iv?.time0_baseline_uuid || '').trim();
      if (childId) childIds.add(childId);
      if (time0BaselineRunId) childIds.add(time0BaselineRunId);
    }
    setExpandedInterventions((prev) => prev.filter((id) => id !== baselineId));
    if (childIds.size > 0) {
      setSelectedRunIds((prev) => prev.filter((id) => !childIds.has(String(id || '').trim())));
    }
  }, [setExpandedInterventions, setSelectedRunIds]);

  const toggleBaselineExpansion = React.useCallback(async (baseline: any) => {
    const baselineId = String(baseline?.run_id || '').trim();
    if (!baselineId) return;
    if (expandedInterventions.includes(baselineId)) {
      collapseBaselineSubtree(baseline);
      return;
    }
    await ensureFamilyDetailsLoaded(baseline);
    setExpandedInterventions((prev) => (
      prev.includes(baselineId) ? prev : [...prev, baselineId]
    ));
  }, [collapseBaselineSubtree, ensureFamilyDetailsLoaded, expandedInterventions, setExpandedInterventions]);

  const commitNoiseSeedValue = React.useCallback((value: string) => {
    const rawValue = value.trim();
    const parsed = Number(rawValue);
    const nextSeed = rawValue && Number.isFinite(parsed) ? Math.max(0, Math.trunc(parsed)) : 0;
    setNoiseSeed(nextSeed);
    return nextSeed;
  }, [setNoiseSeed]);

  const handleNoiseSeedChange = React.useCallback((value: string) => {
    setNoiseSeedDraft(value);
    commitNoiseSeedValue(value);
  }, [commitNoiseSeedValue]);

  const normalizeNoiseSeedDraft = React.useCallback(() => {
    const nextSeed = commitNoiseSeedValue(noiseSeedDraft);
    setNoiseSeedDraft(formatNoiseSeedDraft(nextSeed));
  }, [commitNoiseSeedValue, noiseSeedDraft]);

  const hasExplicitExposedGroups = exposedParameterKeys.length > 0 || exposedInitialStateKeys.length > 0;
  const parameterColumnKeys = React.useMemo(
    () => (hasExplicitExposedGroups ? exposedParameterKeys : paramKeys),
    [exposedParameterKeys, hasExplicitExposedGroups, paramKeys],
  );
  const initialStateColumnKeys = React.useMemo(() => {
    if (!hasExplicitExposedGroups) return [];
    const parameterKeySet = new Set(parameterColumnKeys);
    return exposedInitialStateKeys.filter((key) => !parameterKeySet.has(key));
  }, [exposedInitialStateKeys, hasExplicitExposedGroups, parameterColumnKeys]);
  const totalVariableColumnCount = parameterColumnKeys.length + initialStateColumnKeys.length;

  return (
    <>
      <section className="space-y-3">
        <div className="bg-white rounded-xl border shadow-sm overflow-hidden">
          {registryPage?.mode === 'page' && (
            <div className="flex flex-wrap items-center justify-between gap-3 border-b bg-gray-50 px-4 py-2 text-xs text-gray-500">
              <span className="font-medium">
                Families {registryPage.total_families === 0 ? 0 : ((registryPage.page - 1) * registryPage.page_size) + 1}
                {'-'}
                {Math.min(registryPage.page * registryPage.page_size, registryPage.total_families)}
                {' '}of {registryPage.total_families}
              </span>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  className="rounded border bg-white px-2 py-1 font-semibold text-gray-600 disabled:opacity-40"
                  disabled={!registryPage.has_previous}
                  onClick={() => void handleRegistryPageChange(registryPage.page - 1)}
                >
                  Previous
                </button>
                <span className="font-mono text-[11px]">
                  Page {registryPage.page} / {registryPage.total_pages}
                </span>
                <button
                  type="button"
                  className="rounded border bg-white px-2 py-1 font-semibold text-gray-600 disabled:opacity-40"
                  disabled={!registryPage.has_next}
                  onClick={() => void handleRegistryPageChange(registryPage.page + 1)}
                >
                  Next
                </button>
              </div>
            </div>
          )}
          <div className="overflow-auto max-h-[calc(100vh-520px)]">
            <table className="w-full text-sm text-left">
              <thead className="bg-gray-50 text-gray-400 uppercase text-[10px] font-bold tracking-wider border-b sticky top-0 z-10">
                <tr>
                  <th className="px-4 py-3 w-24 text-center">Status</th>
                  <th className="px-4 py-3 w-28 text-center">Eligible</th>
                  <th className="px-4 py-3 w-56 min-w-[220px]">Run ID</th>
                  <th className="px-4 py-3 w-56 min-w-[220px]">Family ID</th>
                  <th className="px-4 py-3 w-32">Source</th>
                  <th className="px-4 py-3 w-40">Recipe Hash</th>
                  {parameterColumnKeys.map((k) => (
                    <th key={k} className="px-4 py-3 font-mono text-gray-500">
                      <span className="text-left">{k}</span>
                    </th>
                  ))}
                  {initialStateColumnKeys.map((k) => (
                    <th key={k} className="px-4 py-3 font-mono text-gray-500">
                      <span className="text-left">{k}</span>
                    </th>
                  ))}
                  <th className="px-4 py-3 text-right w-24">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {registry.map((r, i) => {
                  const childRunIds = r.interventions.map(iv => iv.name);
                  const plottableSubtreeIds = collectBaselineSubtreeRunIds(r).filter((id) => hasRunData(id));
                  const childStatuses = r.interventions.map((iv) => normalizeStatus((iv as any)?.status));
                  const explicitAggregateStatus = normalizeStatus((r as any)?.aggregate_status);
                  const hasFailure = normalizeStatus((r as any)?.status) === 'failed' || childStatuses.some((status) => status === 'failed');
                  const hasSuccess = normalizeStatus((r as any)?.status) === 'success' || childStatuses.some((status) => status === 'success');
                  const aggregateStatus = (r as any)?.aggregate_status ? explicitAggregateStatus : hasFailure ? 'failed' : hasSuccess ? 'success' : 'not_run';
                  const isSelected = selectedRunIdSet.has(String(r.run_id || '').trim())
                    || plottableSubtreeIds.some((id) => selectedRunIdSet.has(String(id || '').trim()));
                  const isExpanded = expandedInterventions.includes(r.run_id);
                  const childCount = Number.isFinite(Number((r as any)?.intervention_count))
                    ? Math.max(0, Math.trunc(Number((r as any).intervention_count)))
                    : new Set(childRunIds).size;
                  const eligibility = resolveBaselineEligibility(r);
                  const familyId = String((r as any)?.family_id || '').trim();
                  const isFamilyLoading = [familyId, String(r.run_id || '').trim()].some((id) => loadingFamilyIds.includes(id));

                  return (
                    <React.Fragment key={`${r.timestamp}-${i}`}>
                      <tr
                        className={cn(
                          "hover:bg-blue-50/20 transition-colors group cursor-pointer",
                          isSelected && "bg-blue-50/50"
                        )}
                        onClick={(event) => {
                          if (shouldIgnoreRowClick(event.target)) return;
                          void toggleBaselineExpansion(r);
                        }}
                      >
                        <td className="px-4 py-2 text-center">
                          <div className="flex flex-col items-center gap-1">
                            <StatusPill status={aggregateStatus} />
                            {r.interventions.length > 0 && (
                              <span className="text-[9px] text-gray-400 font-mono">
                                {childCount} {childCount === 1 ? 'child' : 'children'}
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-2 text-center">
                          <EligibilityPill eligibility={eligibility} />
                        </td>
                        <td
                          className="px-4 py-2 font-mono text-xs text-gray-500 font-medium w-56 min-w-[220px] cursor-text"
                          onClick={(event) => {
                            event.stopPropagation();
                          }}
                        >
                          <span title={r.run_id} className="truncate block w-full select-text cursor-text">
                            {r.run_id}
                          </span>
                        </td>
                        <td className="px-4 py-2 font-mono text-xs text-gray-500 cursor-text">
                          <span title={(r as any).family_id || ''} className="truncate block w-56 select-text">
                            {(r as any).family_id || '—'}
                          </span>
                        </td>
                        <td className="px-4 py-2 text-xs text-gray-500">
                          <div className="flex flex-col gap-0.5">
                            <span>{(r as any).source || '—'}</span>
                            {(r as any).validation_profile && (
                              <span className="font-mono text-[10px] text-gray-400">{(r as any).validation_profile}</span>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-2 font-mono text-xs text-gray-500 cursor-text">
                          <span title={(r as any).recipe_hash || ''} className="truncate block w-36 select-text">
                            {(r as any).recipe_hash || '—'}
                          </span>
                        </td>
                        {parameterColumnKeys.map(k => (
                          <td key={k} className="px-4 py-2 font-medium text-gray-700">
                            {typeof r.parameters[k] === 'number'
                              ? formatNumericSignificantDigits(r.parameters[k], numericSignificantDigits)
                              : (r.parameters[k] ?? '—')}
                          </td>
                        ))}
                        {initialStateColumnKeys.map(k => (
                          <td key={k} className="px-4 py-2 font-medium text-gray-700">
                            {typeof r.parameters[k] === 'number'
                              ? formatNumericSignificantDigits(r.parameters[k], numericSignificantDigits)
                              : (r.parameters[k] ?? '—')}
                          </td>
                        ))}
                        <td className="px-4 py-2 text-right">
                          <div className="flex items-center justify-end gap-1">
                            <button
                              onClick={(event) => {
                                event.stopPropagation();
                                void toggleRunVisualizationGroup(r);
                              }}
                              disabled={plottableSubtreeIds.length === 0}
                              className={cn(
                                "p-1.5 rounded-lg transition-all",
                                isSelected ? "text-blue-600 bg-blue-50" : "text-gray-400 hover:bg-gray-100",
                                plottableSubtreeIds.length === 0 && "opacity-30"
                              )}
                              title="Toggle baseline, child, and time0 plots"
                            >
                              <Eye size={14} />
                            </button>
                            <button
                              onClick={(event) => {
                                event.stopPropagation();
                                void toggleBaselineExpansion(r);
                              }}
                              className={cn("p-1.5 rounded-lg transition-all", isExpanded ? "text-blue-600 bg-blue-50" : "text-gray-400 hover:bg-gray-100")}
                              title="Show interventions"
                            >
                              <Clock size={14} />
                            </button>
                          </div>
                        </td>
                      </tr>
                      {isExpanded && (
                        <tr>
                          <td colSpan={7 + totalVariableColumnCount} className="px-12 py-3 bg-gray-50/50">
                            <div className="space-y-3">
                              <div className="flex items-center gap-2 text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2 border-b pb-1">
                                <Clock size={12} /> Simulation Steps & Interventions
                              </div>
                              <div className="flex items-center gap-4 bg-white/50 border border-dashed p-2 rounded-lg group/step">
                                <button
                                  onClick={() => hasRunData(r.run_id) && void toggleRunVisualization(r.run_id)}
                                  disabled={!hasRunData(r.run_id)}
                                  className={cn(
                                    "p-1.5 rounded transition-all",
                                    selectedRunIdSet.has(String(r.run_id || '').trim())
                                      ? "bg-blue-100 text-blue-600"
                                      : "text-gray-400 hover:text-blue-500",
                                    !hasRunData(r.run_id) && "opacity-20"
                                  )}
                                  title="Toggle baseline plot"
                                >
                                  <Eye size={14} />
                                </button>
                                <span className="text-[10px] font-black text-gray-400 uppercase w-20">Baseline</span>
                                <span className="font-mono text-xs text-gray-500 truncate">{r.run_id}</span>
                                <StatusPill status={normalizeStatus((r as any)?.status)} />
                              </div>
                              {isFamilyLoading && (
                                <p className="text-xs text-gray-400 italic py-2">Loading interventions...</p>
                              )}
                              {!isFamilyLoading && r.interventions.map((iv) => {
                                const time0BaselineRunId = String((iv as any)?.time0_baseline_uuid || '').trim();
                                const childNotDetectable = (iv as any)?.detectability_failed === true;
                                return (
                                  <div key={iv.name} className="flex items-center gap-4 bg-white p-2 rounded-lg border">
                                    <button
                                      onClick={() => hasRunData(iv.name) && void toggleRunVisualization(iv.name)}
                                      disabled={!hasRunData(iv.name)}
                                      className={cn(
                                        "p-1.5 rounded transition-all",
                                        selectedRunIdSet.has(String(iv.name || '').trim())
                                          ? "bg-blue-100 text-blue-600"
                                          : "text-gray-400 hover:text-blue-500",
                                        !hasRunData(iv.name) && "opacity-20"
                                      )}
                                      title="Toggle intervention plot"
                                    >
                                      <Eye size={14} />
                                    </button>
                                    <span className="text-[10px] font-black text-gray-400 uppercase w-20">Child</span>
                                    <span className="font-mono text-xs text-gray-500 truncate min-w-[12rem]">{iv.name}</span>
                                    <span className="text-xs text-gray-500">
                                      {String(iv.variable)} = {String(iv.value)}
                                    </span>
                                    {(iv as any).direction && (
                                      <span className="text-xs text-gray-400">
                                        {(iv as any).direction}
                                      </span>
                                    )}
                                    <span className="text-xs text-gray-400">
                                      t={formatNumericSignificantDigits(iv.intervention_time, numericSignificantDigits)}
                                    </span>
                                    {(iv as any).surrogate_filter_pass !== undefined && (
                                      <span
                                        className={cn(
                                          "rounded-md border px-2 py-1 text-[10px] font-semibold",
                                          (iv as any).surrogate_filter_pass
                                            ? "border-green-200 bg-green-50 text-green-700"
                                            : "border-red-200 bg-red-50 text-red-700"
                                        )}
                                        title={`surrogate_id=${(iv as any).surrogate_id || ''}; confidence=${(iv as any).true_label_confidence ?? ''}; margin=${(iv as any).confidence_margin ?? ''}`}
                                      >
                                        surrogate
                                      </span>
                                    )}
                                    {time0BaselineRunId && (
                                      <button
                                        onClick={() => hasRunData(time0BaselineRunId) && void toggleRunVisualization(time0BaselineRunId)}
                                        disabled={!hasRunData(time0BaselineRunId)}
                                        className={cn(
                                          "px-2 py-1 rounded-md border text-[10px] font-semibold",
                                          selectedRunIdSet.has(time0BaselineRunId)
                                            ? "border-blue-200 bg-blue-50 text-blue-600"
                                            : "border-gray-200 bg-gray-50 text-gray-400",
                                          !hasRunData(time0BaselineRunId) && "opacity-40"
                                        )}
                                        title="Toggle time-zero baseline plot"
                                      >
                                        time0
                                      </button>
                                    )}
                                    <StatusPill status={normalizeStatus((iv as any)?.status)} />
                                    {childNotDetectable && <ChildFailureMarker />}
                                  </div>
                                );
                              })}
                              {!isFamilyLoading && r.interventions.length === 0 && (
                                <p className="text-xs text-gray-400 italic py-2">No timed interventions defined for this run.</p>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="bg-white p-4 rounded-xl border shadow-sm relative flex h-[420px] max-h-[55vh] flex-col sticky bottom-6 z-10">
        <div className="mb-2 flex shrink-0 flex-wrap items-start justify-between gap-3">
          <div className="mr-1 flex min-w-0 flex-1 flex-col gap-1">
            <span className="text-[11px] font-black uppercase tracking-wider text-gray-500">Plot</span>
            {selectedSignalSnrEntries.length > 0 && (
              <div className="flex min-w-0 flex-col gap-1">
                <SignalSnrRow entries={selectedSignalSnrEntries} />
              </div>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <label className="flex items-center gap-1 text-[10px] font-semibold text-gray-400 normal-case whitespace-nowrap leading-none">
              <span>Noise</span>
              <select
                className="w-24 h-7 rounded border bg-white px-2 py-1 text-[11px] font-medium text-gray-700"
                value={selectedNoiseProfile}
                onChange={(event) => setSelectedNoiseProfile(String(event.target.value || 'none').trim().toLowerCase() || 'none')}
                aria-label="Plot noise profile"
                title="Realistic model-backed noise profile applied to plotted data."
              >
                {availableNoiseProfiles.map((profile) => (
                  <option key={profile} value={profile}>
                    {profile === 'none' ? 'None' : profile.charAt(0).toUpperCase() + profile.slice(1)}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex items-center gap-1 text-[10px] font-semibold text-gray-400 normal-case whitespace-nowrap leading-none">
              <span>Seed</span>
              <input
                type="number"
                step="1"
                min={0}
                inputMode="numeric"
                className="w-20 h-7 rounded border bg-white px-2 py-1 text-[11px] font-mono text-gray-700 text-right"
                value={noiseSeedDraft}
                onChange={(event) => handleNoiseSeedChange(event.target.value)}
                onBlur={normalizeNoiseSeedDraft}
                aria-label="Plot noise seed"
                title="Seed passed to the model noise adder for deterministic plotted noise."
              />
            </label>
          </div>
        </div>
        <div className="flex min-h-0 flex-1 flex-col gap-3">
          <div className="w-full min-h-0 flex-1">
            <TimeSeriesPlot
              allRunsData={runsData}
              selectedRunIds={selectedRunIdsInRegistry}
              availableSignals={availableSignals}
              selectedSignals={selectedSignals}
              samplingRate={samplingRate}
              modelId={selectedModel}
              signalDisplayNames={signalDisplayNames}
              formatSignalLabel={formatSignalLabel}
              onVisibleSignalsChange={handleVisibleSignalsChange}
            />
          </div>
        </div>
      </section>
    </>
  );
}
