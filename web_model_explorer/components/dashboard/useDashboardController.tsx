'use client';

import React, { useEffect, useMemo, useRef } from 'react';
import axios from 'axios';
import { buildInterventionDisplayNames } from './controller/interventionNaming';
import { normalizeRegistry } from './controller/registryNormalization';
import {
  clearDeepLinkRunFromLocation,
  collectDeepLinkKnownRunIds,
  parseDeepLinkFromSearch,
  shouldExpandBaselineForDeepLink,
} from './controller/deepLink';
import { sanitizeRegistryBoundUiState } from './controller/registryStateSanitizer';
import { useDashboardStore } from './useDashboardStore';
import {
  collectBaselineSubtreeRunIds,
  formatTimeLabel,
  formatVariableLabel,
  shouldIgnoreRowClick,
} from './domains/runs';
import {
  buildSamplingDetails,
  buildSamplingIntervals,
  extractExposedVariableGroups,
  isValidNumberInput,
} from './domains/sampling';
import type {
  RunRecord,
  SimulationStatus,
} from './types';

export type DashboardController = ReturnType<typeof useDashboardController>;

const asTrimmedString = (value: unknown): string => String(value ?? '').trim();

function responseMessageFromData(data: unknown): string {
  if (!data || typeof data !== 'object') return '';
  const record = data as Record<string, unknown>;
  for (const key of ['error', 'message'] as const) {
    const value = asTrimmedString(record[key]);
    if (value) return value;
  }
  return '';
}

function formatApiError(err: any, fallback: string): string {
  const responseMessage = responseMessageFromData(err?.response?.data);
  if (responseMessage) return responseMessage;

  const status = err?.response?.status;
  const statusText = asTrimmedString(err?.response?.statusText);
  if (status) {
    return statusText
      ? `${fallback} (HTTP ${status}: ${statusText})`
      : `${fallback} (HTTP ${status}).`;
  }

  const requestUrl = asTrimmedString(err?.config?.url);
  if (err?.request && !err?.response) {
    const endpoint = requestUrl ? ` '${requestUrl}'` : '';
    return `${fallback}: request${endpoint} failed before the server returned a response. Check that the Model Explorer is running on the documented host and port.`;
  }

  const message = asTrimmedString(err?.message);
  return message || fallback;
}

export function useDashboardController() {
  const deepLinkRef = useRef<{
    rawRunId: string | null;
    rawComparatorRunId: string | null;
    rawModelId: string | null;
    compareMode: 'auto' | 'baseline' | 'time0' | 'sibling' | 'none';
    noiseProfile: 'none' | 'low' | 'high' | null;
    noiseSeed: number | null;
    resolved: boolean;
    applied: boolean;
  }>({
    rawRunId: null,
    rawComparatorRunId: null,
    rawModelId: null,
    compareMode: 'auto',
    noiseProfile: null,
    noiseSeed: null,
    resolved: false,
    applied: false,
  });
  const noiseRefreshAttemptedKeyRef = useRef<string>('');
  const noiseRefreshInFlightKeysRef = useRef<Set<string>>(new Set());
  const prevAvailableSignalsRef = useRef<string[]>([]);
  const modelLoadRequestIdRef = useRef(0);
  const registryModelKeyRef = useRef<string>('');

  const {
    selectedModel,
    modelRecord,
    diskRuns,
    loading,
    runsData,
    selectedRunIds,
    selectedSignals,
    signalDisplayNames,
    availableNoiseProfiles,
    selectedNoiseProfile,
    noiseSeed,
    availableSignals,
    expandedInterventions,
    bulkTimeByRunId,
    samplingRate,
    setModels,
    setModelValidation,
    setSelectedModel,
    setModelRecord,
    setOriginalModelRecord,
    setDiskRuns,
    setLoading,
    setRunsData,
    setSelectedRunIds,
    setSelectedSignals,
    setSignalDisplayNames,
    setAvailableNoiseProfiles,
    setSelectedNoiseProfile,
    setNoiseSeed,
    setSelectedSignalCases,
    setAvailableSignals,
    setExpandedInterventions,
    setBulkTimeByRunId,
    setHistory,
    setHistoryIndex,
    setSamplingRate,
    setSamplingRateDraft,
    setSamplingRateDirty,
    setSamplingRateError,
    setSamplingRatePlotOverride,
    setSamplingIntervals,
    setSamplingPerturbationIntervals,
    setSamplingDetails,
    setExposedParameterKeys,
    setExposedInitialStateKeys,
  } = useDashboardStore();

  const registry: RunRecord[] = modelRecord?.baselines ?? [];

  const currentTimeSeriesSamplingRate = useMemo(() => {
    const median = (arr: number[]) => {
      if (arr.length === 0) return null;
      const sorted = [...arr].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      return sorted.length % 2 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
    };
    const collectDeltas = (rows: any[], timeIdx: number) => {
      const deltas: number[] = [];
      let prev: number | null = null;
      for (const row of rows) {
        const t = Number(row?.[timeIdx]);
        if (!Number.isFinite(t)) continue;
        if (prev !== null) {
          const dt = t - prev;
          if (Number.isFinite(dt) && dt > 0) deltas.push(dt);
        }
        prev = t;
        if (deltas.length >= 200) break;
      }
      return deltas;
    };

    const candidateIds = selectedRunIds.length > 0 ? selectedRunIds : Object.keys(runsData);
    for (const runId of candidateIds) {
      const data = runsData[runId];
      if (!data) continue;
      const timeIdx = Array.isArray(data.columns) ? data.columns.indexOf('time') : -1;
      const rows: any[] = Array.isArray(data.data) ? data.data : [];
      if (timeIdx < 0 || rows.length < 3) continue;
      const dt = median(collectDeltas(rows, timeIdx));
      if (typeof dt === 'number' && Number.isFinite(dt) && dt > 0) return 1 / dt;
    }
    return null;
  }, [selectedRunIds.join('|'), runsData]);

  useEffect(() => {
    axios.get('/api/models')
      .then(res => {
        const sorted = [...(res.data.models || [])].sort();
        setModels(sorted);
        const validation = res.data?.validation && typeof res.data.validation === 'object'
          ? res.data.validation
          : {};
        setModelValidation(validation);
      })
      .catch((err) => {
        console.error(err);
        setModels([]);
        setModelValidation({});
        alert(`Failed to load model list: ${formatApiError(err, 'Failed to load model list.')}`);
      });
  }, [setModelValidation, setModels]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const {
      rawRunId,
      rawComparatorRunId,
      rawModelId,
      compareMode,
      noiseProfile,
      noiseSeed,
    } = parseDeepLinkFromSearch(window.location.search);

    if (!rawRunId) return;
    deepLinkRef.current = {
      rawRunId,
      rawComparatorRunId,
      rawModelId,
      compareMode,
      noiseProfile,
      noiseSeed,
      resolved: false,
      applied: false,
    };
  }, []);

  useEffect(() => {
    if (!selectedModel) return;
    const requestId = modelLoadRequestIdRef.current + 1;
    modelLoadRequestIdRef.current = requestId;
    registryModelKeyRef.current = '';
    setModelRecord(null);
    setOriginalModelRecord(null);
    setDiskRuns([]);
    setHistory([]);
    setHistoryIndex(0);
    setRunsData({});
    setSelectedRunIds([]);
    setExpandedInterventions([]);
    setAvailableSignals([]);
    setSelectedSignals([]);
    setSignalDisplayNames({});
    setAvailableNoiseProfiles(['none']);
    setSelectedNoiseProfile('none');
    setSelectedSignalCases([]);
    setSamplingIntervals({});
    setSamplingPerturbationIntervals({});
    setSamplingDetails({});
    setExposedParameterKeys([]);
    setExposedInitialStateKeys([]);
    void fetchRegistry(selectedModel, requestId);
    void fetchDistribution(selectedModel, requestId);
  }, [selectedModel]);

  useEffect(() => {
    const dl = deepLinkRef.current;
    if (dl.applied || dl.resolved || !dl.rawRunId) return;

    if (dl.rawModelId) {
      setSelectedModel(dl.rawModelId);
      dl.resolved = true;
      return;
    }

    void (async () => {
      try {
        const res = await axios.get(`/api/locate-run?run=${encodeURIComponent(dl.rawRunId!)}`);
        const model = String(res.data?.model || '').trim();
        if (!model) {
          alert(`Could not locate run '${dl.rawRunId}'.`);
          clearDeepLinkRunFromLocation();
          dl.applied = true;
          return;
        }
        setSelectedModel(model);
      } catch (err) {
        alert(`Failed to locate run '${dl.rawRunId}': ${formatApiError(err, 'Failed to locate run.')}`);
        clearDeepLinkRunFromLocation();
        dl.applied = true;
      } finally {
        dl.resolved = true;
      }
    })();
  }, [setSelectedModel]);

  const findBaselineForRunId = (runId: string): string | null => {
    const target = String(runId || '').trim();
    if (!target) return null;
    for (const baseline of modelRecord?.baselines ?? []) {
      const baselineId = String((baseline as any)?.run_id || '').trim();
      if (baselineId === target) return baselineId;
      for (const iv of (baseline as any)?.interventions ?? []) {
        const ivId = String((iv as any)?.name || '').trim();
        const time0Id = String((iv as any)?.time0_baseline_uuid || '').trim();
        if (ivId === target || time0Id === target) return baselineId;
      }
    }
    return null;
  };

  const pickComparator = (
    focusRunId: string,
    compareMode: 'auto' | 'baseline' | 'time0' | 'sibling' | 'none'
  ): { baselineId: string | null; comparatorId: string | null } => {
    if (compareMode === 'none') return { baselineId: null, comparatorId: null };
    const baselineId = findBaselineForRunId(focusRunId);
    if (!baselineId) return { baselineId: null, comparatorId: null };
    const run = (modelRecord?.baselines ?? []).find((r) => String(r.run_id) === baselineId);
    const iv = run?.interventions?.find((x: any) => String(x?.name) === focusRunId);
    const time0 = String((iv as any)?.time0_baseline_uuid || '').trim() || null;

    if (compareMode === 'time0') return { baselineId, comparatorId: time0 };
    if (compareMode === 'sibling') {
      const sibling = run?.interventions?.find((x: any) => String(x?.name) !== focusRunId)?.name ?? null;
      return { baselineId, comparatorId: sibling };
    }
    return { baselineId, comparatorId: baselineId };
  };

  useEffect(() => {
    const dl = deepLinkRef.current;
    if (dl.applied || !dl.rawRunId || !dl.resolved || !selectedModel) return;
    if (registryModelKeyRef.current !== selectedModel || !modelRecord) return;

    void (async () => {
      try {
        const knownRunIds = collectDeepLinkKnownRunIds(modelRecord?.baselines as any);
        if (!knownRunIds.has(dl.rawRunId!)) {
          alert(`Run '${dl.rawRunId}' is not present in the current registry for model '${selectedModel}'.`);
          clearDeepLinkRunFromLocation();
          setSelectedRunIds([]);
          dl.applied = true;
          return;
        }
        let baselineId: string | null = null;
        let comparatorId: string | null = null;
        if (dl.rawComparatorRunId) {
          if (!knownRunIds.has(dl.rawComparatorRunId)) {
            alert(`Comparator run '${dl.rawComparatorRunId}' is not present in the current registry for model '${selectedModel}'.`);
            clearDeepLinkRunFromLocation();
            setSelectedRunIds([]);
            dl.applied = true;
            return;
          }
          baselineId = findBaselineForRunId(dl.rawRunId!) || findBaselineForRunId(dl.rawComparatorRunId);
          comparatorId = dl.rawComparatorRunId;
        } else {
          const picked = pickComparator(dl.rawRunId!, dl.compareMode);
          baselineId = picked.baselineId || findBaselineForRunId(dl.rawRunId!);
          comparatorId = picked.comparatorId;
        }

        const ids = Array.from(new Set([dl.rawRunId!, comparatorId].filter(Boolean) as string[]));
        const unique = ids.filter((id) => knownRunIds.has(id) && hasRunData(id));
        if (!unique.includes(dl.rawRunId!)) {
          alert(`Run '${dl.rawRunId}' was not found in current data for model '${selectedModel}'.`);
          clearDeepLinkRunFromLocation();
          setSelectedRunIds([]);
          dl.applied = true;
          return;
        }

        if (shouldExpandBaselineForDeepLink({ baselineId, rawRunId: dl.rawRunId, comparatorId })) {
          const expandedBaselineId = String(baselineId || '').trim();
          setExpandedInterventions((prev) =>
            prev.includes(expandedBaselineId) ? prev : [...prev, expandedBaselineId]
          );
        }

        await ensureRunsLoaded(unique);
        if (dl.noiseProfile !== null) setSelectedNoiseProfile(dl.noiseProfile);
        if (dl.noiseSeed !== null) setNoiseSeed(Math.trunc(Number(dl.noiseSeed)));
        setSelectedRunIds(unique);
      } catch (err) {
        alert(`Failed to apply deep link for run '${dl.rawRunId}': ${formatApiError(err, 'Failed to apply deep link.')}`);
      } finally {
        dl.applied = true;
      }
    })();
  }, [
    selectedModel,
    modelRecord,
    diskRuns,
    runsData,
    availableSignals,
    setNoiseSeed,
  ]);

  const hasRunData = (runId: string, _runs = registry, dataRuns = diskRuns) => dataRuns.includes(runId);

  const handleReloadRegistryFromFile = async () => {
    if (!selectedModel || loading) return;
    setRunsData({});
    setSelectedRunIds([]);
    setExpandedInterventions([]);
    await fetchRegistry(selectedModel);
    await fetchDistribution(selectedModel);
  };

  const fetchRegistry = async (modelKey = selectedModel, requestId?: number) => {
    if (!modelKey) return null;
    const nextRequestId = typeof requestId === 'number'
      ? requestId
      : (modelLoadRequestIdRef.current + 1);
    if (typeof requestId !== 'number') modelLoadRequestIdRef.current = nextRequestId;
    setLoading(true);
    try {
      const controller = new AbortController();
      const res = await axios.get(`/api/registry?model=${modelKey}`, { signal: controller.signal });
      if (modelLoadRequestIdRef.current !== nextRequestId) {
        controller.abort();
        return null;
      }
      const loadedModel = res.data?.modelRecord;
      if (!loadedModel?.metadata || typeof loadedModel.metadata !== 'object') {
        throw new Error('Registry response missing modelRecord metadata.');
      }
      const normalized = normalizeRegistry(loadedModel.baselines ?? [], {
        selectedModel: modelKey,
        buildInterventionDisplayNames,
      });
      const nextAvailableNoiseProfiles: string[] = Array.isArray(res.data?.availableNoiseProfiles)
        ? Array.from(new Set(
          res.data.availableNoiseProfiles
            .map((value: unknown) => String(value ?? '').trim().toLowerCase())
            .filter((value: string) => value === 'none' || value === 'low' || value === 'high')
        ))
        : ['none'];
      if (!nextAvailableNoiseProfiles.includes('none')) nextAvailableNoiseProfiles.unshift('none');

      const registrySamplingRates = new Set<number>();
      for (const run of normalized) {
        const sr = Number((run as any)?.sampling_rate_hz);
        if (Number.isFinite(sr) && sr > 0) registrySamplingRates.add(sr);
      }
      if (registrySamplingRates.size === 1) {
        const only = Array.from(registrySamplingRates)[0];
        setSamplingRate(only);
        setSamplingRateDraft(String(only));
      } else {
        setSamplingRate(null);
        setSamplingRateDraft('');
      }
      setSamplingRateDirty(false);
      setSamplingRateError('');
      setSamplingRatePlotOverride(null);

      const diskRunIds = Array.isArray(res.data.diskRuns) ? res.data.diskRuns : [];
      const nextModelRecord = { ...loadedModel, version: 1, model_id: modelKey, baselines: normalized };
      const sanitizedUiState = sanitizeRegistryBoundUiState(normalized, {
        selectedRunIds,
        expandedInterventions,
        selectedRunRowIds: [],
        runsData,
        bulkTimeByRunId,
      });
      setModelRecord(nextModelRecord);
      setOriginalModelRecord(JSON.parse(JSON.stringify(nextModelRecord)));
      setDiskRuns(diskRunIds);
      setSelectedRunIds(sanitizedUiState.selectedRunIds);
      setExpandedInterventions(sanitizedUiState.expandedInterventions);
      setRunsData(sanitizedUiState.runsData);
      setBulkTimeByRunId(sanitizedUiState.bulkTimeByRunId);
      registryModelKeyRef.current = modelKey;
      setSignalDisplayNames(
        res.data?.signalDisplayNames && typeof res.data.signalDisplayNames === 'object'
          ? res.data.signalDisplayNames
          : {}
      );
      setAvailableNoiseProfiles(nextAvailableNoiseProfiles);
      setSelectedNoiseProfile((prev) => (
        nextAvailableNoiseProfiles.includes(String(prev || '').trim().toLowerCase())
          ? String(prev || '').trim().toLowerCase()
          : 'none'
      ));
      const initialAvailableSignals = Array.isArray(res.data?.availableSignals)
        ? res.data.availableSignals.map((s: unknown) => String(s ?? '').trim()).filter(Boolean)
        : [];
      setAvailableSignals(initialAvailableSignals);
      setSelectedSignals(initialAvailableSignals);
      prevAvailableSignalsRef.current = initialAvailableSignals;
      setHistory([JSON.parse(JSON.stringify(nextModelRecord))]);
      setHistoryIndex(0);
      return { registry: normalized, diskRuns: diskRunIds };
    } catch (err) {
      console.error(err);
      if (modelLoadRequestIdRef.current === nextRequestId) {
        registryModelKeyRef.current = '';
        setModelRecord(null);
        setOriginalModelRecord(null);
        setDiskRuns([]);
        setRunsData({});
        setSelectedRunIds([]);
        setExpandedInterventions([]);
        setAvailableSignals([]);
        setSelectedSignals([]);
        setSignalDisplayNames({});
        setAvailableNoiseProfiles(['none']);
        setSelectedNoiseProfile('none');
        setSelectedSignalCases([]);
        setHistory([]);
        setHistoryIndex(0);
        setBulkTimeByRunId({});
        alert(`Failed to load registry for model '${modelKey}': ${formatApiError(err, 'Failed to load registry.')}`);
      }
      return null;
    } finally {
      if (modelLoadRequestIdRef.current === nextRequestId) setLoading(false);
    }
  };

  const fetchDistribution = async (modelKey = selectedModel, requestId?: number) => {
    if (!modelKey) return;
    const activeRequestId = typeof requestId === 'number' ? requestId : modelLoadRequestIdRef.current;
    try {
      const controller = new AbortController();
      const res = await axios.get(`/api/distribution?model=${modelKey}`, { signal: controller.signal });
      if (modelLoadRequestIdRef.current !== activeRequestId) {
        controller.abort();
        return;
      }
      const details = buildSamplingDetails(res.data.distribution);
      const exposedGroups = extractExposedVariableGroups(res.data.distribution);
      setSamplingDetails(details);
      setSamplingIntervals(buildSamplingIntervals(details, 'initial'));
      setSamplingPerturbationIntervals(buildSamplingIntervals(details, 'perturbation'));
      setExposedParameterKeys(exposedGroups.parameters);
      setExposedInitialStateKeys(exposedGroups.initialState);
    } catch (err) {
      if (modelLoadRequestIdRef.current !== activeRequestId) return;
      console.error(err);
      setSamplingIntervals({});
      setSamplingPerturbationIntervals({});
      setSamplingDetails({});
      setExposedParameterKeys([]);
      setExposedInitialStateKeys([]);
      alert(`Failed to load sampling details for model '${modelKey}': ${formatApiError(err, 'Failed to load sampling details.')}`);
    }
  };

  const ensureRunsLoaded = async (runIds: string[], options?: { force?: boolean }) => {
    const force = Boolean(options?.force);
    const unique = Array.from(new Set(runIds));
    const missing = unique.filter(id => hasRunData(id) && (force || !hasFreshRunData(id)));
    if (missing.length === 0) return {};
    setLoading(true);
    try {
      const results = await Promise.all(missing.map(async runId => {
        try {
          const data = await fetchRunData(runId);
          return { runId, data, error: '' };
        } catch (err) {
          return { runId, data: null, error: formatRunDataError(err) };
        }
      }));
      const loaded: Record<string, any> = {};
      const failures: Array<{ runId: string; error: string }> = [];
      for (const result of results) {
        if (result?.data) loaded[result.runId] = result.data;
        else if (result?.error) failures.push({ runId: result.runId, error: result.error });
      }
      alertRunDataFailures(failures);
      return loaded;
    } finally {
      setLoading(false);
    }
  };

  const toggleRunVisualization = async (runId: string) => {
    if (selectedRunIds.includes(runId)) {
      setSelectedRunIds(prev => prev.filter(id => id !== runId));
      return;
    }

    setSelectedRunIds(prev => (prev.includes(runId) ? prev : [...prev, runId]));
    if (!hasFreshRunData(runId)) {
      setLoading(true);
      try {
        const loaded = await fetchRunData(runId);
        if (!loaded) {
          setSelectedRunIds(prev => prev.filter(id => id !== runId));
          return;
        }
      } catch (err: any) {
        alert(formatRunDataError(err));
        setSelectedRunIds(prev => prev.filter(id => id !== runId));
        return;
      } finally {
        setLoading(false);
      }
    }
  };

  const toggleRunVisualizationGroup = async (run: RunRecord) => {
    const subtreeIds = collectBaselineSubtreeRunIds(run).filter(id => hasRunData(id));
    if (subtreeIds.length === 0) return;

    const allSelected = subtreeIds.every(id => selectedRunIds.includes(id));
    if (allSelected) {
      setSelectedRunIds(prev => prev.filter(id => !subtreeIds.includes(id)));
      return;
    }

    setSelectedRunIds(prev => Array.from(new Set([...prev, ...subtreeIds])));
    const missingData = subtreeIds.filter(id => !hasFreshRunData(id));
    if (missingData.length > 0) {
      setLoading(true);
      try {
        const results = await Promise.all(missingData.map(async (runId) => {
          try {
            await fetchRunData(runId);
            return { runId, loaded: true, error: '' };
          } catch (err) {
            return { runId, loaded: false, error: formatRunDataError(err) };
          }
        }));
        const failedIds = results
          .filter((result) => !result.loaded)
          .map((result) => result.runId);
        if (failedIds.length > 0) {
          setSelectedRunIds(prev => prev.filter(id => !failedIds.includes(id)));
        }
        alertRunDataFailures(
          results
            .filter((result) => !result.loaded)
            .map((result) => ({ runId: result.runId, error: result.error }))
        );
      } finally {
        setLoading(false);
      }
    }
  };

  function formatRunDataError(err: any): string {
    return formatApiError(err, 'Failed to load run data.');
  }

  function alertRunDataFailures(failures: Array<{ runId: string; error: string }>) {
    const lines = failures
      .map((failure) => ({
        runId: String(failure.runId || '').trim(),
        error: String(failure.error || '').trim(),
      }))
      .filter((failure) => failure.runId && failure.error)
      .map((failure) => `${failure.runId}: ${failure.error}`);
    const uniqueLines = Array.from(new Set(lines));
    if (uniqueLines.length > 0) {
      alert(uniqueLines.join('\n'));
    }
  }

  const buildRunDataCacheKey = React.useCallback((
    runId: string,
    referenceRunId: string,
    noiseProfile: string,
    seed: number,
  ) => {
    const normalizedProfile = String(noiseProfile || 'none').trim().toLowerCase() || 'none';
    const normalizedSeed = Math.max(0, Math.trunc(Number(seed) || 0));
    return `${String(runId || '').trim()}::${String(referenceRunId || '').trim()}::${normalizedProfile}::${normalizedSeed}`;
  }, []);

  const activeNoiseProfile = React.useMemo(() => (
    availableNoiseProfiles.includes(selectedNoiseProfile) ? selectedNoiseProfile : 'none'
  ), [availableNoiseProfiles, selectedNoiseProfile]);

  const referenceRunIdForRun = React.useCallback((runId: string, runIds = selectedRunIds) => {
    if (activeNoiseProfile === 'none') return '';
    const normalizedRunId = String(runId || '').trim();
    const normalized = runIds.map((id) => String(id || '').trim()).filter(Boolean);
    if (normalized.length !== 2 || normalized[0] !== normalizedRunId) return '';
    return normalized[1] || '';
  }, [activeNoiseProfile, selectedRunIds.join('|')]);

  async function fetchRunData(runId: string) {
    const normalizedNoiseProfile = activeNoiseProfile;
    const referenceRunId = referenceRunIdForRun(runId);
    const cacheKey = buildRunDataCacheKey(runId, referenceRunId, normalizedNoiseProfile, noiseSeed);
    const query = new URLSearchParams({
      model: selectedModel,
      run: runId,
      max_rows: '50000',
      noise_profile: normalizedNoiseProfile,
      noise_seed: String(Math.max(0, Math.trunc(Number(noiseSeed ?? 0) || 0))),
      _t: String(Date.now()),
    });
    if (referenceRunId) query.set('reference_run', referenceRunId);
    const res = await axios.get(`/api/data?${query.toString()}`, { validateStatus: status => status < 500 });
    if (res.status >= 400 || res.data?.error) {
      throw new Error(res.data?.error || `Failed to load data (HTTP ${res.status}).`);
    }
    const nextData = { ...res.data, _loadedAt: Date.now(), _cacheKey: cacheKey };
    setRunsData(prev => ({ ...prev, [runId]: nextData }));
    return nextData;
  }

  const hasFreshRunData = React.useCallback((runId: string) => {
    const cached = runsData[runId];
    if (!cached) return false;
    const referenceRunId = referenceRunIdForRun(runId);
    return String(cached?._cacheKey || '').trim() === buildRunDataCacheKey(
      runId,
      referenceRunId,
      activeNoiseProfile,
      noiseSeed,
    );
  }, [activeNoiseProfile, buildRunDataCacheKey, noiseSeed, referenceRunIdForRun, runsData]);

  useEffect(() => {
    const selectedRunIdsKey = selectedRunIds.map((runId) => String(runId || '').trim()).filter(Boolean).join('|');
    const refreshKey = `${String(selectedModel || '').trim()}::${activeNoiseProfile}::${String(noiseSeed)}::${selectedRunIdsKey}`;
    if (!selectedModel) {
      noiseRefreshAttemptedKeyRef.current = '';
      noiseRefreshInFlightKeysRef.current.clear();
      return;
    }
    if (!selectedRunIdsKey) {
      noiseRefreshAttemptedKeyRef.current = refreshKey;
      return;
    }
    if (noiseRefreshAttemptedKeyRef.current === refreshKey) return;
    if (noiseRefreshInFlightKeysRef.current.has(refreshKey)) return;

    const staleRunIds = selectedRunIds.filter((runId) => {
      if (!hasRunData(runId)) return false;
      const expectedCacheKey = buildRunDataCacheKey(
        runId,
        referenceRunIdForRun(runId),
        activeNoiseProfile,
        noiseSeed,
      );
      return String(runsData[runId]?._cacheKey || '').trim() !== expectedCacheKey;
    });
    noiseRefreshAttemptedKeyRef.current = refreshKey;
    if (staleRunIds.length === 0) return;

    noiseRefreshInFlightKeysRef.current.add(refreshKey);
    setLoading(true);
    void Promise.all(staleRunIds.map(async (runId) => {
      try {
        await fetchRunData(runId);
        return null;
      } catch (err) {
        return { runId, error: formatRunDataError(err) };
      }
    })).then((failures) => {
      alertRunDataFailures(failures.filter(Boolean) as Array<{ runId: string; error: string }>);
    }).finally(() => {
      noiseRefreshInFlightKeysRef.current.delete(refreshKey);
      if (noiseRefreshInFlightKeysRef.current.size === 0) setLoading(false);
    });
  }, [
    activeNoiseProfile,
    buildRunDataCacheKey,
    noiseSeed,
    referenceRunIdForRun,
    runsData,
    selectedModel,
    selectedRunIds.join('|'),
    setLoading,
  ]);

  useEffect(() => {
    const prevAvailableSignals = prevAvailableSignalsRef.current;
    prevAvailableSignalsRef.current = availableSignals;
    if (availableSignals.length === 0) return;
    setSelectedSignals((prev) => {
      if (prevAvailableSignals.length === 0) return availableSignals;
      const availableSet = new Set(availableSignals);
      return prev.filter((signal) => availableSet.has(signal));
    });
  }, [availableSignals, setSelectedSignals]);

  const handleVisibleSignalsChange = React.useCallback((visibleSignals: string[]) => {
    const visibleSet = new Set(
      (Array.isArray(visibleSignals) ? visibleSignals : [])
        .map((signal) => String(signal ?? '').trim())
        .filter(Boolean)
    );
    setSelectedSignals(availableSignals.filter((signal) => visibleSet.has(signal)));
  }, [availableSignals, setSelectedSignals]);

  useEffect(() => {
    if (availableSignals.length === 0) return;
    const availableSet = new Set(availableSignals);
    setSignalDisplayNames((prev) => {
      const entries = Object.entries(prev);
      if (entries.length === 0) return prev;
      const next: Record<string, string> = {};
      for (const [signal, label] of entries) {
        if (availableSet.has(signal)) next[signal] = label;
      }
      return next;
    });
  }, [availableSignals, setSignalDisplayNames]);

  const fetchSimulationStatus = async () => {
    try {
      const res = await axios.get(`/api/simulate?model=${selectedModel}&_t=${Date.now()}`);
      return res.data.status as SimulationStatus;
    } catch (err) {
      throw new Error(formatApiError(err, 'Failed to load simulation status.'));
    }
  };

  const waitForSimulationCompletion = async () => {
    const status = await fetchSimulationStatus();
    if (status?.status !== 'running') return status;
    return status;
  };

  const getPreviousValue = (runIndex: number, ivIndex: number, variable: string) => {
    const run = registry[runIndex];
    const iv = run?.interventions?.[ivIndex];
    if (!run) return 0;
    if (!iv || !iv.parent_id || iv.parent_id === run.run_id) return run.parameters?.[variable] || 0;
    const parent = run.interventions.find((item) => item.name === iv.parent_id);
    return parent?.variable === variable ? parent.value : run.parameters?.[variable] || 0;
  };

  const paramKeys = Array.from(new Set(registry.flatMap(r => Object.keys(r.parameters || {})))).sort();

  return {
    canVisualizeSamplingRate: true,
    currentTimeSeriesSamplingRate,
    formatTimeLabel,
    formatVariableLabel,
    getPreviousValue,
    handleReloadRegistryFromFile,
    handleVisibleSignalsChange,
    hasRunData,
    isValidNumberInput,
    paramKeys,
    shouldIgnoreRowClick,
    toggleRunVisualization,
    toggleRunVisualizationGroup,
    waitForSimulationCompletion,
  };
}
