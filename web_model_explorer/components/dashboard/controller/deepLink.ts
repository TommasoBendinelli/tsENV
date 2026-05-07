export type DeepLinkCompareMode = 'auto' | 'baseline' | 'time0' | 'sibling' | 'none';

export type ParsedDeepLink = {
  rawRunId: string | null;
  rawComparatorRunId: string | null;
  rawModelId: string | null;
  compareMode: DeepLinkCompareMode;
  noiseProfile: 'none' | 'low' | 'high' | null;
  noiseSeed: number | null;
};

export function parseDeepLinkFromSearch(search: string): ParsedDeepLink {
  const sp = new URLSearchParams(search || '');
  const parseOptionalInteger = (key: string): number | null => {
    if (!sp.has(key)) return null;
    const raw = String(sp.get(key) || '').trim();
    if (!raw) return null;
    const value = Number(raw);
    if (!Number.isFinite(value)) return null;
    return Math.trunc(value);
  };
  const parseNoiseProfile = (): 'none' | 'low' | 'high' | null => {
    if (!sp.has('noise_profile')) return null;
    const raw = String(sp.get('noise_profile') || '').trim().toLowerCase();
    if (!raw) return null;
    if (raw === 'none' || raw === 'low' || raw === 'high') return raw;
    return null;
  };
  const rawRunId = String(sp.get('run') || '').trim() || null;
  const rawComparatorRunId = String(
    sp.get('run2') || sp.get('compare_run') || ''
  ).trim() || null;
  const rawModelId = String(sp.get('model') || '').trim() || null;
  const noiseProfile = parseNoiseProfile();
  const noiseSeed = parseOptionalInteger('noise_seed');
  const hasCompareParam = sp.has('compare');
  const compareRaw = String(sp.get('compare') || '').trim().toLowerCase();

  let compareMode: DeepLinkCompareMode;
  if (!hasCompareParam) {
    // For plain ?run=... links, keep the view to a single run unless a comparator is explicitly provided.
    compareMode = rawComparatorRunId ? 'auto' : 'none';
  } else if (compareRaw === 'baseline') {
    compareMode = 'baseline';
  } else if (compareRaw === 'time0') {
    compareMode = 'time0';
  } else if (compareRaw === 'sibling') {
    compareMode = 'sibling';
  } else if (compareRaw === 'none') {
    compareMode = 'none';
  } else {
    compareMode = 'auto';
  }

  return {
    rawRunId,
    rawComparatorRunId,
    rawModelId,
    compareMode,
    noiseProfile,
    noiseSeed,
  };
}

export function shouldExpandBaselineForDeepLink(opts: {
  baselineId: string | null;
  rawRunId: string | null;
  comparatorId: string | null;
}): boolean {
  const baselineId = String(opts.baselineId || '').trim();
  if (!baselineId) return false;
  const rawRunId = String(opts.rawRunId || '').trim();
  const comparatorId = String(opts.comparatorId || '').trim();
  return (
    baselineId !== rawRunId
    || (comparatorId.length > 0 && comparatorId !== baselineId)
  );
}

type AnyIntervention = {
  name?: string | null;
  time0_baseline_uuid?: string | null;
};

type AnyBaseline = {
  run_id?: string | null;
  interventions?: AnyIntervention[] | null;
};

export function collectDeepLinkKnownRunIds(
  baselines: AnyBaseline[] | null | undefined
): Set<string> {
  const ids = new Set<string>();
  for (const baseline of baselines ?? []) {
    const baselineId = String(baseline?.run_id || '').trim();
    if (baselineId) ids.add(baselineId);
    for (const iv of baseline?.interventions ?? []) {
      const childId = String(iv?.name || '').trim();
      if (childId) ids.add(childId);
      const time0Id = String(iv?.time0_baseline_uuid || '').trim();
      if (time0Id) ids.add(time0Id);
    }
  }
  return ids;
}

export function stripRunQueryParamFromUrl(url: string): string {
  const parsed = new URL(url, 'http://localhost');
  parsed.searchParams.delete('run');
  parsed.searchParams.delete('run2');
  parsed.searchParams.delete('compare_run');
  const qs = parsed.searchParams.toString();
  const hash = parsed.hash || '';
  return `${parsed.pathname}${qs ? `?${qs}` : ''}${hash}`;
}

export function clearDeepLinkRunFromLocation(): void {
  if (typeof window === 'undefined') return;
  const current = `${window.location.pathname}${window.location.search}${window.location.hash}`;
  const next = stripRunQueryParamFromUrl(window.location.href);
  if (next !== current) {
    window.history.replaceState(null, '', next);
  }
}
