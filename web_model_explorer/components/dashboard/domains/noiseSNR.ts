import noiseRulesJson from '@/config/noise_rules.json';

export type NoiseRules = {
  rolling_window_points: number;
  sigma_floor_ratio: number;
};

export const DEFAULT_NOISE_RULES: NoiseRules = {
  rolling_window_points: 11,
  sigma_floor_ratio: 1e-4,
};

const toFiniteNumber = (value: unknown, fallback: number) => {
  const out = Number(value);
  return Number.isFinite(out) ? out : fallback;
};

const normalizeWindow = (value: unknown, fallback: number) => {
  let out = Math.trunc(toFiniteNumber(value, fallback));
  if (out < 1) out = fallback;
  if (out % 2 === 0) out += 1;
  return out;
};

export const NOISE_RULES: NoiseRules = {
  rolling_window_points: normalizeWindow(
    (noiseRulesJson as Partial<NoiseRules>).rolling_window_points,
    DEFAULT_NOISE_RULES.rolling_window_points,
  ),
  sigma_floor_ratio: toFiniteNumber(
    (noiseRulesJson as Partial<NoiseRules>).sigma_floor_ratio,
    DEFAULT_NOISE_RULES.sigma_floor_ratio,
  ),
};

export const noiseMultiplierFromSNRDb = (snrDb: number) => {
  const snr = toFiniteNumber(snrDb, 20);
  return 10 ** (-snr / 20);
};

export const snrDbFromNoiseMultiplier = (noiseMultiplier: number) => {
  const m = toFiniteNumber(noiseMultiplier, 0);
  if (m <= 0) return Number.POSITIVE_INFINITY;
  return -20 * Math.log10(m);
};

export const hashString = (value: string) => {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
};

export const seedForRun = (baseSeed: number, runId: string) => {
  let hash = 2166136261 >>> 0;
  const text = String(runId ?? '');
  for (let i = 0; i < text.length; i += 1) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619) >>> 0;
  }
  const base = Math.trunc(toFiniteNumber(baseSeed, 0)) >>> 0;
  return (base ^ hash) >>> 0;
};

export const makeRng = (seed: number) => {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let result = Math.imul(t ^ (t >>> 15), 1 | t);
    result ^= result + Math.imul(result ^ (result >>> 7), 61 | result);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
};

export const gaussianSample = (rng: () => number) => {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

export const rollingRms = (values: number[], windowPoints: number) => {
  const n = values.length;
  if (n === 0) return [] as number[];
  const w = normalizeWindow(windowPoints, 1);
  const half = Math.floor(w / 2);
  const out = new Array<number>(n).fill(0);
  for (let i = 0; i < n; i += 1) {
    const start = Math.max(0, i - half);
    const end = Math.min(n, i + half + 1);
    let sumSq = 0;
    let count = 0;
    for (let j = start; j < end; j += 1) {
      const v = values[j];
      if (!Number.isFinite(v)) continue;
      sumSq += v * v;
      count += 1;
    }
    out[i] = count > 0 ? Math.sqrt(sumSq / count) : 0;
  }
  return out;
};

export const sampleAdaptiveGaussianNoise = (
  values: Array<number | null>,
  {
    noiseMultiplier,
    seedKey,
    seed = 0,
    rules = NOISE_RULES,
  }: {
    noiseMultiplier: number;
    seedKey: string;
    seed?: number;
    rules?: NoiseRules;
  },
) => {
  const mult = toFiniteNumber(noiseMultiplier, 0);
  const n = values.length;
  const out = new Array<number>(n).fill(0);
  if (n === 0 || mult <= 0) return out;

  const arr = values.map((v) => (v === null ? Number.NaN : Number(v)));
  const localRms = rollingRms(arr, rules.rolling_window_points);
  const finite = arr.filter((v) => Number.isFinite(v));
  const globalRms = finite.length > 0
    ? Math.sqrt(finite.reduce((acc, v) => acc + (v * v), 0) / finite.length)
    : 0;
  const sigmaFloor = toFiniteNumber(rules.sigma_floor_ratio, 0) * globalRms;
  const rng = makeRng(hashString(`${seedKey}:${Math.trunc(seed)}`));

  for (let i = 0; i < n; i += 1) {
    const base = arr[i];
    if (!Number.isFinite(base)) continue;
    const sigma = mult * Math.max(localRms[i], sigmaFloor);
    if (!Number.isFinite(sigma) || sigma <= 0) continue;
    out[i] = gaussianSample(rng) * sigma;
  }
  return out;
};

export const applyAdaptiveNoise = (
  values: Array<number | null>,
  {
    runNoiseMultiplier = 0,
    signalNoiseMultiplier = 0,
    runSeedKey,
    signalSeedKey,
    seed = 0,
    rules = NOISE_RULES,
  }: {
    runNoiseMultiplier?: number;
    signalNoiseMultiplier?: number;
    runSeedKey: string;
    signalSeedKey: string;
    seed?: number;
    rules?: NoiseRules;
  },
) => {
  const runNoise = sampleAdaptiveGaussianNoise(values, {
    noiseMultiplier: runNoiseMultiplier,
    seedKey: runSeedKey,
    seed,
    rules,
  });
  const signalNoise = sampleAdaptiveGaussianNoise(values, {
    noiseMultiplier: signalNoiseMultiplier,
    seedKey: signalSeedKey,
    seed,
    rules,
  });

  return values.map((value, i) => {
    if (value === null || !Number.isFinite(Number(value))) return null;
    return Number(value) + runNoise[i] + signalNoise[i];
  });
};

export const sampleAdaptiveAndBaseGaussianNoise = (
  values: Array<number | null>,
  {
    adaptiveNoiseMultiplier = 0,
    baseNoiseMultiplier = 0,
    absNoiseSigma = 0,
    seedKey,
    seed = 0,
    rules = NOISE_RULES,
  }: {
    adaptiveNoiseMultiplier?: number;
    baseNoiseMultiplier?: number;
    absNoiseSigma?: number;
    seedKey: string;
    seed?: number;
    rules?: NoiseRules;
  },
) => {
  const adaptive = toFiniteNumber(adaptiveNoiseMultiplier, 0);
  const base = toFiniteNumber(baseNoiseMultiplier, 0);
  const absSigma = toFiniteNumber(absNoiseSigma, 0);
  const n = values.length;
  const out = new Array<number>(n).fill(0);
  if (n === 0 || (adaptive <= 0 && base <= 0 && absSigma <= 0)) return out;

  const arr = values.map((v) => (v === null ? Number.NaN : Number(v)));
  const localRms = rollingRms(arr, rules.rolling_window_points);
  const finite = arr.filter((v) => Number.isFinite(v));
  const globalRms = finite.length > 0
    ? Math.sqrt(finite.reduce((acc, v) => acc + (v * v), 0) / finite.length)
    : 0;
  const sigmaFloor = toFiniteNumber(rules.sigma_floor_ratio, 0) * globalRms;
  const sigmaBase = base > 0 ? base * globalRms : 0;
  const rng = makeRng(hashString(`${seedKey}:${Math.trunc(seed)}`));

  for (let i = 0; i < n; i += 1) {
    const baseVal = arr[i];
    if (!Number.isFinite(baseVal)) continue;
    const sigmaAdaptive = adaptive > 0 ? adaptive * Math.max(localRms[i], sigmaFloor) : 0;
    const sigma = Math.hypot(sigmaAdaptive, sigmaBase, absSigma);
    if (!Number.isFinite(sigma) || sigma <= 0) continue;
    out[i] = gaussianSample(rng) * sigma;
  }
  return out;
};

export const applyAdaptiveAndBaseNoise = (
  values: Array<number | null>,
  {
    adaptiveNoiseMultiplier = 0,
    baseNoiseMultiplier = 0,
    absNoiseSigma = 0,
    seedKey,
    seed = 0,
    rules = NOISE_RULES,
  }: {
    adaptiveNoiseMultiplier?: number;
    baseNoiseMultiplier?: number;
    absNoiseSigma?: number;
    seedKey: string;
    seed?: number;
    rules?: NoiseRules;
  },
) => {
  const noise = sampleAdaptiveAndBaseGaussianNoise(values, {
    adaptiveNoiseMultiplier,
    baseNoiseMultiplier,
    absNoiseSigma,
    seedKey,
    seed,
    rules,
  });
  return values.map((value, i) => {
    if (value === null || !Number.isFinite(Number(value))) return null;
    return Number(value) + noise[i];
  });
};
