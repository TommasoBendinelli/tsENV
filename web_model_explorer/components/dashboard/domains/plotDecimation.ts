const DECIMATION_EPS = 1e-12;

const toFiniteNumber = (value: unknown) => {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
};

const median = (values: number[]) => {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
};

const lowerBound = (sorted: number[], value: number) => {
  let lo = 0;
  let hi = sorted.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (sorted[mid] < value) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
};

export const estimateSamplingRateHzFromTimes = (timeValues: Array<number | null>) => {
  const finiteTimes = timeValues
    .map((time) => toFiniteNumber(time))
    .filter((time): time is number => time !== null);
  if (finiteTimes.length < 2) return null;
  const deltas: number[] = [];
  let prev: number | null = null;
  for (const time of finiteTimes) {
    if (prev !== null) {
      const dt = time - prev;
      if (Number.isFinite(dt) && dt > 0) deltas.push(dt);
    }
    prev = time;
  }
  const dt = median(deltas);
  if (dt === null || !Number.isFinite(dt) || dt <= 0) return null;
  const samplingRateHz = 1 / dt;
  return Number.isFinite(samplingRateHz) && samplingRateHz > 0 ? samplingRateHz : null;
};

export type TimeRateDecimationPlan = {
  indices: number[];
  windows: number[][];
};

export const computeTimeRateDecimationPlan = ({
  timeValues,
  targetSamplingRateHz,
}: {
  timeValues: Array<number | null>;
  targetSamplingRateHz: number | null;
}): TimeRateDecimationPlan | null => {
  const target = toFiniteNumber(targetSamplingRateHz);
  if (target === null || target <= 0) return null;

  const finitePairs = timeValues
    .map((time, idx) => ({ idx, time: toFiniteNumber(time) }))
    .filter((item): item is { idx: number; time: number } => item.time !== null);
  if (finitePairs.length < 2) return { indices: [], windows: [] };

  const finiteTimes = finitePairs.map((item) => item.time);
  const baseSamplingRateHz = estimateSamplingRateHzFromTimes(finiteTimes);
  if (baseSamplingRateHz === null) return { indices: [], windows: [] };
  if (target >= baseSamplingRateHz) return null;

  const stride = Math.max(1, Math.round(baseSamplingRateHz / target));
  const startTime = 1 / target;
  const startPos = lowerBound(finiteTimes, startTime - DECIMATION_EPS);
  const indices: number[] = [];
  const windows: number[][] = [];
  for (let pos = startPos; pos < finitePairs.length; pos += stride) {
    const windowPairs = finitePairs.slice(pos, Math.min(pos + stride, finitePairs.length));
    indices.push(finitePairs[pos].idx);
    windows.push(windowPairs.map((item) => item.idx));
  }
  return { indices, windows };
};

export const computeTimeRateDecimatedIndices = ({
  timeValues,
  targetSamplingRateHz,
}: {
  timeValues: Array<number | null>;
  targetSamplingRateHz: number | null;
}) => {
  const plan = computeTimeRateDecimationPlan({ timeValues, targetSamplingRateHz });
  return plan === null ? null : plan.indices;
};

export const decimateIndicesToMaxPoints = (indices: number[], maxPoints: number) => {
  if (!Number.isFinite(maxPoints) || maxPoints <= 0) return [];
  if (indices.length <= maxPoints) return indices;
  const stride = Math.max(1, Math.ceil(indices.length / maxPoints));
  const out: number[] = [];
  for (let i = 0; i < indices.length; i += stride) {
    out.push(indices[i]);
  }
  return out;
};

export const pickSignedMaxAbsByWindows = (
  values: Array<number | null>,
  windows: number[][],
) => windows.map((window) => {
  let best: number | null = null;
  let bestAbs = -Infinity;
  for (const idx of window) {
    const raw = toFiniteNumber(values[idx]);
    if (raw === null) continue;
    const absRaw = Math.abs(raw);
    if (absRaw > bestAbs) {
      bestAbs = absRaw;
      best = raw;
    }
  }
  return best;
});
