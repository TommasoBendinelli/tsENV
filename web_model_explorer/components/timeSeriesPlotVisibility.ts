export type PlotlyTraceLike = {
  visible?: unknown;
  meta?: unknown;
};

const isTraceVisible = (visible: unknown) => visible !== 'legendonly' && visible !== false;

/**
 * Extract the set of currently-visible signal keys from Plotly's `graphDiv.data`.
 * We rely on `trace.meta.signalKey` being set by our plotting code.
 */
export function computeVisibleSignalsFromPlotlyData(
  plotData: unknown,
  signalOrder: string[],
): string[] {
  if (!Array.isArray(signalOrder) || signalOrder.length === 0) return [];
  if (!Array.isArray(plotData)) return [...signalOrder];

  const visibleSignals = new Set<string>();

  for (const trace of plotData as PlotlyTraceLike[]) {
    const meta: any = (trace as any)?.meta;
    const signalKey = typeof meta?.signalKey === 'string' ? String(meta.signalKey) : null;
    if (!signalKey) continue;
    if (!isTraceVisible((trace as any)?.visible)) continue;
    visibleSignals.add(signalKey);
  }

  if (visibleSignals.size === 0) return [];
  return signalOrder.filter((s) => visibleSignals.has(s));
}
