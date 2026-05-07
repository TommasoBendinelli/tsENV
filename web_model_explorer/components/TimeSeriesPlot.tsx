'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import { computeVisibleSignalsFromPlotlyData } from './timeSeriesPlotVisibility';
import { computeTimeRateDecimationPlan, decimateIndicesToMaxPoints, pickSignedMaxAbsByWindows } from './dashboard/domains/plotDecimation';

// Plotly doesn't like SSR
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface TimeSeriesPlotProps {
  allRunsData: Record<string, {
    columns: string[];
    index: number[];
    data: any[][];
  }>;
  selectedRunIds: string[];
  availableSignals: string[];
  selectedSignals: string[];
  samplingRate?: number | null;
  samplingRateOverride?: number | null;
  modelId?: string | null;
  signalDisplayNames?: Record<string, string>;
  formatSignalLabel?: (signal: string, signalDisplayNames?: Record<string, string>) => string;
  onVisibleSignalsChange?: (visibleSignals: string[]) => void;
}

const toFiniteNumberOrNull = (value: any) => {
  const num = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(num) ? num : null;
};

export default function TimeSeriesPlot({
  allRunsData,
  selectedRunIds,
  availableSignals,
  selectedSignals,
  samplingRate,
  samplingRateOverride,
  modelId,
  signalDisplayNames,
  formatSignalLabel,
  onVisibleSignalsChange,
}: TimeSeriesPlotProps) {
  const MAX_POINTS_PER_TRACE = 5000;
  const graphDivRef = React.useRef<any>(null);
  const lastVisibleSigRef = React.useRef<string>('');
  const selectedSignalSet = React.useMemo(
    () => new Set(selectedSignals),
    [selectedSignals]
  );

  const emitVisibleSignals = React.useCallback(() => {
    if (!onVisibleSignalsChange) return;
    const nextVisible = computeVisibleSignalsFromPlotlyData(graphDivRef.current?.data, availableSignals);
    const sig = nextVisible.join('|');
    if (sig === lastVisibleSigRef.current) return;
    lastVisibleSigRef.current = sig;
    onVisibleSignalsChange(nextVisible);
  }, [availableSignals, onVisibleSignalsChange]);

  const scheduleEmitVisibleSignals = React.useCallback((graphDiv?: any) => {
    if (!onVisibleSignalsChange) return;
    if (graphDiv && Array.isArray(graphDiv.data)) graphDivRef.current = graphDiv;
    // Allow Plotly to apply the restyle first, then read visibility from graphDiv.
    setTimeout(() => emitVisibleSignals(), 0);
  }, [emitVisibleSignals, onVisibleSignalsChange]);

  const { traces, usesTimeAxis } = React.useMemo(() => {
    const traces: any[] = [];
    let usesTimeAxis = true;
    const parsedOverride = Number(samplingRateOverride);
    const targetSamplingRateHz = Number.isFinite(parsedOverride) && parsedOverride > 0 ? parsedOverride : null;
    const parsedSamplingRate = Number(samplingRate);
    const fallbackSamplingRateHz = Number.isFinite(parsedSamplingRate) && parsedSamplingRate > 0
      ? parsedSamplingRate
      : null;

    for (const runId of selectedRunIds) {
      const data = allRunsData[runId];
      if (!data) continue;
      const timeIndex = data.columns.indexOf('time');
      const hasTime = timeIndex !== -1;
      if (!hasTime) {
        usesTimeAxis = false;
      }

      const sampleIndex = Array.isArray(data.index) && data.index.length === data.data.length
        ? data.index
        : Array.from({ length: data.data.length }, (_, idx) => idx);
      const rows = Array.isArray(data.data) ? data.data : [];

      let indices: number[] | null = null;
      let rateDecimationWindows: number[][] | null = null;
      if (targetSamplingRateHz !== null) {
        if (hasTime) {
          const timeValues = rows.map((row) => toFiniteNumberOrNull(row?.[timeIndex]));
          const plan = computeTimeRateDecimationPlan({
            timeValues,
            targetSamplingRateHz,
          });
          if (plan !== null) {
            indices = plan.indices;
            rateDecimationWindows = plan.windows;
          }
        } else if (fallbackSamplingRateHz !== null && targetSamplingRateHz < fallbackSamplingRateHz) {
          const stride = Math.max(1, Math.round(fallbackSamplingRateHz / targetSamplingRateHz));
          indices = Array.from({ length: Math.ceil(rows.length / stride) }, (_, idx) => idx * stride)
            .filter((idx) => idx < rows.length);
        }
      }
      if (indices === null && rows.length > MAX_POINTS_PER_TRACE) {
        const stride = Math.max(1, Math.ceil(rows.length / MAX_POINTS_PER_TRACE));
        indices = Array.from({ length: Math.ceil(rows.length / stride) }, (_, idx) => idx * stride)
          .filter((idx) => idx < rows.length);
      } else if (indices !== null && indices.length > MAX_POINTS_PER_TRACE) {
        if (rateDecimationWindows) {
          const stride = Math.max(1, Math.ceil(indices.length / MAX_POINTS_PER_TRACE));
          indices = indices.filter((_, idx) => idx % stride === 0);
          rateDecimationWindows = rateDecimationWindows.filter((_, idx) => idx % stride === 0);
        } else {
          indices = decimateIndicesToMaxPoints(indices, MAX_POINTS_PER_TRACE);
        }
      }

      const plotRows = indices ? indices.map((idx) => rows[idx]) : rows;
      const plotX = hasTime
        ? (indices ? indices.map((idx) => toFiniteNumberOrNull(rows[idx]?.[timeIndex])) : rows.map((row) => toFiniteNumberOrNull(row?.[timeIndex])))
        : (indices ? indices.map((idx) => sampleIndex[idx]) : sampleIndex);
      const plotIdx = indices ? indices.map((idx) => sampleIndex[idx]) : sampleIndex;
      const useMassSpringSignedWindow = modelId === 'MassSpringDamperWithPID' && rateDecimationWindows !== null;

      for (const signal of availableSignals) {
        const signalIndex = data.columns.indexOf(signal);
        if (signalIndex === -1) continue;
        const displayLabel = formatSignalLabel
          ? formatSignalLabel(signal, signalDisplayNames)
          : (signalDisplayNames?.[signal] || signal);
        const rawY = useMassSpringSignedWindow
          ? pickSignedMaxAbsByWindows(
            rows.map((row) => {
              const raw = Number(row?.[signalIndex]);
              return Number.isFinite(raw) ? raw : null;
            }),
            rateDecimationWindows || [],
          )
          : plotRows.map((row) => {
            const raw = Number(row[signalIndex]);
            return Number.isFinite(raw) ? raw : null;
          });
        const hoverTemplate = hasTime
          ? 'idx=%{customdata}<br>time=%{x}<br>value=%{y}<extra>%{fullData.name}</extra>'
          : 'idx=%{customdata}<br>index=%{x}<br>value=%{y}<extra>%{fullData.name}</extra>';

        traces.push({
          x: plotX,
          y: rawY,
          customdata: plotIdx,
          hovertemplate: hoverTemplate,
          name: `${displayLabel} (${runId})`,
          type: 'scatter',
          mode: 'lines',
          visible: selectedSignalSet.has(signal) ? true : 'legendonly',
          meta: { signalKey: signal, runId },
        });
      }
    }

    return { traces, usesTimeAxis };
  }, [
    allRunsData,
    formatSignalLabel,
    samplingRate,
    samplingRateOverride,
    modelId,
    availableSignals,
    selectedRunIds,
    selectedSignals,
    selectedSignalSet,
    signalDisplayNames,
  ]);

  return (
    <div className="w-full h-full min-h-[400px]">
      <Plot
        data={traces}
        layout={{
          autosize: true,
          margin: { l: 50, r: 50, b: 50, t: 50, pad: 4 },
          xaxis: { title: { text: usesTimeAxis ? 'Time [s]' : 'Index' } },
          yaxis: { title: { text: 'Value' } },
          template: 'plotly_white',
          legend: { orientation: 'h', y: -0.2 },
          hovermode: 'closest',
        } as any}
        style={{ width: '100%', height: '100%' }}
        useResizeHandler={true}
        onInitialized={(_, graphDiv) => {
          graphDivRef.current = graphDiv;
        }}
        onUpdate={(_, graphDiv) => {
          graphDivRef.current = graphDiv;
        }}
        onRestyle={() => {
          scheduleEmitVisibleSignals();
        }}
      />
    </div>
  );
}
