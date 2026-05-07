'use client';

import React from 'react';
import { X } from 'lucide-react';
import { useDashboardStore } from '../useDashboardStore';

export function DashboardOverlays() {
  const simulationNotice = useDashboardStore((state) => state.simulationNotice);
  const setSimulationNotice = useDashboardStore((state) => state.setSimulationNotice);

  return (
    <>
      {simulationNotice && (
        <div className="fixed top-20 right-6 z-30 w-80 animate-in slide-in-from-right duration-300">
          <div className="bg-white border shadow-xl rounded-2xl p-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-[11px] uppercase tracking-[0.25em] text-gray-400 font-semibold">
                  Simulation Status
                </p>
                <div className="mt-2 flex items-center gap-2">
                  <span className="text-xs font-semibold text-emerald-700 bg-emerald-50 border border-emerald-100 rounded-full px-2.5 py-1">
                    Done {simulationNotice.done}{simulationNotice.total !== undefined ? `/${simulationNotice.total}` : ''}
                  </span>
                  <span className="text-xs font-semibold text-amber-700 bg-amber-50 border border-amber-100 rounded-full px-2.5 py-1">
                    Remaining {simulationNotice.missing}
                  </span>
                  {simulationNotice.running && (
                    <span className="text-xs font-semibold text-blue-700 bg-blue-50 border border-blue-100 rounded-full px-2.5 py-1">
                      Running…
                    </span>
                  )}
                </div>
              </div>
              <button
                onClick={() => setSimulationNotice(null)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
                aria-label="Close simulation status"
              >
                <X size={16} />
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
