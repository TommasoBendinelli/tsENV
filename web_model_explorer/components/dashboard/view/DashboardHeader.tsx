'use client';

import React from 'react';
import { Activity, Loader2, RefreshCw } from 'lucide-react';
import { cn } from '../utils';
import { useDashboardControllerContext } from '@/components/dashboard/DashboardControllerContext';
import { useDashboardStore } from '../useDashboardStore';

export function DashboardHeader(props: { runsDirName?: string }) {
  const selectedModel = useDashboardStore((state) => state.selectedModel);
  const policies = useDashboardStore((state) => state.policies);
  const selectedPolicy = useDashboardStore((state) => state.selectedPolicy);
  const setSelectedPolicy = useDashboardStore((state) => state.setSelectedPolicy);
  const loading = useDashboardStore((state) => state.loading);
  const runsDirName = String(props.runsDirName || '').trim() || 'runs';
  const runsPathLabel = selectedModel
    ? `models/simulink/${selectedModel}/${runsDirName}`
    : `models/simulink/*/${runsDirName}`;

		  const {
	    handleReloadRegistryFromFile,
	  } = useDashboardControllerContext();

  return (
    <header className="h-16 bg-white border-b px-6 flex items-center justify-between z-10">
      <div className="flex items-center gap-3">
        <div className="bg-blue-100 p-2 rounded-lg text-blue-600">
          <Activity size={20} />
        </div>
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold text-gray-800 tracking-tight">
            {selectedModel ? selectedModel : 'Select a model'}
          </h1>
          <span
            className={cn(
              'hidden md:inline-flex items-center rounded-full border px-3 py-1 text-xs font-semibold',
              runsDirName === 'runs'
                ? 'border-gray-200 bg-gray-100 text-gray-600'
                : 'border-amber-200 bg-amber-50 text-amber-700'
            )}
            title={`Model Explorer reads run artifacts from ${runsPathLabel}`}
          >
            Runs folder: <span className="ml-1 font-mono">{runsDirName}</span>
          </span>
          <label className="hidden lg:flex items-center gap-2 text-xs font-semibold text-gray-500">
            <span>Policy</span>
            <select
              value={selectedPolicy}
              disabled={!selectedModel || loading || policies.length === 0}
              onChange={(event) => setSelectedPolicy(event.target.value)}
              className="h-8 min-w-[16rem] rounded-md border border-gray-200 bg-white px-2 font-mono text-xs text-gray-700 disabled:opacity-40"
              title="Resolved run-graph policy_id"
            >
              {policies.length === 0 ? (
                <option value="">No policies</option>
              ) : policies.map((policy) => (
                <option key={policy} value={policy}>{policy}</option>
              ))}
            </select>
          </label>
        </div>
      </div>

      <div className="flex items-center gap-2">
	        <button
	          onClick={handleReloadRegistryFromFile}
	          disabled={!selectedModel || loading}
	          className="flex items-center gap-2 px-3 py-2 text-gray-500 hover:bg-gray-100 rounded-lg transition-colors border bg-white disabled:opacity-30"
          title="Reload registry from file"
        >
          {loading ? <Loader2 className="animate-spin" size={18} /> : <RefreshCw size={18} />}
          <span className="text-sm font-medium">Reload from file</span>
        </button>
      </div>
    </header>
  );
}
