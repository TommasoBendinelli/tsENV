'use client';

import React from 'react';
import { Database } from 'lucide-react';
import { cn } from '../utils';
import { useDashboardStore } from '../useDashboardStore';

export function DashboardSidebar() {
  const models = useDashboardStore((state) => state.models);
  const modelValidation = useDashboardStore((state) => state.modelValidation);
  const selectedModel = useDashboardStore((state) => state.selectedModel);
  const setSelectedModel = useDashboardStore((state) => state.setSelectedModel);

  const handleSelectModel = (model: string) => {
    setSelectedModel(model);
  };

  return (
      <div className="w-64 bg-white border-r flex flex-col shadow-sm z-10">
        <div className="p-4 border-b flex items-center gap-2 font-bold text-blue-600 bg-gray-50/50">
          <Database size={20} />
          <span className="tracking-tight uppercase text-xs font-black">Model Explorer</span>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          {models.map(m => {
            const validation = modelValidation?.[m];
            const failed = Boolean(validation && !validation.ok);
            const tooltip = failed
              ? (validation?.reasons || []).filter(Boolean).join('\n')
              : '';
            return (
            <button
              key={m}
              onClick={() => handleSelectModel(m)}
              className={cn(
                "w-full text-left px-3 py-2 rounded-lg text-sm transition-all duration-200 flex items-center justify-between gap-2",
                selectedModel === m 
                  ? "bg-blue-600 text-white font-semibold shadow-md shadow-blue-200 translate-x-1" 
                  : "text-gray-600 hover:bg-gray-100 hover:text-gray-900"
              )}
            >
              <span className="truncate">{m}</span>
              {failed ? (
                <span
                  className={cn(
                    "shrink-0 inline-flex items-center justify-center w-5 h-5 rounded-full text-xs font-black",
                    selectedModel === m ? "bg-white/20 text-white" : "bg-amber-100 text-amber-700"
                  )}
                  title={tooltip}
                  aria-label={`Model folder validation failed: ${tooltip}`}
                >
                  !
                </span>
              ) : null}
            </button>
            );
          })}
        </div>
      </div>
  );
}
