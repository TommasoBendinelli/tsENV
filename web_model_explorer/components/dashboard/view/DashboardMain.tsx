'use client';

import React from 'react';
import { useDashboardStore } from '../useDashboardStore';
import { DashboardMainRuns } from './main/DashboardMainRuns';
import { DashboardMainWelcome } from './main/DashboardMainWelcome';

export function DashboardMain() {
  const selectedModel = useDashboardStore((state) => state.selectedModel);

  return (
    <main className="flex-1 overflow-y-auto p-6 space-y-8 bg-[#f8fafc]">
      {!selectedModel ? (
        <DashboardMainWelcome />
      ) : (
        <DashboardMainRuns />
      )}
    </main>
  );
}
