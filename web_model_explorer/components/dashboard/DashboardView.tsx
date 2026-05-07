'use client';

import React from 'react';
import { DashboardHeader } from './view/DashboardHeader';
import { DashboardMain } from './view/DashboardMain';
import { DashboardOverlays } from './view/DashboardOverlays';
import { DashboardSidebar } from './view/DashboardSidebar';

export function DashboardView(props: { runsDirName: string }) {
  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden font-sans text-gray-900">
      <DashboardOverlays />
      <DashboardSidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <DashboardHeader runsDirName={props.runsDirName} />
        <DashboardMain />
      </div>
    </div>
  );
}
