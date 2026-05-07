'use client';

import React from 'react';
import { Activity } from 'lucide-react';

export function DashboardMainWelcome() {
  return (
    <div className="h-full flex flex-col items-center justify-center text-gray-400">
      <div className="bg-white p-8 rounded-3xl border shadow-sm flex flex-col items-center border-dashed">
        <Activity size={64} className="mb-4 text-blue-200" />
        <p className="text-lg font-medium text-gray-500">Welcome to Baseline Model Visualiser</p>
        <p className="text-sm text-gray-400">Select a model from the explorer to manage simulations</p>
      </div>
    </div>
  );
}
