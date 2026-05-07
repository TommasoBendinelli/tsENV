'use client';

import React from 'react';
import type { DashboardController } from './useDashboardController';

const DashboardControllerContext = React.createContext<DashboardController | null>(null);

export function DashboardControllerProvider(props: {
  controller: DashboardController;
  children: React.ReactNode;
}) {
  return (
    <DashboardControllerContext.Provider value={props.controller}>
      {props.children}
    </DashboardControllerContext.Provider>
  );
}

export function useDashboardControllerContext() {
  const controller = React.useContext(DashboardControllerContext);
  if (!controller) {
    throw new Error('useDashboardControllerContext must be used within DashboardControllerProvider');
  }
  return controller;
}

