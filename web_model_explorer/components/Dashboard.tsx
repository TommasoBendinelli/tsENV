'use client';

import { DashboardView } from './dashboard/DashboardView';
import { useDashboardController } from './dashboard/useDashboardController';
import { DashboardControllerProvider } from '@/components/dashboard/DashboardControllerContext';

export default function Dashboard(props: { runsDirName: string }) {
  const controller = useDashboardController();
  return (
    <DashboardControllerProvider controller={controller}>
      <DashboardView runsDirName={props.runsDirName} />
    </DashboardControllerProvider>
  );
}
