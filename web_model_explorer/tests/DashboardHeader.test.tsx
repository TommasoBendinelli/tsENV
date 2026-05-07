import { render, screen } from '@testing-library/react';
import { beforeEach, expect, test, vi } from 'vitest';
import { DashboardHeader } from '../components/dashboard/view/DashboardHeader';
import { DashboardControllerProvider } from '../components/dashboard/DashboardControllerContext';
import { useDashboardStore } from '../components/dashboard/useDashboardStore';

const controller = {
  handleReloadRegistryFromFile: vi.fn(),
};

beforeEach(() => {
  useDashboardStore.setState({
    selectedModel: '',
    loading: false,
    simulating: false,
    history: [],
    historyIndex: 0,
  });
});

test('write controls are not shown', () => {
  render(
    <DashboardControllerProvider controller={controller}>
      <DashboardHeader runsDirName="runs_7161" />
    </DashboardControllerProvider>
  );
  expect(screen.queryByRole('button', { name: /save/i })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /run simulations/i })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /cleanup orphans/i })).not.toBeInTheDocument();
});

test('Header shows the configured runs folder', () => {
  useDashboardStore.setState({ selectedModel: 'DampedMassBetweenWalls' });
  render(
    <DashboardControllerProvider controller={controller}>
      <DashboardHeader runsDirName="runs_7161" />
    </DashboardControllerProvider>
  );
  expect(screen.getByText(/runs folder:/i)).toBeInTheDocument();
  expect(screen.getByText('runs_7161')).toBeInTheDocument();
});
