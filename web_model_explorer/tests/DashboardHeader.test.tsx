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
      <DashboardHeader runsDirName="custom_runs" />
    </DashboardControllerProvider>
  );
  expect(screen.queryByRole('button', { name: /save/i })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /run simulations/i })).not.toBeInTheDocument();
  expect(screen.queryByRole('button', { name: /cleanup orphans/i })).not.toBeInTheDocument();
});

test('Header shows the configured runs folder and selected policy', () => {
  useDashboardStore.setState({
    selectedModel: 'DampedMassBetweenWalls',
    policies: ['policy_a'],
    selectedPolicy: 'policy_a',
  });
  render(
    <DashboardControllerProvider controller={controller}>
      <DashboardHeader runsDirName="custom_runs" />
    </DashboardControllerProvider>
  );
  expect(screen.getByText(/runs folder:/i)).toBeInTheDocument();
  expect(screen.getByText('custom_runs')).toBeInTheDocument();
  expect(screen.getByDisplayValue('policy_a')).toBeInTheDocument();
});
