import { render, screen, fireEvent } from '@testing-library/react';
import { beforeEach, expect, test, vi } from 'vitest';
import { DashboardSidebar } from '../components/dashboard/view/DashboardSidebar';
import { useDashboardStore } from '../components/dashboard/useDashboardStore';

beforeEach(() => {
  useDashboardStore.setState({
    models: ['ModelA', 'ModelB'],
    modelValidation: {},
    selectedModel: '',
  });
});

test('selecting a model updates the store', () => {
  render(<DashboardSidebar />);
  fireEvent.click(screen.getByRole('button', { name: 'ModelA' }));
  expect(useDashboardStore.getState().selectedModel).toBe('ModelA');
});

test('selecting a model with validation warnings does not show a popup', () => {
  const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});
  useDashboardStore.setState({
    modelValidation: {
      ModelA: {
        ok: false,
        reasons: ['Missing plans/<policy_id>/run_nodes.jsonl and run_edges.jsonl.'],
      },
    },
  });

  render(<DashboardSidebar />);
  fireEvent.click(screen.getByRole('button', { name: /ModelA/ }));

  expect(alertSpy).not.toHaveBeenCalled();
  expect(useDashboardStore.getState().selectedModel).toBe('ModelA');
  alertSpy.mockRestore();
});

test('selecting a model with only missing-file warnings does not show a popup', () => {
  const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});
  useDashboardStore.setState({
    modelValidation: {
      ModelA: {
        ok: false,
        reasons: ['Missing generated/metadata.json (run workflows/simulate/build_metadata.py to generate it).'],
      },
    },
  });

  render(<DashboardSidebar />);
  fireEvent.click(screen.getByRole('button', { name: /ModelA/ }));

  expect(alertSpy).not.toHaveBeenCalled();
  expect(useDashboardStore.getState().selectedModel).toBe('ModelA');
  alertSpy.mockRestore();
});
