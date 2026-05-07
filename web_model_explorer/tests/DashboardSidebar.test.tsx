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

test('selecting a model with stale model_run_specs shows a popup', () => {
  const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});
  useDashboardStore.setState({
    modelValidation: {
      ModelA: {
        ok: false,
        reasons: [
          "model_run_specs.json appears stale: baseline 'abc' baseline_parameters_hash does not match current baseline_parameters.",
        ],
      },
    },
  });

  render(<DashboardSidebar />);
  fireEvent.click(screen.getByRole('button', { name: /ModelA/ }));

  expect(alertSpy).toHaveBeenCalledWith(expect.stringContaining('model_run_specs.json appears stale'));
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
