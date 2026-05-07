import { describe, expect, test, vi } from 'vitest';
import RunShortcutPage from '../app/[run]/page';
import { buildRunShortcutRedirectPath } from '../app/runShortcut';
import { redirect } from 'next/navigation';

vi.mock('next/navigation', () => ({
  redirect: vi.fn(),
}));

describe('/{uuid} run shortcut route', () => {
  test('builds a run deep-link redirect target', () => {
    expect(buildRunShortcutRedirectPath('abc123')).toBe('/?run=abc123');
    expect(buildRunShortcutRedirectPath(' abc 123 ')).toBe('/?run=abc%20123');
  });

  test('redirects through existing run deep-link flow', () => {
    RunShortcutPage({ params: { run: 'abc123' } });
    expect(redirect).toHaveBeenCalledWith('/?run=abc123');
  });
});
