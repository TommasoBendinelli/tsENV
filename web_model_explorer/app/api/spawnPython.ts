import { spawn } from 'child_process';
import type { ChildProcessWithoutNullStreams, SpawnOptions } from 'child_process';

export function spawnPython(
  command: string,
  args: string[],
  options: SpawnOptions
): ChildProcessWithoutNullStreams {
  // Ensure stdout/stderr are always streams so callers can safely capture output.
  const proc = spawn(command, args, {
    ...options,
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  return proc as unknown as ChildProcessWithoutNullStreams;
}
