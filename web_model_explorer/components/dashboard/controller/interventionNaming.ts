import type { Intervention, RunRecord } from '../types';

export const normalizeInterventionVariable = (value: unknown) => {
  const raw = String(value ?? '').trim();
  if (!raw) return '';
  return raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+/, '')
    .replace(/_+$/, '');
};

export const parseVariableScopedInterventionId = (value: string) => {
  const trimmed = String(value || '').trim();
  const match = trimmed.match(/^(.*)_(\d+)$/);
  if (!match) return null;
  const base = String(match[1] || '').trim();
  const parsed = Number(match[2]);
  if (!base || !Number.isFinite(parsed)) return null;
  return { base, index: parsed };
};

export const createInterventionNameAllocator = (runs: RunRecord[], reserved?: Set<string>) => {
  const used = reserved ? new Set(reserved) : new Set<string>();
  const usedByBase = new Map<string, Set<number>>();

  const addUsedIndex = (base: string, index: number) => {
    const set = usedByBase.get(base) ?? new Set<number>();
    set.add(index);
    usedByBase.set(base, set);
  };

  for (const run of runs) {
    for (const iv of run.interventions) {
      const name = String(iv.name || '').trim();
      if (name) used.add(name);
      const parsed = parseVariableScopedInterventionId(name);
      if (!parsed) continue;
      addUsedIndex(parsed.base, parsed.index);
    }
  }

  const nextName = (variable: string) => {
    const base = normalizeInterventionVariable(variable) || 'param';
    const indices = usedByBase.get(base) ?? new Set<number>();
    let nextIndex = 0;
    while (indices.has(nextIndex) || used.has(`${base}_${String(nextIndex).padStart(3, '0')}`)) {
      nextIndex += 1;
    }
    const candidate = `${base}_${String(nextIndex).padStart(3, '0')}`;
    used.add(candidate);
    addUsedIndex(base, nextIndex);
    return candidate;
  };

  return { nextName, used };
};

export const buildNextInterventionName = (opts: {
  runs: RunRecord[];
  variable: string;
  reserved?: Set<string>;
}) => {
  const allocator = createInterventionNameAllocator(opts.runs, opts.reserved);
  return allocator.nextName(opts.variable);
};

export const buildInterventionDisplayNames = (run: RunRecord): Intervention[] => {
  return run.interventions.map((iv) => ({
    ...iv,
    display_name: String(iv.name || '').trim(),
  }));
};
