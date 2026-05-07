'use client';

import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const toSentenceCase = (value: string) => {
  const normalized = value
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/[_\\-]+/g, ' ');
  const parts = normalized.match(/[A-Za-z0-9]+/g) || [];
  if (parts.length === 0) return '';
  const lower = parts.map(part => part.toLowerCase());
  lower[0] = lower[0].charAt(0).toUpperCase() + lower[0].slice(1);
  return lower.join(' ');
};
