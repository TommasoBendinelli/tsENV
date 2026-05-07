import { describe, expect, test } from 'vitest';
import fixture from './fixtures/noise_snr_fixture.json';
import {
  applyAdaptiveNoise,
  applyAdaptiveAndBaseNoise,
  noiseMultiplierFromSNRDb,
  seedForRun,
  sampleAdaptiveAndBaseGaussianNoise,
  sampleAdaptiveGaussianNoise,
  snrDbFromNoiseMultiplier,
} from '@/components/dashboard/domains/noiseSNR';

describe('noiseSNR domain', () => {
  test('converts multiplier <-> SNR dB', () => {
    const mult = noiseMultiplierFromSNRDb(20);
    expect(mult).toBeCloseTo(0.1, 12);
    expect(snrDbFromNoiseMultiplier(mult)).toBeCloseTo(20, 12);
  });

  test('matches fixture for adaptive run/signal noise', () => {
    const values = fixture.input.values.map((v) => Number(v));
    const runNoise = sampleAdaptiveGaussianNoise(values, {
      noiseMultiplier: fixture.input.run_noise_multiplier,
      seedKey: fixture.input.run_seed_key,
      seed: fixture.input.seed,
      rules: fixture.rules,
    });
    const signalNoise = sampleAdaptiveGaussianNoise(values, {
      noiseMultiplier: fixture.input.signal_noise_multiplier,
      seedKey: fixture.input.signal_seed_key,
      seed: fixture.input.seed,
      rules: fixture.rules,
    });
    const noisy = applyAdaptiveNoise(values, {
      runNoiseMultiplier: fixture.input.run_noise_multiplier,
      signalNoiseMultiplier: fixture.input.signal_noise_multiplier,
      runSeedKey: fixture.input.run_seed_key,
      signalSeedKey: fixture.input.signal_seed_key,
      seed: fixture.input.seed,
      rules: fixture.rules,
    });

    expect(runNoise).toHaveLength(fixture.expected.run_noise.length);
    expect(signalNoise).toHaveLength(fixture.expected.signal_noise.length);
    expect(noisy).toHaveLength(fixture.expected.noisy_values.length);

    runNoise.forEach((v, i) => {
      expect(v).toBeCloseTo(fixture.expected.run_noise[i], 12);
    });
    signalNoise.forEach((v, i) => {
      expect(v).toBeCloseTo(fixture.expected.signal_noise[i], 12);
    });
    noisy.forEach((v, i) => {
      expect(v).not.toBeNull();
      expect(Number(v)).toBeCloseTo(fixture.expected.noisy_values[i], 12);
    });
  });

  test('matches fixture for adaptive + base noise', () => {
    const values = fixture.input.values.map((v) => Number(v));
    const adaptiveBaseNoise = sampleAdaptiveAndBaseGaussianNoise(values, {
      adaptiveNoiseMultiplier: fixture.input.adaptive_noise_multiplier,
      baseNoiseMultiplier: fixture.input.base_noise_multiplier,
      seedKey: fixture.input.adaptive_base_seed_key,
      seed: fixture.input.seed,
      rules: fixture.rules,
    });
    const noisy = applyAdaptiveAndBaseNoise(values, {
      adaptiveNoiseMultiplier: fixture.input.adaptive_noise_multiplier,
      baseNoiseMultiplier: fixture.input.base_noise_multiplier,
      seedKey: fixture.input.adaptive_base_seed_key,
      seed: fixture.input.seed,
      rules: fixture.rules,
    });

    expect(adaptiveBaseNoise).toHaveLength(fixture.expected.adaptive_base_noise.length);
    expect(noisy).toHaveLength(fixture.expected.adaptive_base_noisy_values.length);
    adaptiveBaseNoise.forEach((v, i) => {
      expect(v).toBeCloseTo(fixture.expected.adaptive_base_noise[i], 12);
    });
    noisy.forEach((v, i) => {
      expect(v).not.toBeNull();
      expect(Number(v)).toBeCloseTo(fixture.expected.adaptive_base_noisy_values[i], 12);
    });
  });

  test('matches fixture for CLI-style local/global seeded per run', () => {
    const values = fixture.input.values.map((v) => Number(v));
    const runId = String(fixture.input.cli_run_id);
    const signal = String(fixture.input.cli_signal);
    const derivedSeed = seedForRun(
      Number(fixture.input.cli_base_seed),
      runId,
    );
    expect(derivedSeed).toBe(Number(fixture.expected.cli_run_seed));
    const noisy = applyAdaptiveAndBaseNoise(values, {
      adaptiveNoiseMultiplier: Number(fixture.input.cli_local_noise_multiplier),
      baseNoiseMultiplier: Number(fixture.input.cli_global_noise_multiplier),
      seedKey: `signal:${runId}:${signal}`,
      seed: derivedSeed,
      rules: fixture.rules,
    });
    noisy.forEach((v, i) => {
      expect(v).not.toBeNull();
      expect(Number(v)).toBeCloseTo(fixture.expected.cli_noisy_values[i], 12);
    });
  });

  test('matches fixture for CLI-style local/global + abs seeded per run', () => {
    const values = fixture.input.values.map((v) => Number(v));
    const runId = String(fixture.input.cli_run_id);
    const signal = String(fixture.input.cli_signal);
    const derivedSeed = seedForRun(
      Number(fixture.input.cli_base_seed),
      runId,
    );
    const noisy = applyAdaptiveAndBaseNoise(values, {
      adaptiveNoiseMultiplier: Number(fixture.input.cli_local_noise_multiplier),
      baseNoiseMultiplier: Number(fixture.input.cli_global_noise_multiplier),
      absNoiseSigma: Number(fixture.input.cli_abs_sigma),
      seedKey: `signal:${runId}:${signal}`,
      seed: derivedSeed,
      rules: fixture.rules,
    });
    noisy.forEach((v, i) => {
      expect(v).not.toBeNull();
      expect(Number(v)).toBeCloseTo(fixture.expected.cli_noisy_values_with_abs[i], 12);
    });
  });
});
