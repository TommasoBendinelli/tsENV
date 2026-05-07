import { describe, expect, test } from 'vitest';
import {
  stripRunQueryParamFromUrl,
  collectDeepLinkKnownRunIds,
  parseDeepLinkFromSearch,
  shouldExpandBaselineForDeepLink,
} from '../components/dashboard/controller/deepLink';

describe('shouldExpandBaselineForDeepLink', () => {
  test('expands when deep-linked run is an intervention under a baseline', () => {
    expect(
      shouldExpandBaselineForDeepLink({
        baselineId: 'base_1',
        rawRunId: 'iv_1',
        comparatorId: 'base_1',
      })
    ).toBe(true);
  });

  test('expands when deep-linked run is a baseline and comparator is an intervention', () => {
    expect(
      shouldExpandBaselineForDeepLink({
        baselineId: 'base_1',
        rawRunId: 'base_1',
        comparatorId: 'iv_1',
      })
    ).toBe(true);
  });

  test('does not expand when deep-linked run and comparator are the same baseline', () => {
    expect(
      shouldExpandBaselineForDeepLink({
        baselineId: 'base_1',
        rawRunId: 'base_1',
        comparatorId: 'base_1',
      })
    ).toBe(false);
  });

  test('expands run-only deep link when baseline id is known and comparator is absent', () => {
    expect(
      shouldExpandBaselineForDeepLink({
        baselineId: 'base_1',
        rawRunId: 'iv_1',
        comparatorId: null,
      })
    ).toBe(true);
  });

  test('collects only registry-linked ids (baseline, child, and time0)', () => {
    const known = collectDeepLinkKnownRunIds([
      {
        run_id: 'base_1',
        interventions: [
          { name: 'iv_1', time0_baseline_uuid: 't0_1' },
        ],
      },
    ] as any);
    expect(known.has('base_1')).toBe(true);
    expect(known.has('iv_1')).toBe(true);
    expect(known.has('t0_1')).toBe(true);
    expect(known.has('orphan_disk_only')).toBe(false);
  });

  test('strips stale deep-link run query while preserving other params', () => {
    const next = stripRunQueryParamFromUrl(
      'http://localhost:3002/?model=DampedMassBetweenWalls&run=80f606b233975a5d0634e4ff2a127752&run2=2e18838f7942037306f7fdd993ae0ca6&compare=baseline'
    );
    expect(next).toBe('/?model=DampedMassBetweenWalls&compare=baseline');
  });

  test('strips legacy compare_run deep-link query while preserving other params', () => {
    const next = stripRunQueryParamFromUrl(
      'http://localhost:3002/?model=DampedMassBetweenWalls&run=80f606b233975a5d0634e4ff2a127752&compare_run=2e18838f7942037306f7fdd993ae0ca6&compare=none'
    );
    expect(next).toBe('/?model=DampedMassBetweenWalls&compare=none');
  });

  test('strips run query and keeps hash if run is the only query param', () => {
    const next = stripRunQueryParamFromUrl(
      'http://localhost:3002/?run=80f606b233975a5d0634e4ff2a127752#plot'
    );
    expect(next).toBe('/#plot');
  });

  test('parses run-only deep link as single-run mode (compare none)', () => {
    const parsed = parseDeepLinkFromSearch(
      '?model=BallDrop&run=39711d03a0eedb1f48101ab254dd3597'
    );
    expect(parsed.rawModelId).toBe('BallDrop');
    expect(parsed.rawRunId).toBe('39711d03a0eedb1f48101ab254dd3597');
    expect(parsed.rawComparatorRunId).toBe(null);
    expect(parsed.compareMode).toBe('none');
    expect(parsed.noiseProfile).toBe(null);
    expect(parsed.noiseSeed).toBe(null);
  });

  test('parses profile-based deep-link noise parameters', () => {
    const parsed = parseDeepLinkFromSearch(
      '?model=BallDrop&run=39711d03a0eedb1f48101ab254dd3597&noise_profile=low&noise_seed=7'
    );
    expect(parsed.noiseProfile).toBe('low');
    expect(parsed.noiseSeed).toBe(7);
  });

  test('ignores invalid deep-link noise profile', () => {
    const parsed = parseDeepLinkFromSearch(
      '?model=BallDrop&run=39711d03a0eedb1f48101ab254dd3597&noise_profile=wild&noise_seed=7'
    );
    expect(parsed.noiseProfile).toBe(null);
    expect(parsed.noiseSeed).toBe(7);
  });

  test('ignores undocumented deep-link noise parameters but keeps seed', () => {
    const parsed = parseDeepLinkFromSearch(
      '?model=BallDrop&run=39711d03a0eedb1f48101ab254dd3597&noise_local=0.01&noise_global=0.02&noise_abs=0.003&noise_seed=7'
    );
    expect(parsed.noiseProfile).toBe(null);
    expect('noiseAbs' in parsed).toBe(false);
    expect(parsed.noiseSeed).toBe(7);
  });
});
