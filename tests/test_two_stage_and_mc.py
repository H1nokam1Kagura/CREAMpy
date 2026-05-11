"""
Tests for:
  - BassParams.ceiling_series (time-varying ceiling)
  - TwoStagePipeline
  - Monte Carlo wrapper
"""

import dataclasses
import math
import pytest

from creampy import (
    ModelParams, ClosedEconomy,
    MCBassParams, MCModelParams, MCTwoStageParams, MCResult,
    run_bass_welfare_mc, run_two_stage_welfare_mc,
    sample,
)
from creampy.adoption import BassModel, BassParams
from creampy.adoption.pipeline import (
    TwoStageBassParams, TwoStagePipeline, TwoStagePipelineResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

YEARS = list(range(2025, 2046))

@pytest.fixture
def base_model():
    return ModelParams(
        K=0.12, epsilon=0.5, eta=-0.5, P0=200.0, Q0=1_000_000.0,
        discount_rate=0.05, base_year=2025,
    )

@pytest.fixture
def ts_params():
    return TwoStageBassParams(
        p_int=0.02, q_int=0.30, ceiling_int=0.60,
        p_con=0.005, q_con=0.35, ceiling_con=0.75,
        ptrs=1.0, t0=2025, years=YEARS,
    )

BASS_DISTS = MCBassParams(
    p       = {"dist": "beta",      "a": 2.0,  "b": 50.0},
    q       = {"dist": "beta",      "a": 5.0,  "b": 10.0},
    ceiling = {"dist": "beta",      "a": 5.0,  "b": 3.0},
    ptrs    = {"dist": "fixed",     "value": 0.80},
    t0=2025, years=YEARS,
)

WELFARE_DISTS = MCModelParams(
    K       = {"dist": "lognormal", "mu": -2.0, "sigma": 0.3},
    epsilon = {"dist": "fixed",     "value": 0.5},
    eta     = {"dist": "lognormal", "mu": -0.7, "sigma": 0.2, "negate": True},
    P0      = {"dist": "fixed",     "value": 200.0},
    Q0      = {"dist": "fixed",     "value": 1_000_000.0},
    discount_rate=0.05, base_year=2025,
)


# ── ceiling_series tests ──────────────────────────────────────────────────────

def test_ceiling_series_same_length_as_years():
    """ceiling_series with correct length validates without error."""
    p = BassParams(p=0.01, q=0.35, ceiling=0.70,
                   t0=2025, years=[2025, 2026, 2027],
                   ceiling_series=[0.20, 0.40, 0.60])
    BassModel(p)  # should not raise


def test_ceiling_series_wrong_length_raises():
    with pytest.raises(ValueError, match="length"):
        BassModel(BassParams(p=0.01, q=0.35, ceiling=0.70,
                             t0=2025, years=[2025, 2026, 2027],
                             ceiling_series=[0.20, 0.40]))  # 2 != 3


def test_ceiling_series_out_of_range_raises():
    with pytest.raises(ValueError, match="out-of-range"):
        BassModel(BassParams(p=0.01, q=0.35, ceiling=0.70,
                             t0=2025, years=[2025],
                             ceiling_series=[1.5]))  # > 1


def test_ceiling_series_zero_suppresses_adoption():
    """A zero in ceiling_series closes the gate for that year."""
    p = BassParams(p=0.05, q=0.40, ceiling=0.80,
                   t0=2025, years=[2025, 2026],
                   ceiling_series=[0.0, 0.50])
    result = BassModel(p).run()
    # Year 2025: gate closed → zero adoption
    assert result.year_results[0].new_frac == 0.0
    # Year 2026: gate open → some adoption
    assert result.year_results[1].new_frac > 0.0


def test_ceiling_series_growing_matches_manual():
    """Linearly growing ceiling_series produces monotone rising adoption."""
    series = [0.1 * i for i in range(1, 6)]  # 0.1, 0.2, 0.3, 0.4, 0.5
    p = BassParams(p=0.02, q=0.30, ceiling=0.50,
                   t0=2025, years=list(range(2025, 2030)),
                   ceiling_series=series)
    result = BassModel(p).run()
    cumuls = [r.cumul_frac for r in result.year_results]
    assert all(cumuls[i] >= cumuls[i-1] - 1e-10 for i in range(1, len(cumuls)))
    # Final cumul must not exceed max of ceiling_series
    assert cumuls[-1] <= max(series) + 1e-10


def test_no_ceiling_series_unchanged():
    """Without ceiling_series, BassModel behaves identically to before."""
    p = BassParams(p=0.01, q=0.40, ceiling=0.70, ptrs=1.0,
                   t0=2025, years=YEARS)
    r = BassModel(p).run()
    assert r.adoption_fracs[-1] > 0.0
    assert all(r.adoption_fracs[i] >= r.adoption_fracs[i-1] - 1e-10
               for i in range(1, len(r.adoption_fracs)))


# ── TwoStagePipeline tests ────────────────────────────────────────────────────

def test_two_stage_returns_correct_type(ts_params, base_model):
    result = TwoStagePipeline(ts_params, base_model).run()
    assert isinstance(result, TwoStagePipelineResult)


def test_two_stage_stage2_gated_by_stage1(ts_params, base_model):
    """Stage 2 adoption is always <= Stage 1 cumulative × ceiling_con."""
    result = TwoStagePipeline(ts_params, base_model).run()
    for s1, s2 in zip(result.stage1.year_results, result.stage2.year_results):
        max_possible = s1.cumul_frac * ts_params.ceiling_con
        assert s2.cumul_frac <= max_possible + 1e-9, (
            f"Year {s2.year}: stage2 cumul {s2.cumul_frac:.4f} > "
            f"stage1 × ceiling_con {max_possible:.4f}"
        )


def test_two_stage_stage2_lags_stage1(ts_params, base_model):
    """Stage 2 peak year must not precede Stage 1 peak year."""
    result = TwoStagePipeline(ts_params, base_model).run()
    assert result.stage2.peak_year is None or (
        result.stage1.peak_year is not None and
        result.stage2.peak_year >= result.stage1.peak_year
    )


def test_two_stage_welfare_positive(ts_params, base_model):
    result = TwoStagePipeline(ts_params, base_model).run()
    assert result.welfare.npv_W > 0
    assert result.welfare.npv_PS + result.welfare.npv_CS == pytest.approx(
        result.welfare.npv_W, rel=1e-9
    )


def test_two_stage_adoption_fracs_from_stage2(ts_params, base_model):
    result = TwoStagePipeline(ts_params, base_model).run()
    assert result.adoption_fracs == result.stage2.adoption_fracs


def test_two_stage_rejects_model_with_years(ts_params):
    bad_model = ModelParams(
        K=0.12, epsilon=0.5, eta=-0.5, P0=200.0, Q0=1e6,
        years=[2025, 2026], adoption_fracs=[0.1, 0.2],
    )
    with pytest.raises(ValueError, match="years"):
        TwoStagePipeline(ts_params, bad_model)


def test_two_stage_ptrs_scales_stage2(ts_params, base_model):
    """Halving ptrs halves Stage 2 adoption_fracs."""
    ts_half = dataclasses.replace(ts_params, ptrs=0.50)
    r_full = TwoStagePipeline(ts_params, base_model).run()
    r_half = TwoStagePipeline(ts_half, base_model).run()
    for full, half in zip(r_full.stage2.adoption_fracs,
                          r_half.stage2.adoption_fracs):
        assert half == pytest.approx(full * 0.5, rel=1e-9)


# ── Monte Carlo: sample() ─────────────────────────────────────────────────────

def test_sample_fixed():
    assert sample({"dist": "fixed", "value": 0.42}) == pytest.approx(0.42)


def test_sample_fixed_negated():
    v = sample({"dist": "fixed", "value": 0.5, "negate": True})
    assert v == pytest.approx(-0.5)


def test_sample_beta_in_range():
    for _ in range(100):
        v = sample({"dist": "beta", "a": 2.0, "b": 5.0})
        assert 0.0 <= v <= 1.0


def test_sample_lognormal_positive():
    for _ in range(100):
        v = sample({"dist": "lognormal", "mu": -1.0, "sigma": 0.5})
        assert v > 0.0


def test_sample_lognormal_negated_negative():
    for _ in range(100):
        v = sample({"dist": "lognormal", "mu": -0.7, "sigma": 0.2, "negate": True})
        assert v < 0.0


def test_sample_uniform_in_range():
    for _ in range(100):
        v = sample({"dist": "uniform", "low": 0.1, "high": 0.5})
        assert 0.1 <= v <= 0.5


def test_sample_unknown_dist_raises():
    with pytest.raises(ValueError, match="Unknown distribution"):
        sample({"dist": "cauchy", "x0": 0.0, "gamma": 1.0})


# ── Monte Carlo: MCResult ─────────────────────────────────────────────────────

def test_mc_result_percentiles():
    r = MCResult(n_samples=5, npv_W=[1.0, 2.0, 3.0, 4.0, 5.0],
                 npv_PS=[0.5]*5, npv_CS=[0.5]*5)
    assert r.p10() <= r.p50() <= r.p90()
    assert r.mean() == pytest.approx(3.0)


def test_mc_result_prob_exceeds():
    r = MCResult(n_samples=4, npv_W=[10.0, 20.0, 30.0, 40.0],
                 npv_PS=[5.0]*4, npv_CS=[5.0]*4)
    assert r.prob_exceeds(25.0) == pytest.approx(0.5)
    assert r.prob_exceeds(0.0) == pytest.approx(1.0)
    assert r.prob_exceeds(100.0) == pytest.approx(0.0)


def test_mc_result_summary_keys():
    r = MCResult(n_samples=10, npv_W=list(range(1, 11)),
                 npv_PS=[0.5]*10, npv_CS=[0.5]*10)
    s = r.summary()
    for key in ("n_samples", "n_failed", "mean", "std", "p10", "p50", "p90",
                "prob_positive"):
        assert key in s


# ── Monte Carlo: runner tests ─────────────────────────────────────────────────

def test_run_bass_welfare_mc_seed_reproducible():
    r1 = run_bass_welfare_mc(BASS_DISTS, WELFARE_DISTS, n=50, seed=7)
    r2 = run_bass_welfare_mc(BASS_DISTS, WELFARE_DISTS, n=50, seed=7)
    assert r1.npv_W == r2.npv_W


def test_run_bass_welfare_mc_sample_count():
    r = run_bass_welfare_mc(BASS_DISTS, WELFARE_DISTS, n=100, seed=1)
    assert r.n_samples == 100
    assert len(r.npv_W) + r.n_failed == 100


def test_run_bass_welfare_mc_positive_npv():
    r = run_bass_welfare_mc(BASS_DISTS, WELFARE_DISTS, n=200, seed=42)
    assert r.prob_exceeds(0.0) > 0.95   # reasonable params should mostly be positive


def test_run_bass_welfare_mc_p10_lt_p90():
    r = run_bass_welfare_mc(BASS_DISTS, WELFARE_DISTS, n=200, seed=42)
    assert r.p10() < r.p90()


def test_run_two_stage_welfare_mc_seed_reproducible(base_model):
    ts_dists = MCTwoStageParams(
        p_int       = {"dist": "beta", "a": 2.0, "b": 10.0},
        q_int       = {"dist": "beta", "a": 5.0, "b": 10.0},
        ceiling_int = {"dist": "beta", "a": 6.0, "b": 4.0},
        p_con       = {"dist": "beta", "a": 1.5, "b": 120.0},
        q_con       = {"dist": "beta", "a": 4.0, "b": 8.0},
        ceiling_con = {"dist": "beta", "a": 5.0, "b": 2.5},
        ptrs        = {"dist": "fixed", "value": 0.80},
        t0=2025, years=YEARS,
    )
    r1 = run_two_stage_welfare_mc(ts_dists, WELFARE_DISTS, n=50, seed=99)
    r2 = run_two_stage_welfare_mc(ts_dists, WELFARE_DISTS, n=50, seed=99)
    assert r1.npv_W == r2.npv_W


def test_run_two_stage_welfare_mc_npv_positive(base_model):
    ts_dists = MCTwoStageParams(
        p_int       = {"dist": "fixed", "value": 0.02},
        q_int       = {"dist": "fixed", "value": 0.30},
        ceiling_int = {"dist": "fixed", "value": 0.60},
        p_con       = {"dist": "fixed", "value": 0.005},
        q_con       = {"dist": "fixed", "value": 0.35},
        ceiling_con = {"dist": "fixed", "value": 0.75},
        ptrs        = {"dist": "fixed", "value": 1.0},
        t0=2025, years=YEARS,
    )
    r = run_two_stage_welfare_mc(ts_dists, WELFARE_DISTS, n=50, seed=1)
    assert r.prob_exceeds(0.0) > 0.90
