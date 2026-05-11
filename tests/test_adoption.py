"""
Pytest test suite for creampy.adoption (Bass diffusion model + pipeline).

All test cases have exact or analytically verifiable expected values.
Run with: pytest tests/
"""

import dataclasses
import math
import pytest

from creampy.adoption import BassModel, BassParams
from creampy.adoption.pipeline import Pipeline, to_dreampy_table, to_dreampy_csv
from creampy import ModelParams


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def standard_bass():
    """Canonical parameters: SSA crop with moderate peer learning."""
    return BassParams(
        p=0.01, q=0.40, ceiling=0.70, ptrs=1.0,
        t0=2025, years=list(range(2025, 2041)),
    )


@pytest.fixture
def standard_model():
    """ModelParams without years/adoption_fracs for Pipeline use."""
    return ModelParams(
        K=0.13, epsilon=0.5, eta=-0.5,
        P0=200.0, Q0=1_000_000.0,
        discount_rate=0.05, base_year=2025,
    )


# ── TC-B1: Pure innovation (q=0) — analytical closed-form ─────────────────────

def test_tcb1_pure_innovation_analytical():
    """
    With q=0 (no imitation) and ceiling=1, the Bass recurrence simplifies to:
      A(t) = 1 - (1 - p)^t    (geometric convergence)

    Verify year-by-year against the closed-form.
    """
    p, ceiling = 0.05, 1.0
    par = BassParams(p=p, q=0.0, ceiling=ceiling, ptrs=1.0,
                     t0=2025, years=list(range(2025, 2036)))
    result = BassModel(par).run()

    for i, r in enumerate(result.year_results):
        expected = ceiling * (1.0 - (1.0 - p) ** (i + 1))
        assert r.cumul_frac == pytest.approx(expected, rel=1e-9), (
            f"Year {r.year}: got {r.cumul_frac:.8f}, expected {expected:.8f}"
        )


# ── TC-B2: Ceiling enforcement ────────────────────────────────────────────────

def test_tcb2_never_exceeds_ceiling():
    """Cumulative adoption must never exceed ceiling, even at saturation."""
    par = BassParams(p=0.10, q=0.60, ceiling=0.65, ptrs=1.0,
                     t0=2020, years=list(range(2020, 2060)))
    result = BassModel(par).run()
    for r in result.year_results:
        assert r.cumul_frac <= par.ceiling + 1e-12, (
            f"Year {r.year}: cumul_frac={r.cumul_frac} > ceiling={par.ceiling}"
        )


# ── TC-B3: Monotonicity ───────────────────────────────────────────────────────

def test_tcb3_monotone_non_decreasing(standard_bass):
    """Cumulative adoption is monotonically non-decreasing."""
    result = BassModel(standard_bass).run()
    fracs = [r.cumul_frac for r in result.year_results]
    for i in range(1, len(fracs)):
        assert fracs[i] >= fracs[i - 1] - 1e-12, (
            f"Adoption decreased at index {i}: {fracs[i-1]} -> {fracs[i]}"
        )


# ── TC-B4: ptrs scales adoption ───────────────────────────────────────────────

def test_tcb4_ptrs_scales_proportionally():
    """risk_adj_frac == cumul_frac * ptrs for every year."""
    par = BassParams(p=0.01, q=0.35, ceiling=0.80, ptrs=0.75,
                     t0=2025, years=list(range(2025, 2041)))
    result = BassModel(par).run()
    for r in result.year_results:
        assert r.risk_adj_frac == pytest.approx(r.cumul_frac * par.ptrs, rel=1e-9)


# ── TC-B5: No adoption before launch year ────────────────────────────────────

def test_tcb5_zero_before_launch():
    """Years before t0 produce zero adoption fractions."""
    par = BassParams(p=0.01, q=0.40, ceiling=0.70, ptrs=0.80,
                     t0=2028, years=list(range(2025, 2041)))
    result = BassModel(par).run()
    pre_launch = [r for r in result.year_results if r.year < par.t0]
    assert len(pre_launch) == 3
    for r in pre_launch:
        assert r.cumul_frac == 0.0
        assert r.new_frac   == 0.0
        assert r.risk_adj_frac == 0.0


# ── TC-B6: First year adoption equals p * ceiling (when q=0 implicitly) ───────

def test_tcb6_first_year_value():
    """
    In the first year after launch, A(t0) = (p + q * 0) * (ceiling - 0) = p * ceiling.
    This holds regardless of q (since A(t0-1)=0).
    """
    par = BassParams(p=0.02, q=0.45, ceiling=0.80, ptrs=1.0,
                     t0=2025, years=[2025])
    result = BassModel(par).run()
    expected_first = par.p * par.ceiling
    assert result.year_results[0].cumul_frac == pytest.approx(expected_first, rel=1e-9)


# ── TC-B7: Peak timing approximation ─────────────────────────────────────────

def test_tcb7_peak_timing_continuous_approx():
    """
    Discrete peak year should be within 1 year of the continuous approximation
    t* = ln(q/p)/(p+q).
    """
    p, q = 0.01, 0.40
    par = BassParams(p=p, q=q, ceiling=1.0, ptrs=1.0,
                     t0=2025, years=list(range(2025, 2060)))
    result = BassModel(par).run()
    continuous_t_star = math.log(q / p) / (p + q)
    continuous_peak_year = round(par.t0 + continuous_t_star)
    assert abs(result.peak_year - continuous_peak_year) <= 1, (
        f"Peak year {result.peak_year} differs from continuous approx "
        f"{continuous_peak_year} by more than 1 year"
    )


# ── TC-B8: Saturation — adoption_fracs convenience property ──────────────────

def test_tcb8_adoption_fracs_matches_risk_adj():
    """BassResult.adoption_fracs returns the same as risk_adj_frac list."""
    par = BassParams(p=0.01, q=0.35, ceiling=0.70, ptrs=0.85,
                     t0=2025, years=list(range(2025, 2041)))
    result = BassModel(par).run()
    assert result.adoption_fracs == [r.risk_adj_frac for r in result.year_results]


# ── TC-B9: Input validation ───────────────────────────────────────────────────

def test_tcb9_invalid_p_zero_raises():
    with pytest.raises(ValueError, match="p must be in"):
        BassModel(BassParams(p=0.0, q=0.3, ceiling=0.7, ptrs=1.0,
                             t0=2025, years=[2025]))


def test_tcb9_invalid_ptrs_raises():
    with pytest.raises(ValueError, match="ptrs"):
        BassModel(BassParams(p=0.01, q=0.3, ceiling=0.7, ptrs=0.0,
                             t0=2025, years=[2025]))


def test_tcb9_invalid_ceiling_raises():
    with pytest.raises(ValueError, match="ceiling"):
        BassModel(BassParams(p=0.01, q=0.3, ceiling=1.1, ptrs=1.0,
                             t0=2025, years=[2025]))


# ── Pipeline tests ────────────────────────────────────────────────────────────

def test_pipeline_years_match_bass(standard_bass, standard_model):
    """Pipeline welfare years must match Bass years."""
    result = Pipeline(standard_bass, standard_model).run()
    assert result.welfare.params.years == result.bass.years


def test_pipeline_adoption_injected_correctly(standard_bass, standard_model):
    """Pipeline must inject Bass risk-adjusted adoption into welfare model."""
    result = Pipeline(standard_bass, standard_model).run()
    assert result.welfare.params.adoption_fracs == result.bass.adoption_fracs


def test_pipeline_npv_positive(standard_bass, standard_model):
    """With positive K and non-zero adoption, NPV_W must be positive."""
    result = Pipeline(standard_bass, standard_model).run()
    assert result.welfare.npv_W > 0
    assert result.welfare.npv_PS > 0
    assert result.welfare.npv_CS > 0


def test_pipeline_rejects_years_in_model(standard_bass):
    """Pipeline must reject ModelParams that already have years set."""
    bad_model = ModelParams(
        K=0.13, epsilon=0.5, eta=-0.5, P0=200.0, Q0=1e6,
        years=[2025, 2026], adoption_fracs=[0.1, 0.2],
    )
    with pytest.raises(ValueError, match="years"):
        Pipeline(standard_bass, bad_model)


def test_pipeline_ptrs_zero_gives_zero_welfare(standard_model):
    """ptrs=1e-10 (effectively zero) should give near-zero welfare."""
    # We can't use ptrs=0.0 (validation rejects it), so use a near-zero value
    par = BassParams(p=0.01, q=0.40, ceiling=0.70, ptrs=1e-10,
                     t0=2025, years=list(range(2025, 2041)))
    result = Pipeline(par, standard_model).run()
    assert result.welfare.npv_W == pytest.approx(0.0, abs=1.0)


# ── DREAMpy export tests ──────────────────────────────────────────────────────

def test_to_dreampy_table_keys_match_years(standard_bass):
    """to_dreampy_table must return one entry per year."""
    bass_result = BassModel(standard_bass).run()
    table = to_dreampy_table(bass_result)
    assert list(table.keys()) == standard_bass.years


def test_to_dreampy_table_risk_adjusted_default(standard_bass):
    """Default (risk_adjusted=True) returns risk_adj_frac values."""
    par = dataclasses.replace(standard_bass, ptrs=0.75)
    bass_result = BassModel(par).run()
    table = to_dreampy_table(bass_result)
    for r in bass_result.year_results:
        assert table[r.year] == pytest.approx(r.risk_adj_frac, rel=1e-9)


def test_to_dreampy_table_raw(standard_bass):
    """risk_adjusted=False returns unadjusted cumulative fracs."""
    bass_result = BassModel(standard_bass).run()
    table = to_dreampy_table(bass_result, risk_adjusted=False)
    for r in bass_result.year_results:
        assert table[r.year] == pytest.approx(r.cumul_frac, rel=1e-9)


def test_to_dreampy_csv_creates_file(standard_bass, tmp_path):
    """to_dreampy_csv must create a readable CSV with correct columns."""
    import csv as csv_mod
    bass_result = BassModel(standard_bass).run()
    out = tmp_path / "adoption.csv"
    to_dreampy_csv(bass_result, str(out))
    assert out.exists()
    rows = [r for r in csv_mod.DictReader(
        line for line in out.read_text().splitlines() if not line.startswith("#")
    )]
    assert len(rows) == len(standard_bass.years)
    assert "year" in rows[0]
    assert "adoption_raw" in rows[0]
    assert "adoption_risk_adjusted" in rows[0]
