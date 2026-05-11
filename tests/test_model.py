"""
Pytest test suite for creampy.model.

Nine closed-form analytical cases with exact solutions; all independent of
external data. Run with: pytest tests/
"""

import dataclasses
import math
import pytest
from creampy import ClosedEconomy, ModelParams, k_from_yield_gain


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def base_params():
    return ModelParams(
        K=0.10, epsilon=0.5, eta=-0.5,
        P0=100.0, Q0=1_000_000.0,
        years=[2024], adoption_fracs=[1.0],
        discount_rate=0.05, base_year=2024,
    )


# ── Helper ────────────────────────────────────────────────────────────────────

def _w(K, eps, eta, P0, Q0, adpt, shift="K"):
    return ClosedEconomy._welfare(K, eps, eta, P0, Q0, adpt, shift)


# ── TC1-TC2: zero inputs ──────────────────────────────────────────────────────

def test_tc1_zero_K():
    dPS, dCS, dW = _w(0.0, 0.5, -0.5, 200.0, 1e6, 1.0)
    assert dPS == 0.0
    assert dCS == 0.0
    assert dW  == 0.0


def test_tc2_zero_adoption():
    _, _, dW = _w(0.15, 0.5, -0.5, 200.0, 1e6, 0.0)
    assert dW == 0.0


# ── TC3: symmetric elasticities ───────────────────────────────────────────────

def test_tc3_symmetric_split():
    dPS, dCS, dW = _w(0.15, 0.5, -0.5, 200.0, 1e6, 1.0)
    assert dPS == pytest.approx(dCS, rel=1e-9)


def test_tc3_dW_exact():
    K, eps, eta_abs, P0, Q0 = 0.15, 0.5, 0.5, 200.0, 1e6
    denom = eps + eta_abs
    Z = eps * eta_abs / denom
    expected = K * P0 * Q0 * (1.0 + 0.5 * K * Z)
    _, _, dW = _w(K, eps, -eta_abs, P0, Q0, 1.0)
    assert dW == pytest.approx(expected, rel=1e-9)


# ── TC4-TC5: K-shift welfare partition ───────────────────────────────────────

def test_tc4_kshift_inelastic_demand():
    """Inelastic demand (|η|<ε): producers get less than consumers."""
    dPS, dCS, _ = _w(0.01, 0.5, -0.2, 100.0, 1e6, 1.0, "K")
    assert dPS / dCS == pytest.approx(0.2 / 0.5, rel=1e-3)


def test_tc5_kshift_elastic_demand():
    """Elastic demand (|η|>ε): producers get more than consumers."""
    dPS, dCS, _ = _w(0.01, 0.5, -1.5, 100.0, 1e6, 1.0, "K")
    assert dPS / dCS == pytest.approx(1.5 / 0.5, rel=1e-3)


# ── TC6: J-shift reverses partition ──────────────────────────────────────────

def test_tc6_jshift_partition_reverses():
    """J-shift: ps_share = ε/(ε+|η|), cs_share = |η|/(ε+|η|)."""
    dPS, dCS, _ = _w(0.01, 0.5, -0.2, 100.0, 1e6, 1.0, "J")
    assert dPS / dCS == pytest.approx(0.5 / 0.2, rel=1e-3)


# ── TC7: single-year NPV ──────────────────────────────────────────────────────

def test_tc7_single_year_npv_equals_dW(base_params):
    result = ClosedEconomy(base_params).run()
    K, eps, eta_abs = 0.10, 0.5, 0.5
    denom = eps + eta_abs
    Z = eps * eta_abs / denom
    expected = K * 100.0 * 1e6 * (1.0 + 0.5 * K * Z)
    assert result.npv_W == pytest.approx(expected, rel=1e-9)
    assert result.npv_W == pytest.approx(result.npv_PS + result.npv_CS, rel=1e-12)


# ── TC8-TC9: linearity ────────────────────────────────────────────────────────

def test_tc8_linearity_Q0(base_params):
    r1 = ClosedEconomy(base_params).run()
    r2 = ClosedEconomy(dataclasses.replace(base_params, Q0=2e6)).run()
    assert r2.npv_W == pytest.approx(2.0 * r1.npv_W, rel=1e-9)


def test_tc9_linearity_P0(base_params):
    r1 = ClosedEconomy(base_params).run()
    r2 = ClosedEconomy(dataclasses.replace(base_params, P0=200.0)).run()
    assert r2.npv_W == pytest.approx(2.0 * r1.npv_W, rel=1e-9)


# ── Input validation ──────────────────────────────────────────────────────────

def test_invalid_epsilon_raises():
    with pytest.raises(ValueError, match="epsilon"):
        ClosedEconomy(ModelParams(K=0.1, epsilon=-0.5, eta=-0.5, P0=100, Q0=1e6,
                                  years=[2024], adoption_fracs=[1.0]))


def test_invalid_eta_raises():
    with pytest.raises(ValueError, match="eta"):
        ClosedEconomy(ModelParams(K=0.1, epsilon=0.5, eta=0.5, P0=100, Q0=1e6,
                                  years=[2024], adoption_fracs=[1.0]))


def test_mismatched_years_raises():
    with pytest.raises(ValueError, match="same length"):
        ClosedEconomy(ModelParams(K=0.1, epsilon=0.5, eta=-0.5, P0=100, Q0=1e6,
                                  years=[2024, 2025], adoption_fracs=[1.0]))


# ── k_from_yield_gain helper ──────────────────────────────────────────────────

def test_k_from_yield_gain_15pct():
    K = k_from_yield_gain(0.15)
    assert K == pytest.approx(0.15 / 1.15, rel=1e-9)


def test_k_from_yield_gain_zero():
    assert k_from_yield_gain(0.0) == 0.0


def test_k_from_yield_gain_negative_raises():
    with pytest.raises(ValueError):
        k_from_yield_gain(-0.1)
