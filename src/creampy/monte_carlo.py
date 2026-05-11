"""
CREAMpy Monte Carlo wrapper.
=============================

Converts any deterministic CREAMpy model into a probabilistic one by sampling
parameters from specified distributions and aggregating the resulting NPV outputs.

Zero runtime dependencies — uses Python's stdlib ``random`` module throughout.

Distribution specification format
----------------------------------
Each uncertain parameter is described by a plain dict:

    {"dist": "beta",      "a": 2.0,  "b": 5.0}
        Beta(a, b)  —  support [0, 1].
        Use for fractions: p, q, ceiling, ptrs.

    {"dist": "lognormal", "mu": -2.0, "sigma": 0.35}
        LogNormal(mu_log, sigma_log)  —  always positive.
        Use for K, P0, Q0, or any positive unbounded parameter.

    {"dist": "normal",    "mu": 0.5,  "sigma": 0.1}
        Normal(mu, sigma)  —  unbounded; may go negative.
        Use for elasticities (eta is naturally negative).

    {"dist": "uniform",   "low": 0.1, "high": 0.5}
        Uniform[low, high].
        Use when only bounds are known.

    {"dist": "fixed",     "value": 0.5}
        Constant — no sampling.  Use for base_year, discount_rate, etc.

Add ``"negate": true`` to any spec to flip the sign of the sample — useful
for ``eta`` (demand elasticity) which must be negative but is most naturally
described as a positive lognormal:

    {"dist": "lognormal", "mu": -0.7, "sigma": 0.3, "negate": True}  # for eta

Maturity-to-sigma mapping
--------------------------
Calibrate distribution width to the L0–L4 maturity level of each parameter.
A rule of thumb for portfolio-level work:

    L0 — very wide prior (no evidence):    sigma/central ≈ 0.50
    L1 — document-derived:                 sigma/central ≈ 0.30
    L2 — expert-asserted (single session): sigma/central ≈ 0.20
    L3 — corroborated (second source):     sigma/central ≈ 0.10
    L4 — verified, multi-source:           sigma/central ≈ 0.05

Usage — Bass + welfare
-----------------------
    from creampy.monte_carlo import (
        MCBassParams, MCModelParams, MCResult, run_bass_welfare_mc
    )

    bass_dists = MCBassParams(
        p       = {"dist": "beta",      "a": 1.5,  "b": 150.0},
        q       = {"dist": "beta",      "a": 6.0,  "b": 12.0},
        ceiling = {"dist": "beta",      "a": 5.0,  "b": 3.0},
        ptrs    = {"dist": "beta",      "a": 3.0,  "b": 1.5},
        t0=2025, years=list(range(2025, 2046)),
    )
    welfare_dists = MCModelParams(
        K       = {"dist": "lognormal", "mu": -2.1, "sigma": 0.35},
        epsilon = {"dist": "fixed",     "value": 0.5},
        eta     = {"dist": "lognormal", "mu": -0.7, "sigma": 0.25, "negate": True},
        P0      = {"dist": "fixed",     "value": 200.0},
        Q0      = {"dist": "lognormal", "mu": 13.8,  "sigma": 0.20},
    )
    result = run_bass_welfare_mc(bass_dists, welfare_dists, n=2000, seed=42)
    print(f"NPV_W  p10={result.p10():>14,.0f}  p50={result.p50():>14,.0f}  "
          f"p90={result.p90():>14,.0f}")
    print(f"Prob NPV > 100M: {result.prob_exceeds(100_000_000):.1%}")

Usage — two-stage Bass + welfare
----------------------------------
    from creampy.monte_carlo import MCTwoStageParams, run_two_stage_welfare_mc

    ts_dists = MCTwoStageParams(
        p_int       = {"dist": "beta",      "a": 2.0,  "b": 10.0},
        q_int       = {"dist": "beta",      "a": 5.0,  "b": 10.0},
        ceiling_int = {"dist": "beta",      "a": 6.0,  "b": 4.0},
        p_con       = {"dist": "beta",      "a": 1.2,  "b": 120.0},
        q_con       = {"dist": "beta",      "a": 4.0,  "b": 8.0},
        ceiling_con = {"dist": "beta",      "a": 5.0,  "b": 2.5},
        ptrs        = {"dist": "beta",      "a": 3.0,  "b": 1.5},
        t0=2025, years=list(range(2025, 2046)),
    )
    result = run_two_stage_welfare_mc(ts_dists, welfare_dists, n=2000, seed=42)
"""

from __future__ import annotations

import dataclasses
import math
import random
import statistics

from .model import ClosedEconomy, ModelParams
from .adoption.bass import BassModel, BassParams
from .adoption.pipeline import TwoStageBassParams, TwoStagePipeline
from .adoption.network_platform import NetworkPlatformModel, NetworkPlatformParams

__all__ = [
    # Distribution sampling
    "sample",
    # Parameter distribution containers
    "MCBassParams",
    "MCModelParams",
    "MCTwoStageParams",
    "MCNetworkPlatformParams",
    # Result container
    "MCResult",
    # Runners
    "run_bass_welfare_mc",
    "run_two_stage_welfare_mc",
    "run_platform_welfare_mc",
]


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample(spec: dict) -> float:
    """Draw one sample from a distribution specification dict.

    Supported ``dist`` values and required keys:

        beta       -- a, b
        lognormal  -- mu, sigma  (mu and sigma are in log-space)
        normal     -- mu, sigma
        uniform    -- low, high
        fixed      -- value      (no randomness; returns value unchanged)

    Optional key ``negate`` (bool, default False): if True the returned
    value is negated.  Use for parameters that must be negative (e.g. eta).

    Raises ValueError for unknown distribution names.
    """
    dist = spec.get("dist")
    if dist == "beta":
        v = random.betavariate(spec["a"], spec["b"])
    elif dist == "lognormal":
        v = math.exp(random.gauss(spec["mu"], spec["sigma"]))
    elif dist == "normal":
        v = random.gauss(spec["mu"], spec["sigma"])
    elif dist == "uniform":
        v = random.uniform(spec["low"], spec["high"])
    elif dist == "fixed":
        v = float(spec["value"])
    else:
        raise ValueError(
            f"Unknown distribution '{dist}'. "
            "Use: beta, lognormal, normal, uniform, fixed."
        )
    return -v if spec.get("negate") else v


# ── Parameter distribution containers ────────────────────────────────────────

@dataclasses.dataclass
class MCBassParams:
    """Distribution specifications for a single Bass diffusion run.

    All fields except ``t0`` and ``years`` are distribution dicts
    (see module docstring for format).
    """
    p:       dict
    q:       dict
    ceiling: dict
    ptrs:    dict
    t0:      int
    years:   list[int]

    def draw(self) -> BassParams:
        """Sample one concrete BassParams from the distributions."""
        return BassParams(
            p       = sample(self.p),
            q       = sample(self.q),
            ceiling = sample(self.ceiling),
            ptrs    = sample(self.ptrs),
            t0      = self.t0,
            years   = self.years,
        )


@dataclasses.dataclass
class MCModelParams:
    """Distribution specifications for ClosedEconomy welfare parameters.

    ``discount_rate``, ``base_year``, ``scenario``, and ``shift_type`` are
    fixed at construction time — they are structural, not stochastic.
    """
    K:             dict
    epsilon:       dict
    eta:           dict   # use "negate": True with a positive-valued dist
    P0:            dict
    Q0:            dict
    discount_rate: float = 0.05
    base_year:     int   = 2025
    scenario:      str   = "central"
    shift_type:    str   = "K"

    def draw(self, years: list[int], adoption_fracs: list[float]) -> ModelParams:
        """Sample one concrete ModelParams, injecting the supplied adoption schedule."""
        return ModelParams(
            K             = sample(self.K),
            epsilon       = sample(self.epsilon),
            eta           = sample(self.eta),
            P0            = sample(self.P0),
            Q0            = sample(self.Q0),
            years         = years,
            adoption_fracs= adoption_fracs,
            discount_rate = self.discount_rate,
            base_year     = self.base_year,
            scenario      = self.scenario,
            shift_type    = self.shift_type,
        )


@dataclasses.dataclass
class MCTwoStageParams:
    """Distribution specifications for a two-stage Bass pipeline."""
    p_int:       dict
    q_int:       dict
    ceiling_int: dict
    p_con:       dict
    q_con:       dict
    ceiling_con: dict
    ptrs:        dict
    t0:          int
    years:       list[int]

    def draw(self) -> TwoStageBassParams:
        """Sample one concrete TwoStageBassParams."""
        return TwoStageBassParams(
            p_int       = sample(self.p_int),
            q_int       = sample(self.q_int),
            ceiling_int = sample(self.ceiling_int),
            p_con       = sample(self.p_con),
            q_con       = sample(self.q_con),
            ceiling_con = sample(self.ceiling_con),
            ptrs        = sample(self.ptrs),
            t0          = self.t0,
            years       = self.years,
        )


@dataclasses.dataclass
class MCNetworkPlatformParams:
    """Distribution specifications for a NetworkPlatformModel run."""
    p_p:      dict
    q_p:      dict
    M_p:      dict
    sigma:    dict
    lambda_q: dict
    S_crit:   dict
    p_f:      dict
    q_f:      dict
    M_f_max:  dict
    ptrs:     dict
    t0:       int
    years:    list[int]
    N_p0:     float = 0.0

    def draw(self) -> NetworkPlatformParams:
        """Sample one concrete NetworkPlatformParams."""
        return NetworkPlatformParams(
            p_p      = sample(self.p_p),
            q_p      = sample(self.q_p),
            M_p      = sample(self.M_p),
            sigma    = sample(self.sigma),
            lambda_q = sample(self.lambda_q),
            S_crit   = sample(self.S_crit),
            p_f      = sample(self.p_f),
            q_f      = sample(self.q_f),
            M_f_max  = sample(self.M_f_max),
            ptrs     = sample(self.ptrs),
            t0       = self.t0,
            years    = self.years,
            N_p0     = self.N_p0,
        )


# ── Result container ──────────────────────────────────────────────────────────

@dataclasses.dataclass
class MCResult:
    """Monte Carlo output across n_samples runs.

    All NPV values are in the same currency/unit as P0 × Q0
    (typically USD, matching ModelParams inputs).
    """
    n_samples:   int
    npv_W:       list[float]
    npv_PS:      list[float]
    npv_CS:      list[float]
    n_failed:    int = 0   # samples rejected due to invalid parameter draws

    def _sorted_W(self) -> list[float]:
        return sorted(self.npv_W)

    def percentile(self, p: float) -> float:
        """NPV_W at the p-th percentile (p in [0, 1])."""
        s = self._sorted_W()
        return s[max(0, min(len(s) - 1, int(p * len(s))))]

    def p10(self) -> float:
        """NPV_W 10th percentile (pessimistic scenario)."""
        return self.percentile(0.10)

    def p50(self) -> float:
        """NPV_W median."""
        return self.percentile(0.50)

    def p90(self) -> float:
        """NPV_W 90th percentile (optimistic scenario)."""
        return self.percentile(0.90)

    def mean(self) -> float:
        """Mean NPV_W."""
        return statistics.mean(self.npv_W) if self.npv_W else 0.0

    def std(self) -> float:
        """Standard deviation of NPV_W."""
        return statistics.stdev(self.npv_W) if len(self.npv_W) > 1 else 0.0

    def prob_exceeds(self, threshold: float) -> float:
        """Fraction of samples where NPV_W > threshold."""
        if not self.npv_W:
            return 0.0
        return sum(1 for v in self.npv_W if v > threshold) / len(self.npv_W)

    def summary(self) -> dict:
        """Key statistics as a plain dict — suitable for JSON serialisation."""
        w = self._sorted_W()   # sort once; reused for all three percentiles
        n = len(w)

        def pct(p: float) -> float:
            return w[max(0, min(n - 1, int(p * n)))]

        return {
            "n_samples":     self.n_samples,
            "n_failed":      self.n_failed,
            "mean":          round(statistics.mean(self.npv_W) if w else 0.0, 0),
            "std":           round(statistics.stdev(self.npv_W) if n > 1 else 0.0, 0),
            "p10":           round(pct(0.10), 0),
            "p50":           round(pct(0.50), 0),
            "p90":           round(pct(0.90), 0),
            "prob_positive": round(self.prob_exceeds(0.0), 4),
        }


# ── Monte Carlo runners ───────────────────────────────────────────────────────

def run_bass_welfare_mc(
    bass:    MCBassParams,
    welfare: MCModelParams,
    n:       int          = 1_000,
    seed:    int | None   = None,
) -> MCResult:
    """Monte Carlo over a single Bass diffusion + ClosedEconomy welfare run.

    Parameters
    ----------
    bass : MCBassParams
        Distributions over Bass adoption parameters.
    welfare : MCModelParams
        Distributions over welfare model parameters.
    n : int
        Number of Monte Carlo samples (default 1 000).
    seed : int or None
        Random seed for reproducibility.  None = non-deterministic.

    Returns
    -------
    MCResult
        Distribution of NPV_W, NPV_PS, NPV_CS across n samples.
        Failed samples (invalid parameter draws) are skipped and counted
        in MCResult.n_failed.
    """
    if seed is not None:
        random.seed(seed)

    npv_W, npv_PS, npv_CS = [], [], []
    n_failed = 0

    for _ in range(n):
        try:
            bass_p   = bass.draw()
            adoption = BassModel(bass_p).run().adoption_fracs
            model_p  = welfare.draw(bass_p.years, adoption)
            r        = ClosedEconomy(model_p).run()
            npv_W.append(r.npv_W)
            npv_PS.append(r.npv_PS)
            npv_CS.append(r.npv_CS)
        except (ValueError, ZeroDivisionError, OverflowError):
            n_failed += 1

    return MCResult(
        n_samples=n, npv_W=npv_W, npv_PS=npv_PS,
        npv_CS=npv_CS, n_failed=n_failed,
    )


def run_two_stage_welfare_mc(
    two_stage: MCTwoStageParams,
    welfare:   MCModelParams,
    n:         int        = 1_000,
    seed:      int | None = None,
) -> MCResult:
    """Monte Carlo over a two-stage Bass + ClosedEconomy welfare run.

    Parameters
    ----------
    two_stage : MCTwoStageParams
        Distributions over both Bass stages.
    welfare : MCModelParams
        Distributions over welfare parameters.
    n, seed : as per run_bass_welfare_mc.
    """
    if seed is not None:
        random.seed(seed)

    npv_W, npv_PS, npv_CS = [], [], []
    n_failed = 0

    for _ in range(n):
        try:
            ts_p           = two_stage.draw()
            stage1, stage2 = TwoStagePipeline(ts_p).run_stages()
            model_p        = welfare.draw(stage2.years, stage2.adoption_fracs)
            r       = ClosedEconomy(model_p).run()
            npv_W.append(r.npv_W)
            npv_PS.append(r.npv_PS)
            npv_CS.append(r.npv_CS)
        except (ValueError, ZeroDivisionError, OverflowError):
            n_failed += 1

    return MCResult(
        n_samples=n, npv_W=npv_W, npv_PS=npv_PS,
        npv_CS=npv_CS, n_failed=n_failed,
    )


def run_platform_welfare_mc(
    platform: MCNetworkPlatformParams,
    welfare:  MCModelParams,
    n:        int        = 1_000,
    seed:     int | None = None,
) -> MCResult:
    """Monte Carlo over a NetworkPlatformModel + ClosedEconomy welfare run.

    Parameters
    ----------
    platform : MCNetworkPlatformParams
        Distributions over network platform parameters.
    welfare : MCModelParams
        Distributions over welfare parameters.
    n, seed : as per run_bass_welfare_mc.
    """
    if seed is not None:
        random.seed(seed)

    npv_W, npv_PS, npv_CS = [], [], []
    n_failed = 0

    for _ in range(n):
        try:
            plat_p   = platform.draw()
            plat_res = NetworkPlatformModel(plat_p).run()
            model_p  = welfare.draw(plat_res.years, plat_res.adoption_fracs)
            r        = ClosedEconomy(model_p).run()
            npv_W.append(r.npv_W)
            npv_PS.append(r.npv_PS)
            npv_CS.append(r.npv_CS)
        except (ValueError, ZeroDivisionError, OverflowError):
            n_failed += 1

    return MCResult(
        n_samples=n, npv_W=npv_W, npv_PS=npv_PS,
        npv_CS=npv_CS, n_failed=n_failed,
    )
