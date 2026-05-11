"""
Network-effects platform diffusion model.
==========================================

Models a two-sided platform where a supply-side network (data providers,
institutional adopters) must reach critical mass before a demand-side
population (end users, farmers) can derive sufficient value to adopt.

Canonical theoretical foundation:
  Rochet, J-C. & Tirole, J. (2003). Platform competition in two-sided
  markets. Journal of the European Economic Association, 1(4), 990–1029.

  Armstrong, M. (2006). Competition in two-sided markets. RAND Journal
  of Economics, 37(3), 668–691.

Agricultural application context:
  Feder, G., Just, R.E. & Zilberman, D. (1985). Adoption of agricultural
  innovations in developing countries. Economic Development and Cultural
  Change, 33(2), 255–298.

  GSMA (2023). State of the Industry Report on Mobile Money. GSMA,
  Geneva. [calibration reference dataset]

The model is validated against the mobile money sector (GSMA data) as
the best available two-sided platform dataset in LMIC contexts. The
structural analog to agricultural weather platforms is:

  Mobile money agent network  ↔  NMHS / weather data providers
  Mobile money users          ↔  Farmers using weather forecasts
  Agent commission revenue    ↔  Forecast quality premium / service fees

Seven-equation coupled system (one iteration per year)
------------------------------------------------------

  [1]  a_p(t)  = [p_p  +  σ·A_f(t−1)/M_f_max  +  q_p·S(t−1)]
                  × [M_p − N_p(t−1)]

  [2]  N_p(t)  = N_p(t−1) + a_p(t)
  [3]  S(t)    = N_p(t) / M_p                    ← installed base fraction

  [4]  Q(t)    = 1 − exp(−λ · S(t))              ← platform quality (0–1)

  [5]  M_f(t)  = M_f_max · max(0, (S(t) − S_crit) / (1 − S_crit))
                                                  ← gated farmer market

  [6]  a_f(t)  = [p_f  +  q_f · A_f(t−1)]  ×  [M_f(t) − A_f(t−1)]

  [7]  A_f(t)  = A_f(t−1) + a_f(t)

Welfare is computed by passing A_f(t) × ptrs to ClosedEconomy.

Key parameters
--------------
  p_p, q_p   -- provider Bass coefficients (innovation, imitation)
  M_p        -- total addressable provider count (e.g. 50 NMHS in a region)
  sigma      -- cross-side spillover: farmer uptake → provider recruitment
  lambda_q   -- quality saturation rate (higher = quality rises faster with S)
  S_crit     -- critical mass fraction; no farmer market below this coverage
  p_f, q_f   -- farmer Bass coefficients
  M_f_max    -- maximum farmer market at full platform quality
  ptrs       -- probability of technical / regulatory success (scales A_f)

Degenerate cases
----------------
  σ=0, S_crit=0, λ large:  reduces to two independent Bass curves
  σ=0, S_crit>0:           reduces to gated two-stage Bass (no feedback)
  M_p=1, S_crit=0:         single provider, quality immediate → standard Bass
  p_p very small, short horizon: critical mass never crossed → A_f = 0

Usage
-----
  from creampy.adoption.network_platform import NetworkPlatformModel, NetworkPlatformParams

  params = NetworkPlatformParams(
      p_p=0.02, q_p=0.30, M_p=50.0,
      sigma=0.10, lambda_q=5.0, S_crit=0.20,
      p_f=0.005, q_f=0.35, M_f_max=5_000_000.0, ptrs=0.85,
      t0=2025, years=list(range(2025, 2046)),
  )
  result = NetworkPlatformModel(params).run()
  print(result.adoption_fracs)   # pass to ClosedEconomy
"""

from __future__ import annotations

import dataclasses
import math

__all__ = [
    "NetworkPlatformParams",
    "NetworkPlatformYearResult",
    "NetworkPlatformResult",
    "NetworkPlatformModel",
]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclasses.dataclass
class NetworkPlatformParams:
    """Inputs to one network-effects platform run.

    Provider-side (supply) parameters
    ----------------------------------
    p_p : float
        Provider innovation coefficient.  Probability a provider joins from
        external stimulus (mandate, funding, policy) regardless of existing
        network.  Typical range: 0.005–0.05.
    q_p : float
        Provider imitation coefficient.  Peer effect: a provider joins because
        neighbouring providers already have.  Typical range: 0.10–0.40.
    M_p : float
        Total addressable provider market (e.g., number of NMHS in target
        region, or total potential mobile money agent outlets).

    Network and quality parameters
    --------------------------------
    sigma : float
        Cross-side spillover coefficient.  Fraction of demand-side adoption
        rate that feeds back as additional recruitment pressure on providers.
        σ = 0 decouples the two sides (two-stage model, no feedback).
        Typical range: 0.05–0.20.
    lambda_q : float
        Quality saturation rate.  Controls how fast Q(t) approaches 1.0 as
        the installed base grows.  Q(t) = 1 − exp(−λ · S(t)).
        λ = 5 → Q ≈ 0.99 at S = 0.9.  λ = 1 → Q ≈ 0.63 at S = 1.
        Higher λ = quality rises more steeply with early provider growth.
    S_crit : float
        Critical mass threshold.  Provider coverage fraction below which
        the farmer market is effectively zero (platform not viable).
        S_crit = 0 removes the gate (always viable).
        S_crit = 0.2 means 20 % of providers must be on board first.

    Demand-side (farmer) parameters
    --------------------------------
    p_f : float
        Farmer innovation coefficient.  Typical SSA range: 0.001–0.01.
    q_f : float
        Farmer imitation coefficient.  Typical SSA range: 0.20–0.50.
    M_f_max : float
        Maximum farmer market at full platform quality (Q = 1).  The
        effective farmer market at time t is M_f_max × coverage_factor(S(t)).
    ptrs : float
        Probability of technical and regulatory success (0, 1].
        Scales the adoption curve: A_adj(t) = A_f(t) × ptrs.
        Applied once at output; never multiplied internally.

    Projection settings
    -------------------
    t0 : int
        Launch year.  Provider recruitment starts in t0; years before t0
        yield zero adoption on both sides.
    years : list of int
        Calendar years for the projection.  May extend before t0 (returns
        zero) and after saturation.
    N_p0 : float
        Optional seed: number of providers already active at launch (before
        the first year's Bass growth).  Useful when a platform launches with
        founding partners already committed.  Default: 0.
    """
    # Provider side
    p_p:      float
    q_p:      float
    M_p:      float

    # Network / quality
    sigma:    float
    lambda_q: float
    S_crit:   float

    # Demand side
    p_f:      float
    q_f:      float
    M_f_max:  float

    # Projection
    t0:       int
    years:    list[int]

    # Optional
    ptrs:     float = 1.0
    N_p0:     float = 0.0


@dataclasses.dataclass
class NetworkPlatformYearResult:
    """Per-year state of both sides of the platform.

    All counts and fractions are *unadjusted* (before ptrs).
    Multiply adoption_frac by ptrs for the risk-adjusted value.
    """
    year:             int
    # Provider side
    new_providers:    float   # new providers recruited in this year
    N_p:              float   # cumulative provider count
    S:                float   # installed base fraction = N_p / M_p
    # Quality and gate
    Q:                float   # platform quality  = 1 − exp(−λ·S)
    M_f_effective:    float   # effective farmer market (gated by S_crit)
    # Demand side
    new_farmers:      float   # new farmer adopters in this year (unadjusted)
    A_f:              float   # cumulative farmer adoption fraction (unadjusted)
    adoption_frac:    float   # A_f × ptrs  (pass this to ClosedEconomy)


@dataclasses.dataclass
class NetworkPlatformResult:
    """Full output of one network-platform model run."""
    params:       NetworkPlatformParams
    year_results: list[NetworkPlatformYearResult]
    peak_year:    int | None    # year of highest new_farmers; None if no launch
    peak_new_farmer_frac: float # new farmer fraction at peak (unadjusted)
    crit_mass_year: int | None  # first year S(t) >= S_crit; None if never reached

    @property
    def years(self) -> list[int]:
        """Calendar years in order."""
        return [r.year for r in self.year_results]

    @property
    def adoption_fracs(self) -> list[float]:
        """Risk-adjusted cumulative farmer adoption — pass to ModelParams."""
        return [r.adoption_frac for r in self.year_results]

    @property
    def provider_coverage(self) -> list[float]:
        """Installed base fraction S(t) per year."""
        return [r.S for r in self.year_results]


# ── Core model ────────────────────────────────────────────────────────────────

class NetworkPlatformModel:
    """Two-sided network-effects platform diffusion model.

    Implements the seven-equation coupled system described in the module
    docstring.  Instantiate with ``NetworkPlatformParams`` and call
    ``.run()`` for ``NetworkPlatformResult``.

    The model is stateless — each ``.run()`` call is independent.
    """

    def __init__(self, params: NetworkPlatformParams) -> None:
        self.p = params
        self._validate()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        par = self.p
        if not (0.0 < par.p_p < 1.0):
            raise ValueError(f"p_p must be in (0, 1), got {par.p_p}")
        if not (0.0 <= par.q_p < 1.0):
            raise ValueError(f"q_p must be in [0, 1), got {par.q_p}")
        if par.M_p <= 0:
            raise ValueError(f"M_p must be positive, got {par.M_p}")
        if not (0.0 <= par.sigma):
            raise ValueError(f"sigma must be non-negative, got {par.sigma}")
        if par.lambda_q <= 0:
            raise ValueError(f"lambda_q must be positive, got {par.lambda_q}")
        if not (0.0 <= par.S_crit < 1.0):
            raise ValueError(f"S_crit must be in [0, 1), got {par.S_crit}")
        if not (0.0 < par.p_f < 1.0):
            raise ValueError(f"p_f must be in (0, 1), got {par.p_f}")
        if not (0.0 <= par.q_f < 1.0):
            raise ValueError(f"q_f must be in [0, 1), got {par.q_f}")
        if par.M_f_max <= 0:
            raise ValueError(f"M_f_max must be positive, got {par.M_f_max}")
        if not (0.0 < par.ptrs <= 1.0):
            raise ValueError(f"ptrs must be in (0, 1], got {par.ptrs}")
        if not (0.0 <= par.N_p0 <= par.M_p):
            raise ValueError(f"N_p0 must be in [0, M_p], got {par.N_p0}")
        if not par.years:
            raise ValueError("years must be a non-empty list.")

    # ── Static per-period steps ───────────────────────────────────────────────

    @staticmethod
    def _provider_step(p_p: float, q_p: float, sigma: float,
                       M_p: float, N_p_prev: float,
                       A_f_prev: float, M_f_max: float) -> float:
        """Equation [1]: new provider recruits in one period.

        Combines:
          - Innovation effect (p_p): external pressure regardless of network
          - Imitation effect (q_p · S): peer effect from existing providers
          - Cross-side feedback (σ · A_f/M_f_max): demand-side uptake pulls in providers

        Returns new_providers (capped so N_p does not exceed M_p).
        """
        remaining = M_p - N_p_prev
        if remaining <= 0.0:
            return 0.0
        S_prev = N_p_prev / M_p
        feedback = sigma * (A_f_prev / M_f_max) if M_f_max > 0 else 0.0
        rate = p_p + q_p * S_prev + feedback
        return min(remaining, rate * remaining)

    @staticmethod
    def _quality(lambda_q: float, S: float) -> float:
        """Equation [4]: platform quality as a function of installed base.

        Q(t) = 1 − exp(−λ · S)

        Properties:
          - Q(0) = 0 (no providers → no quality)
          - Q → 1 as S → 1 (full coverage → full quality)
          - Concave: each additional provider adds less than the last
          - Higher λ → quality rises faster with early provider growth
        """
        return 1.0 - math.exp(-lambda_q * S)

    @staticmethod
    def _effective_farmer_market(M_f_max: float, S: float,
                                 S_crit: float) -> float:
        """Equation [5]: farmer market gated by critical mass.

        Below S_crit: platform quality insufficient → M_f = 0.
        Above S_crit: market grows linearly from 0 to M_f_max.

        This is the tipping point: a hard gate at S_crit followed by
        linear scaling to full market at S = 1.
        """
        if S <= S_crit:
            return 0.0
        return M_f_max * (S - S_crit) / (1.0 - S_crit)

    @staticmethod
    def _farmer_step(p_f: float, q_f: float,
                     A_f_prev: float, M_f_effective: float) -> float:
        """Equation [6]: new farmer adopters in one period.

        Standard discrete Bass applied to the time-varying effective market.

        A_f_prev is an absolute count; M_f_effective is the current market
        ceiling (also absolute count).  The imitation term q_f uses the
        adoption *fraction* A_f_prev / M_f_effective — normalising here
        is mandatory.  Using the raw absolute count would make q_f·A_f
        enormous for large markets and cause the entire market to saturate
        in the first two periods regardless of q_f.
        """
        remaining = M_f_effective - A_f_prev
        if remaining <= 0.0:
            return 0.0
        frac_adopted = A_f_prev / M_f_effective   # normalised to [0, 1]
        return (p_f + q_f * frac_adopted) * remaining

    # ── Full projection ───────────────────────────────────────────────────────

    def run(self) -> NetworkPlatformResult:
        """Run the coupled model across all years."""
        par = self.p
        year_results: list[NetworkPlatformYearResult] = []

        N_p   = par.N_p0   # cumulative provider count
        A_f   = 0.0        # cumulative farmer adoption (unadjusted fraction ∈ [0, M_f_max])

        peak_year:       int | None = None
        peak_new_farmer  = 0.0
        crit_mass_year:  int | None = None

        for yr in par.years:
            if yr < par.t0:
                # Pre-launch: both sides at zero
                S = N_p / par.M_p
                year_results.append(NetworkPlatformYearResult(
                    year=yr, new_providers=0.0, N_p=N_p, S=S,
                    Q=self._quality(par.lambda_q, S),
                    M_f_effective=0.0, new_farmers=0.0,
                    A_f=0.0, adoption_frac=0.0,
                ))
                continue

            # [1] Provider recruitment
            new_p = self._provider_step(
                par.p_p, par.q_p, par.sigma,
                par.M_p, N_p, A_f, par.M_f_max,
            )
            N_p = min(par.M_p, N_p + new_p)

            # [3] Installed base fraction
            S = N_p / par.M_p

            # Track first crossing of critical mass
            if crit_mass_year is None and S >= par.S_crit:
                crit_mass_year = yr

            # [4] Platform quality
            Q = self._quality(par.lambda_q, S)

            # [5] Effective farmer market
            M_f_eff = self._effective_farmer_market(par.M_f_max, S, par.S_crit)

            # [6] Farmer adoption step
            new_f = self._farmer_step(par.p_f, par.q_f, A_f, M_f_eff)
            A_f   = min(M_f_eff, A_f + new_f)

            # Track peak farmer recruitment
            if peak_year is None or new_f > peak_new_farmer:
                peak_new_farmer = new_f
                peak_year = yr

            year_results.append(NetworkPlatformYearResult(
                year=yr, new_providers=new_p, N_p=N_p, S=S,
                Q=Q, M_f_effective=M_f_eff, new_farmers=new_f,
                A_f=A_f,
                adoption_frac=A_f * par.ptrs / par.M_f_max,  # fraction of M_f_max, risk-adjusted
            ))

        return NetworkPlatformResult(
            params=par,
            year_results=year_results,
            peak_year=peak_year,
            peak_new_farmer_frac=peak_new_farmer / par.M_f_max if par.M_f_max > 0 else 0.0,
            crit_mass_year=crit_mass_year,
        )
