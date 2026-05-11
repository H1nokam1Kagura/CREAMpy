"""
Bass diffusion model — discrete annual adoption curve.
======================================================

Canonical reference:
  Bass, F.M. (1969). A new product growth for model consumer durables.
  Management Science, 15(5), 215–227.

Agricultural parameterisation guidance:
  Feder, G., Just, R.E. & Zilberman, D. (1985). Adoption of agricultural
  innovations in developing countries. Economic Development and Cultural
  Change, 33(2), 255–298.

  Diederen, P., van Meijl, H. & Wolters, A. (2003). Modernisation in
  agriculture: What makes a farmer adopt an innovation? Small Business
  Economics, 21(2), 125–141.

Model summary
-------------
The discrete Bass model describes how a new product spreads through a
population over annual periods.

Two adoption mechanisms drive uptake:
  - Innovation effect (p): awareness from external sources (extension
    services, mass media, demonstration sites). Proportional to the
    remaining non-adopters regardless of current adoption level.
  - Imitation effect (q): peer-to-peer word-of-mouth. Proportional to
    both current adopters and remaining non-adopters.

Recurrence (starting from A(t0 - 1) = 0):

  a(t) = [ p  +  q * A(t-1) ]  *  [ ceiling - A(t-1) ]
  A(t) = A(t-1) + a(t)

where:
  A(t)    -- cumulative adoption fraction at end of year t  (0 to ceiling)
  a(t)    -- new adoption fraction in year t
  p       -- coefficient of innovation       (typical range: 0.001 – 0.05)
  q       -- coefficient of imitation        (typical range: 0.10 – 0.60)
  ceiling -- maximum attainable adoption fraction (0 < ceiling <= 1.0)
             Models structural non-adoption: some farmers never adopt.

Risk adjustment
---------------
An optional ``ptrs`` (probability of technical and regulatory success)
multiplies the adoption curve:

  A_adj(t) = A(t) * ptrs

``ptrs`` is the caller's responsibility.  CREAMpy never applies it a
second time internally — if you pass risk-adjusted fracs to ClosedEconomy,
K must NOT also embed ptrs.

Typical parameter values for SSA crop research
-----------------------------------------------
  p = 0.003 – 0.01    (low external influence; extension reach is limited)
  q = 0.25  – 0.45    (moderate peer learning; farmers talk to neighbours)
  ceiling = 0.50–0.80 (structural non-adoption: resource, soil, market constraints)
  ptrs = 0.50 – 0.85  (varies by technology type and regulatory environment)

Peak adoption timing
--------------------
For the continuous Bass model the peak adoption rate occurs at:

  t* = ln(q/p) / (p + q)    [years after t0]

The discrete model peak is at approximately the same time. Use
``BassResult.peak_year`` for the discrete peak.

Usage
-----
  from creampy.adoption import BassParams, BassModel

  params = BassParams(
      p=0.01, q=0.40, ceiling=0.70, ptrs=0.80,
      t0=2025, years=list(range(2025, 2046)),
  )
  result = BassModel(params).run()
  print(result.adoption_fracs)    # risk-adjusted; pass to ClosedEconomy
  print(f"Peak year: {result.peak_year}")
"""

from __future__ import annotations

import dataclasses
import math

__all__ = ["BassParams", "BassYearResult", "BassResult", "BassModel"]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclasses.dataclass
class BassParams:
    """Inputs to one Bass diffusion run.

    Parameters
    ----------
    p : float
        Coefficient of innovation (external influence), in (0, 1).
        Models uptake driven by sources outside the adopter network:
        extension services, advertising, demo trials.
        Typical SSA crop research value: 0.003–0.01.
    q : float
        Coefficient of imitation (internal influence), in [0, 1).
        Models peer-to-peer word-of-mouth: a farmer adopts because
        their neighbours already have.
        Typical SSA crop research value: 0.25–0.45.
    ceiling : float
        Maximum attainable cumulative adoption fraction (0, 1].
        Represents structural non-adoption: the fraction of the target
        population that will never adopt due to resource constraints,
        soil suitability, market access, or risk aversion.
        Set to 1.0 for a standard Bass model with no ceiling.
        Ignored for any year where ``ceiling_series`` provides a value.
    t0 : int
        Launch year — the first calendar year in which adoption can occur.
        Years in ``years`` before ``t0`` are assigned A = 0.
    years : list of int
        Calendar years for which to compute adoption. May extend before
        t0 (returns 0) and/or after saturation.
    ptrs : float
        Probability of technical and regulatory success (0, 1].
        Multiplies the entire adoption curve: A_adj(t) = A(t) * ptrs.
        Represents the probability that the technology pipeline succeeds
        and the product reaches the market.
        Set to 1.0 to return unadjusted (deterministic) adoption.
    ceiling_series : list of float or None
        Per-year ceiling values — one per entry in ``years``, in the same
        order.  When provided, overrides the scalar ``ceiling`` for each
        year.  Used by ``TwoStagePipeline`` to supply a time-varying
        ceiling equal to the Stage-1 intermediary adoption at each period:

          ceiling_series[t] = A_int(t) * ceiling_con

        All values must be in (0, 1].  If the series reaches 0.0 for a
        period (no intermediaries yet), that year produces zero adoption
        regardless of p/q — the gate is closed.  Length must equal
        len(years).
    """
    p:              float
    q:              float
    ceiling:        float
    t0:             int
    years:          list[int]
    ptrs:           float             = 1.0
    ceiling_series: list[float] | None = None


@dataclasses.dataclass
class BassYearResult:
    """Per-year adoption outputs.

    All fractions are proportions of the total potential market.
    """
    year:           int
    new_frac:       float   # new adoption fraction in this year (a(t), unadjusted)
    cumul_frac:     float   # cumulative adoption fraction at year-end (A(t), unadjusted)
    risk_adj_frac:  float   # cumul_frac * ptrs (pass this to ClosedEconomy)


@dataclasses.dataclass
class BassResult:
    """Full output of one Bass diffusion run.

    ``adoption_fracs`` (= risk_adj_frac per year) is the list to pass
    directly to ``ModelParams.adoption_fracs`` in ClosedEconomy.
    """
    params:        BassParams
    year_results:  list[BassYearResult]
    peak_year:     int | None   # year with highest new_frac; None if no post-launch years
    peak_new_frac: float        # new adoption fraction at peak (unadjusted); 0.0 if no launch
    continuous_peak_offset: float | None  # ln(q/p)/(p+q) years after t0; None when q<=p

    # Convenience views (same data as year_results, as plain lists)
    @property
    def years(self) -> list[int]:
        """Calendar years, in order."""
        return [r.year for r in self.year_results]

    @property
    def adoption_fracs(self) -> list[float]:
        """Risk-adjusted cumulative adoption fractions — pass to ModelParams."""
        return [r.risk_adj_frac for r in self.year_results]

    @property
    def cumulative_fracs(self) -> list[float]:
        """Unadjusted cumulative fracs (before ptrs)."""
        return [r.cumul_frac for r in self.year_results]


# ── Core model ────────────────────────────────────────────────────────────────

class BassModel:
    """Discrete annual Bass diffusion model.

    Instantiate with ``BassParams``, call ``.run()`` for ``BassResult``.

    The model is stateless — construct a new instance per parameter set.
    """

    def __init__(self, params: BassParams) -> None:
        self.p = params
        self._validate()

    def _validate(self) -> None:
        par = self.p
        if not (0.0 < par.p < 1.0):
            raise ValueError(f"p must be in (0, 1), got {par.p}. "
                             "p=0 means no external influence (nothing ever starts); "
                             "p=1 means the entire market adopts in the first year.")
        if not (0.0 <= par.q < 1.0):
            raise ValueError(f"q must be in [0, 1), got {par.q}.")
        if not (0.0 < par.ceiling <= 1.0):
            raise ValueError(f"ceiling must be in (0, 1], got {par.ceiling}.")
        if not (0.0 < par.ptrs <= 1.0):
            raise ValueError(f"ptrs must be in (0, 1], got {par.ptrs}. "
                             "Use ptrs=1.0 for a deterministic (risk-ignored) run.")
        if not par.years:
            raise ValueError("years must be a non-empty list.")
        if par.ceiling_series is not None:
            if len(par.ceiling_series) != len(par.years):
                raise ValueError(
                    f"ceiling_series length ({len(par.ceiling_series)}) must equal "
                    f"years length ({len(par.years)})."
                )
            bad = [v for v in par.ceiling_series if not (0.0 <= v <= 1.0)]
            if bad:
                raise ValueError(
                    f"All ceiling_series values must be in [0, 1]: "
                    f"{len(bad)} out-of-range value(s), first 3: {bad[:3]}"
                )

    # ── Static per-year recurrence ────────────────────────────────────────────

    @staticmethod
    def _step(p: float, q: float, ceiling: float,
              prev_cumul: float) -> tuple[float, float]:
        """Single discrete Bass period.

        Returns (new_frac, new_cumul).  ``new_cumul`` is capped at ceiling.

        The calculation follows Bass (1969) adapted to annual discrete periods:

          a(t) = [p + q * prev_cumul] * [ceiling - prev_cumul]
          A(t) = prev_cumul + a(t)

        When prev_cumul reaches ceiling, a(t)=0 and A(t)=ceiling (saturation).
        """
        remaining = ceiling - prev_cumul
        if remaining <= 0.0:
            return 0.0, ceiling                     # already saturated
        new_frac = (p + q * prev_cumul) * remaining
        new_cumul = min(ceiling, prev_cumul + new_frac)
        return new_frac, new_cumul

    # ── Full projection ───────────────────────────────────────────────────────

    def run(self) -> BassResult:
        """Run the Bass model across all years and return a BassResult."""
        par = self.p
        year_results: list[BassYearResult] = []

        # Continuous-model peak approximation (only valid when q > p).
        # None signals "not applicable" rather than 0.0 which is ambiguous.
        if par.q > par.p:
            continuous_peak_offset: float | None = (
                math.log(par.q / par.p) / (par.p + par.q)
            )
        else:
            continuous_peak_offset = None  # p >= q: peak is at or before t0

        cumul      = 0.0
        peak_year: int | None = None
        peak_new   = 0.0
        use_series = par.ceiling_series is not None   # resolved once, not per-year

        for i, yr in enumerate(par.years):
            if yr < par.t0:
                year_results.append(BassYearResult(
                    year=yr, new_frac=0.0, cumul_frac=0.0, risk_adj_frac=0.0
                ))
                continue

            ceiling_t = par.ceiling_series[i] if use_series else par.ceiling

            # Gate: if intermediary coverage is zero, no consumer adoption possible
            if ceiling_t <= 0.0:
                year_results.append(BassYearResult(
                    year=yr, new_frac=0.0, cumul_frac=cumul,
                    risk_adj_frac=cumul * par.ptrs,
                ))
                continue

            new_frac, cumul = self._step(par.p, par.q, ceiling_t, cumul)

            if peak_year is None or new_frac > peak_new:
                peak_new  = new_frac
                peak_year = yr

            year_results.append(BassYearResult(
                year=yr,
                new_frac=new_frac,
                cumul_frac=cumul,
                risk_adj_frac=cumul * par.ptrs,
            ))

        return BassResult(
            params=par,
            year_results=year_results,
            peak_year=peak_year,
            peak_new_frac=peak_new,
            continuous_peak_offset=continuous_peak_offset,
        )
