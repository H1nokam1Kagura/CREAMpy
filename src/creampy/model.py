"""
CREAMpy — Crops Research EvaluAtion for Management, Python edition
==================================================================

Python implementation of the DREAM Closed Economy partial-equilibrium surplus
model. Computes research-induced welfare gains (producer surplus ΔPS, consumer
surplus ΔCS) for a single commodity market under a research-induced supply shift.

Canonical references
--------------------
  Alston, J.M., Norton, G.W. & Pardey, P.G. (1995). Science Under Scarcity.
    Cornell University Press. Ch. 4–5.

  Wood, S., Maredia, M. & Pardey, P.G. (2001). Prioritizing agricultural
    research for sustainable development using DREAM. IFPRI, Washington DC.

  Falck-Zepeda, J.B. et al. (2019–2022). IFPRI Discussion Papers
    1896, 1911, 1926, 2107.

Model summary
-------------
Two supply-shift conventions are supported:

  K-shift  (parallel cost reduction — default)
    Supply curve shifts DOWN by K (proportion of price). Quantity rises,
    price falls. Interpretation: a unit cost reduction, e.g. from drought-
    tolerant varieties that reduce losses per hectare.

    ps_share = |η| / (ε + |η|)   producers benefit more with inelastic supply
    cs_share = ε   / (ε + |η|)   consumers benefit more with elastic supply

  J-shift  (horizontal output expansion — yield-augmenting)
    Supply curve shifts RIGHT by J (proportion of quantity). Same price-
    quantity arithmetic, but welfare partition reverses because the extra
    output accrues to producers first.

    ps_share = ε   / (ε + |η|)   producers benefit more with elastic supply
    cs_share = |η| / (ε + |η|)   consumers benefit more with inelastic demand

Core equations (ANP 1995, Ch. 4 — linearised closed-economy approximation)
---------------------------------------------------------------------------
  denom      = ε − η              (> 0, since η < 0)
  Z          = ε · |η| / denom    surplus shape factor (second-order correction)
  correction = 1 + 0.5 · K · Z   area of the welfare triangle

  ΔPS_t = K · P0_t · Q0_t · A_t · ps_share · correction
  ΔCS_t = K · P0_t · Q0_t · A_t · cs_share · correction
  ΔW_t  = ΔPS_t + ΔCS_t          = K · P0_t · Q0_t · A_t · correction

  NPV_x = Σ_t [ ΔX_t / (1 + r)^(t − base_year) ]   x ∈ {PS, CS, W}

where A_t is the adoption fraction at time t (supplied by the caller).

Usage
-----
  from creampy import ClosedEconomy, ModelParams

  params = ModelParams(
      K              = 0.15,       # supply shift (= yield_gain / (1 + yield_gain))
      epsilon        = 0.5,        # supply elasticity (positive)
      eta            = -0.5,       # demand elasticity (negative)
      P0             = 200.0,      # base producer price (USD/tonne)
      Q0             = 1_000_000.0,# base quantity (tonnes)
      years          = list(range(2025, 2040)),
      adoption_fracs = [min(1.0, i / 10) for i in range(1, 16)],
      discount_rate  = 0.05,
      base_year      = 2025,
  )
  result = ClosedEconomy(params).run()
  print(f"NPV welfare: USD {result.npv_W:,.0f}")

  # Validate against 9 closed-form analytical cases:
  python -m creampy --validate
"""

from __future__ import annotations

import dataclasses

__version__ = "1.0.0"
__all__ = ["ModelParams", "YearResult", "ModelResult", "ClosedEconomy", "k_from_yield_gain"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def k_from_yield_gain(yield_gain_fraction: float) -> float:
    """Convert a proportional yield gain to a K-shift coefficient.

    K = Δy / (1 + Δy)

    Example: a 15 % yield gain (yield_gain_fraction=0.15) gives K ≈ 0.1304.
    """
    if yield_gain_fraction < 0:
        raise ValueError(f"yield_gain_fraction must be non-negative, got {yield_gain_fraction}")
    return yield_gain_fraction / (1.0 + yield_gain_fraction)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ModelParams:
    """Inputs to one model run.

    Parameters
    ----------
    K : float
        Supply shift coefficient in [0, 1].
        For a yield-gain scenario use ``k_from_yield_gain(delta_yield)``.
        For a cost-reduction scenario K = cost_saving_per_unit / base_price.
    epsilon : float
        Supply elasticity (positive, e.g. 0.5).
    eta : float
        Demand elasticity (negative, e.g. -0.5).
    P0 : float
        Baseline producer price (USD/tonne or any consistent currency/unit).
    Q0 : float
        Baseline production quantity (tonnes or consistent unit).
    years : list of int
        Projection years. Length must equal len(adoption_fracs).
    adoption_fracs : list of float
        Adoption fraction in [0, 1] for each year. Caller is responsible for
        any risk adjustment (e.g. multiplying by a probability-of-technical-
        success factor before passing in).
    discount_rate : float
        Annual discount rate (default 0.05 = 5 %).
    base_year : int
        Discounting anchor year (default 2025). A flow in ``base_year`` has
        pv_factor = 1.0; flows before ``base_year`` have pv_factor > 1.
        Must not exceed ``min(years)`` for results to be meaningful.
    scenario : str
        Label attached to the run_uid (default "central").
    shift_type : str
        "K" for parallel cost-reduction shift (default);
        "J" for horizontal yield-augmenting shift.
        See module docstring for welfare-partition differences.
    price_growth : float
        Annual real price growth rate (default 0.0 = constant real price).
    qty_growth : float
        Annual production quantity growth rate (default 0.0 = constant base).
    """
    K:              float
    epsilon:        float
    eta:            float
    P0:             float
    Q0:             float
    years:          list[int]   = dataclasses.field(default_factory=list)
    adoption_fracs: list[float] = dataclasses.field(default_factory=list)
    discount_rate:  float = 0.05
    base_year:      int   = 2025
    scenario:       str   = "central"
    shift_type:     str   = "K"
    price_growth:   float = 0.0
    qty_growth:     float = 0.0


@dataclasses.dataclass
class YearResult:
    """Undiscounted welfare flows for a single projection year.

    All monetary values (dPS, dCS, dW) are in the same currency/unit as P0·Q0
    and are *undiscounted*. Multiply by ``pv_factor`` to get the present value
    contribution to NPV.
    """
    year:      int
    adoption:  float    # adoption fraction in this year (0–1, as supplied)
    P0_t:      float    # price in year t: P0 * (1 + price_growth)^(t - base_year)
    Q0_t:      float    # quantity in year t: Q0 * (1 + qty_growth)^(t - base_year)
    dPS:       float    # producer surplus gain in year t (undiscounted)
    dCS:       float    # consumer surplus gain in year t (undiscounted)
    dW:        float    # total welfare gain = dPS + dCS (undiscounted)
    pv_factor: float    # discount factor: 1 / (1 + r)^(t - base_year)


@dataclasses.dataclass
class ModelResult:
    """Full model output for one run.

    ``npv_W = npv_PS + npv_CS`` holds exactly (to float precision).
    """
    params:       ModelParams
    year_results: list[YearResult]
    npv_PS:       float
    npv_CS:       float
    npv_W:        float
    run_uid:      str
    vintage:      str   # ISO date string (YYYY-MM-DD) when the run was executed


# ── Core model ────────────────────────────────────────────────────────────────

class ClosedEconomy:
    """DREAM Closed Economy partial-equilibrium surplus model.

    Instantiate with a ``ModelParams`` object and call ``.run()`` to get a
    ``ModelResult``. The instance is stateless — the same object can be
    reused with different inputs by calling ``ClosedEconomy(new_params).run()``.
    """

    def __init__(self, params: ModelParams) -> None:
        self.p = params
        self._validate()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        p = self.p
        if p.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {p.epsilon}")
        if p.eta >= 0:
            raise ValueError(f"eta must be negative, got {p.eta}")
        if not (0.0 <= p.K <= 1.0):
            raise ValueError(f"K must be in [0, 1], got {p.K}")
        if p.P0 <= 0:
            raise ValueError(f"P0 must be positive, got {p.P0}")
        if p.Q0 <= 0:
            raise ValueError(f"Q0 must be positive, got {p.Q0}")
        if len(p.years) != len(p.adoption_fracs):
            raise ValueError(
                f"years ({len(p.years)}) and adoption_fracs ({len(p.adoption_fracs)}) "
                "must have the same length"
            )
        if any(not (0.0 <= a <= 1.0) for a in p.adoption_fracs):
            raise ValueError("All adoption_fracs must be in [0, 1]")
        if not (0.0 <= p.discount_rate < 1.0):
            raise ValueError(
                f"discount_rate must be in [0, 1), got {p.discount_rate}. "
                "Use 0.0 for an undiscounted (summed) result."
            )
        if p.shift_type not in ("K", "J"):
            raise ValueError(f"shift_type must be 'K' or 'J', got {p.shift_type!r}")

    # ── Per-year surplus ──────────────────────────────────────────────────────

    @staticmethod
    def _welfare(K: float, epsilon: float, eta: float,
                 P0_t: float, Q0_t: float, adoption: float,
                 shift_type: str) -> tuple[float, float, float]:
        """Return (ΔPS, ΔCS, ΔW) for a single year.

        ANP (1995) Ch. 4 linearised closed-economy approximation.

        Welfare partition depends on shift type:
          K-shift: ps_share = |η|/(ε+|η|);  cs_share = ε/(ε+|η|)
          J-shift: ps_share = ε/(ε+|η|);    cs_share = |η|/(ε+|η|)
        Both apply the same second-order correction: 1 + 0.5·K·Z
        """
        if K == 0.0 or adoption == 0.0:
            return 0.0, 0.0, 0.0

        eta_abs = -eta                          # |η|, positive
        denom   = epsilon + eta_abs             # = ε − η > 0
        Z       = epsilon * eta_abs / denom     # surplus shape factor

        if shift_type == "J":
            ps_share = epsilon  / denom
            cs_share = eta_abs  / denom
        else:                                   # "K" (default)
            ps_share = eta_abs  / denom
            cs_share = epsilon  / denom

        correction = 1.0 + 0.5 * K * Z
        base = K * P0_t * Q0_t * adoption

        dPS = base * ps_share * correction
        dCS = base * cs_share * correction
        dW  = dPS + dCS
        return dPS, dCS, dW

    # ── Full projection run ───────────────────────────────────────────────────

    def run(self) -> ModelResult:
        """Execute the model across all years and return a ``ModelResult``."""
        from datetime import date

        p = self.p
        year_results: List[YearResult] = []

        for yr, adpt in zip(p.years, p.adoption_fracs):
            lag  = yr - p.base_year
            P0_t = p.P0 * (1.0 + p.price_growth) ** lag
            Q0_t = p.Q0 * (1.0 + p.qty_growth)   ** lag
            pv   = 1.0  / (1.0 + p.discount_rate) ** lag

            dPS, dCS, dW = self._welfare(
                p.K, p.epsilon, p.eta, P0_t, Q0_t, adpt, p.shift_type
            )
            year_results.append(YearResult(
                year=yr, adoption=adpt, P0_t=P0_t, Q0_t=Q0_t,
                dPS=dPS, dCS=dCS, dW=dW, pv_factor=pv,
            ))

        npv_PS = sum(r.dPS * r.pv_factor for r in year_results)
        npv_CS = sum(r.dCS * r.pv_factor for r in year_results)
        npv_W  = npv_PS + npv_CS

        vintage = date.today().isoformat()
        run_uid = f"dream_closed_economy__{p.scenario}__{__version__}__{vintage}"

        return ModelResult(
            params=p, year_results=year_results,
            npv_PS=npv_PS, npv_CS=npv_CS, npv_W=npv_W,
            run_uid=run_uid, vintage=vintage,
        )
