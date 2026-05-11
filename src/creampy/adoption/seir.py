"""
Discrete annual SEIR + vaccination model for livestock disease control.
=======================================================================

Models the welfare impact of a government vaccination programme by computing
the annual reduction in disease incidence relative to the untreated endemic
equilibrium.  The incidence reduction is passed to ClosedEconomy as the
adoption schedule, where K represents the per-animal productivity loss when
the animal is diseased.

Canonical references
--------------------
  Kermack, W.O. & McKendrick, A.G. (1927). A contribution to the mathematical
  theory of epidemics. Proc. Royal Society A, 115(772), 700–721.

  Anderson, R.M. & May, R.M. (1991). Infectious Diseases of Humans: Dynamics
  and Control. Oxford University Press.

  Rweyemamu, M. et al. (2008). Epidemiological patterns of foot-and-mouth
  disease worldwide. Transboundary and Emerging Diseases, 55(1), 57–72.

GGO-HIP use cases
-----------------
  animal-health-fmd-control   BB7  FMD Control, India + East Africa
  animal-health-solutions           Bovine TB, Ticks, Tryps, SSA

Compartments
------------
  S(t): susceptible fraction of herd
  E(t): exposed (incubating, not yet infectious)
  I(t): infectious fraction
  R(t): recovered / immune (includes vaccinated animals)

  Invariant: S(t) + E(t) + I(t) + R(t) = 1.0  at every timestep.

Discrete annual recurrence
--------------------------
  Vaccination schedule:
    v(t) = min(coverage_target,
               coverage_target × (t − t0 + 1) / coverage_ramp_years)
           for t ≥ t0;  0 otherwise.

  Waning immunity (optional):
    waning(t) = waning_rate × R(t)  [animals returning to susceptible pool]

  Transitions:
    new_E(t)  = β × S(t) × I(t)           [force of infection; mass action]
    new_I(t)  = σ × E(t)                   [progression from incubation]
    new_R(t)  = γ × I(t)                   [recovery / natural immunity]
    vacc(t)   = v(t) × S(t)               [vaccination; S → R directly]

  State update:
    S(t+1) = S(t) − new_E(t) − vacc(t)  + waning(t)
    E(t+1) = E(t) + new_E(t) − new_I(t)
    I(t+1) = I(t) + new_I(t) − new_R(t)
    R(t+1) = R(t) + new_R(t) + vacc(t)  − waning(t)

  All compartments clipped to [0, 1] and renormalised to sum to 1
  after each step to prevent floating-point drift.

Welfare link
------------
  incidence_reduction(t) = max(0, I_baseline − I(t))

  where I_baseline = I at endemic equilibrium before the programme starts
  (the initial condition I0, held constant as the counterfactual).

  adoption_fracs[t] = incidence_reduction(t) × ptrs

  Pass adoption_fracs to ClosedEconomy with:
    K = productivity_loss_per_infectious_fraction
        (e.g. 0.10 → a 10 % yield / liveweight loss per fraction-sick)

  This gives ΔPS/ΔCS/ΔW in units of USD per year from the livestock
  herd improvement, consistent with the Bass pipeline output.

Herd immunity threshold
-----------------------
  R0 = β / γ  (basic reproduction number)
  Herd immunity threshold: coverage_min = 1 − 1/R0

  When the steady-state vaccination coverage exceeds coverage_min, the
  disease is eventually eliminated (I → 0).  Below it, disease persists
  at a lower endemic level.  SeirResult.herd_immunity_achieved records
  the first year this threshold is breached.

Parameter guidance for SSA livestock diseases
----------------------------------------------
  FMD (Foot-and-Mouth):
    R0 ≈ 5–12 → β = R0 × γ;  γ ≈ 10 (annual);  σ ≈ 18 (annual)
    initial_prevalence ≈ 0.05–0.15 (5–15 % of herd actively infected)
    coverage_target ≈ 0.60–0.80;  coverage_ramp_years ≈ 5–8
    Waning immunity: 6–12 months → waning_rate ≈ 0.8–1.0 per year

  Bovine TB (slow, chronic):
    R0 ≈ 1.5–3;  γ ≈ 0.5 (2-year infectious period);  σ ≈ 2
    initial_prevalence ≈ 0.05–0.20
    coverage_target from test-and-slaughter programmes ≈ 0.40–0.70

  Trypanosomiasis (tick-borne, vector-mediated — approximation):
    Use a simplified β capturing tsetse fly density and treatment rates.
    Waning immunity effectively zero (no immune memory).

Usage
-----
  from creampy.adoption.seir import SeirModel, SeirParams
  from creampy import ClosedEconomy, ModelParams

  seir_p = SeirParams(
      beta=8.0, sigma=18.0, gamma=10.0,
      initial_prevalence=0.10,
      coverage_target=0.70, coverage_ramp_years=6,
      waning_rate=0.90,
      t0=2025, years=list(range(2025, 2046)),
      ptrs=0.80,
  )
  seir = SeirModel(seir_p).run()
  print(f"Herd immunity achieved: {seir.herd_immunity_achieved}")
  print(f"Final incidence reduction: {seir.adoption_fracs[-1]:.3%}")

  welfare = ClosedEconomy(ModelParams(
      K=0.12,               # 12 % productivity loss per infectious fraction
      epsilon=0.5, eta=-0.5,
      P0=500.0,             # USD/animal/year baseline value
      Q0=200_000.0,         # herd size
      years=seir.years,
      adoption_fracs=seir.adoption_fracs,
      discount_rate=0.05, base_year=2025,
  )).run()
"""

from __future__ import annotations

import dataclasses
import math

__all__ = ["SeirParams", "SeirYearResult", "SeirResult", "SeirModel"]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclasses.dataclass
class SeirParams:
    """Inputs to one SEIR + vaccination programme run.

    Parameters
    ----------
    beta : float
        Annual transmission rate (force of infection per susceptible per
        infectious fraction).  beta = R0 × gamma.  For FMD typical range
        is 5–120 (annual scale); use R0 × gamma where gamma is annual.
    sigma : float
        Annual incubation rate: fraction of E-compartment that becomes
        infectious per year.  sigma = 1 / mean_incubation_years.
        For FMD incubation ≈ 2–14 days: sigma ≈ 26–180 at annual scale.
    gamma : float
        Annual recovery rate: fraction of I-compartment that recovers per
        year.  gamma = 1 / mean_infectious_years.
        For FMD infectious period ≈ 2–4 weeks: gamma ≈ 13–26 at annual scale.
    initial_prevalence : float
        I/N at the endemic equilibrium before the programme starts.
        This is the counterfactual baseline held constant for welfare
        calculation.  Typical range: 0.02–0.20.
    coverage_target : float
        Target vaccination coverage fraction (0, 1].
        The programme aims to vaccinate this fraction of susceptibles each
        year at steady state.
    coverage_ramp_years : int
        Years to reach coverage_target from zero.  Linear ramp.
        Must be >= 1.
    t0 : int
        Year the vaccination programme starts.  Years before t0 yield
        zero incidence reduction (no programme yet).
    years : list of int
        Projection years.
    ptrs : float
        Probability of technical and regulatory success (0, 1].
        Scales the incidence reduction: adoption_fracs = reduction × ptrs.
    waning_rate : float
        Annual waning rate: fraction of R-compartment returning to S per
        year.  0.0 = permanent immunity (no waning); 1.0 = all immunity
        lost within one year (e.g. FMD where revaccination is annual).
        Default: 0.0.
    initial_E_fraction : float
        E/N at programme start (exposed but not yet infectious).
        Default: 0.0 (assume endemic equilibrium has negligible exposed pool
        relative to infectious).
    """
    beta:                float
    sigma:               float
    gamma:               float
    initial_prevalence:  float
    coverage_target:     float
    coverage_ramp_years: int
    t0:                  int
    years:               list[int]
    ptrs:                float = 1.0
    waning_rate:         float = 0.0
    initial_E_fraction:  float = 0.0


@dataclasses.dataclass
class SeirYearResult:
    """Per-year SEIR compartment state and welfare-relevant outputs."""
    year:                int
    S:                   float   # susceptible fraction
    E:                   float   # exposed fraction
    I:                   float   # infectious fraction
    R:                   float   # recovered/immune fraction
    vaccination_rate:    float   # v(t): fraction of S vaccinated this year
    incidence_reduction: float   # max(0, I_baseline - I(t)); unadjusted
    adoption_frac:       float   # incidence_reduction * ptrs (→ ClosedEconomy)


@dataclasses.dataclass
class SeirResult:
    """Full output of one SEIR vaccination programme run.

    ``adoption_fracs`` is the list to pass to ``ModelParams.adoption_fracs``.
    """
    params:                SeirParams
    year_results:          list[SeirYearResult]
    R0:                    float          # beta / gamma
    herd_immunity_threshold: float        # 1 - 1/R0
    herd_immunity_achieved:  int | None   # first year cumulative vacc > threshold
    endemic_prevalence:    float          # I0 = initial_prevalence (counterfactual)

    @property
    def years(self) -> list[int]:
        return [r.year for r in self.year_results]

    @property
    def adoption_fracs(self) -> list[float]:
        """Risk-adjusted incidence reduction — pass to ModelParams."""
        return [r.adoption_frac for r in self.year_results]

    @property
    def incidence_series(self) -> list[float]:
        """Unadjusted I(t) per year."""
        return [r.I for r in self.year_results]


# ── Core model ────────────────────────────────────────────────────────────────

class SeirModel:
    """Discrete annual SEIR + vaccination programme model.

    Instantiate with ``SeirParams``, call ``.run()`` for ``SeirResult``.
    The model is stateless — each ``.run()`` is independent.
    """

    def __init__(self, params: SeirParams) -> None:
        self.p = params
        self._validate()

    def _validate(self) -> None:
        par = self.p
        if par.beta <= 0:
            raise ValueError(f"beta must be positive, got {par.beta}")
        if par.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {par.sigma}")
        if par.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {par.gamma}")
        if not (0.0 < par.initial_prevalence < 1.0):
            raise ValueError(
                f"initial_prevalence must be in (0, 1), got {par.initial_prevalence}"
            )
        if not (0.0 < par.coverage_target <= 1.0):
            raise ValueError(
                f"coverage_target must be in (0, 1], got {par.coverage_target}"
            )
        if par.coverage_ramp_years < 1:
            raise ValueError(
                f"coverage_ramp_years must be >= 1, got {par.coverage_ramp_years}"
            )
        if not (0.0 < par.ptrs <= 1.0):
            raise ValueError(f"ptrs must be in (0, 1], got {par.ptrs}")
        if not (0.0 <= par.waning_rate <= 1.0):
            raise ValueError(
                f"waning_rate must be in [0, 1], got {par.waning_rate}"
            )
        if not (0.0 <= par.initial_E_fraction < 1.0):
            raise ValueError(
                f"initial_E_fraction must be in [0, 1), got {par.initial_E_fraction}"
            )
        if not par.years:
            raise ValueError("years must be a non-empty list.")

    @staticmethod
    def _vaccination_rate(t: int, t0: int,
                          coverage_target: float,
                          ramp_years: int) -> float:
        """Annual vaccination coverage at year t.

        Linear ramp from 0 at t0 to coverage_target at t0 + ramp_years - 1,
        then constant at coverage_target.  Zero before t0.
        """
        if t < t0:
            return 0.0
        offset = t - t0 + 1
        return min(coverage_target, coverage_target * offset / ramp_years)

    @staticmethod
    def _step(S: float, E: float, I: float, R: float,
              beta: float, sigma: float, gamma: float,
              v_t: float, waning_rate: float) -> tuple[float, float, float, float]:
        """One discrete annual SEIR step with vaccination and waning immunity.

        Returns (S_new, E_new, I_new, R_new) renormalised to sum to 1.
        All inputs and outputs are fractions of total herd (N = 1).

        Transitions use mass-action incidence (β × S × I) which is the
        standard for livestock disease models at herd level.
        """
        # Exponential discretisation: prob(infected in one year) = 1 - exp(-β·I).
        # Bounded in [0,1] regardless of β, avoiding saturation when β·I > 1
        # (which occurs for fast-moving diseases at annual timestep scale).
        prob_infect = 1.0 - math.exp(-beta * I)
        new_E   = S * prob_infect                        # new exposures
        new_I   = E * (1.0 - math.exp(-sigma))          # incubation → infectious
        new_R   = I * (1.0 - math.exp(-gamma))          # recovery
        vacc    = min(max(0.0, S - new_E), v_t * S)     # vaccinate remaining S
        waning  = waning_rate * R                        # waning immunity: R → S

        S_new = S - new_E - vacc + waning
        E_new = E + new_E - new_I
        I_new = I + new_I - new_R
        R_new = R + new_R + vacc - waning

        # Clip to [0, 1] and renormalise to prevent floating-point drift
        S_new = max(0.0, S_new)
        E_new = max(0.0, E_new)
        I_new = max(0.0, I_new)
        R_new = max(0.0, R_new)
        total = S_new + E_new + I_new + R_new
        if total > 0:
            S_new /= total
            E_new /= total
            I_new /= total
            R_new /= total
        return S_new, E_new, I_new, R_new

    def run(self) -> SeirResult:
        """Run the SEIR model across all years and return a SeirResult."""
        par = self.p

        R0  = par.beta / par.gamma
        hit = 1.0 - 1.0 / R0 if R0 > 1.0 else 0.0   # herd immunity threshold

        # Initial conditions at programme start
        I0 = par.initial_prevalence
        E0 = par.initial_E_fraction
        R0_init = 0.0                 # no prior vaccination assumed
        S0 = max(0.0, 1.0 - I0 - E0 - R0_init)
        total0 = S0 + E0 + I0 + R0_init
        S_cur = S0 / total0
        E_cur = E0 / total0
        I_cur = I0 / total0
        R_cur = R0_init / total0

        # Cumulative vaccination for herd immunity tracking
        cumulative_vaccinated = 0.0
        herd_immunity_achieved: int | None = None
        year_results: list[SeirYearResult] = []

        for yr in par.years:
            v_t = self._vaccination_rate(
                yr, par.t0, par.coverage_target, par.coverage_ramp_years
            )

            if yr < par.t0:
                # Pre-programme: disease at endemic level, no intervention
                year_results.append(SeirYearResult(
                    year=yr, S=S_cur, E=E_cur, I=I_cur, R=R_cur,
                    vaccination_rate=0.0,
                    incidence_reduction=0.0,
                    adoption_frac=0.0,
                ))
                continue

            S_cur, E_cur, I_cur, R_cur = self._step(
                S_cur, E_cur, I_cur, R_cur,
                par.beta, par.sigma, par.gamma,
                v_t, par.waning_rate,
            )

            # Track herd immunity threshold crossing
            cumulative_vaccinated += v_t
            if (herd_immunity_achieved is None
                    and R_cur >= hit and hit > 0):
                herd_immunity_achieved = yr

            incidence_reduction = max(0.0, I0 - I_cur)

            year_results.append(SeirYearResult(
                year=yr,
                S=S_cur, E=E_cur, I=I_cur, R=R_cur,
                vaccination_rate=v_t,
                incidence_reduction=incidence_reduction,
                adoption_frac=incidence_reduction * par.ptrs,
            ))

        return SeirResult(
            params=par,
            year_results=year_results,
            R0=R0,
            herd_immunity_threshold=hit,
            herd_immunity_achieved=herd_immunity_achieved,
            endemic_prevalence=I0,
        )
