"""
CREAMpy full pipeline: Bass diffusion adoption -> DREAM welfare.
================================================================

Two execution paths are provided, matching the two ways researchers
use these models in practice:

PATH A — Fully in Python (recommended for scripting and automation)
    Bass adoption model  -->  CREAMpy ClosedEconomy  -->  NPV of welfare

    All computation happens inside this package. No Excel, no DREAMpy
    executable, no manual data entry.

    Use: Pipeline(bass_params, partial_model_params).run()

PATH B — Bass here, welfare in DREAMpy (for DREAMpy users)
    Bass adoption model  -->  adoption table export  -->  DREAMpy Excel  -->  NPV

    Generate the adoption schedule with CREAMpy's Bass model, then export
    it in the format DREAMpy expects and paste it into your DREAMpy template.
    Useful if your institution already uses DREAMpy and you only need the
    diffusion front-end.

    Use: to_dreampy_table(bass_result)
         to_dreampy_csv(bass_result, path)

DREAMpy adoption table format
------------------------------
DREAMpy v2.2.3 Closed Economy template expects an "Adoption" schedule with
one row per year. Typical column layout (confirm against your template version):

  Column A: Year
  Column B: Adoption rate (raw, before ptrs)
  Column C: Risk-adjusted adoption rate (= raw * ptrs)  <-- CREAMpy uses this

The export functions return Column C values (risk_adj_frac), which incorporate
ptrs from BassParams. If DREAMpy applies ptrs separately, export Column B
values (cumul_frac) and set DREAMpy's own ptrs field to match BassParams.ptrs.

Usage
-----
  from creampy.adoption import BassParams, BassModel
  from creampy.adoption.pipeline import Pipeline, PipelineParams, to_dreampy_table
  from creampy import ModelParams

  # --- PATH A: fully in Python ---
  bass = BassParams(p=0.01, q=0.40, ceiling=0.70, ptrs=0.80,
                    t0=2025, years=list(range(2025, 2046)))
  model = ModelParams(K=0.13, epsilon=0.5, eta=-0.5, P0=200.0, Q0=1_000_000.0,
                      discount_rate=0.05, base_year=2025)
  result = Pipeline(bass, model).run()
  print(f"NPV welfare: USD {result.welfare.npv_W:,.0f}")
  print(f"Bass peak year: {result.bass.peak_year}")

  # --- PATH B: export for DREAMpy ---
  bass_result = BassModel(bass).run()
  table = to_dreampy_table(bass_result)
  # Paste `table` into DREAMpy's adoption schedule.
  to_dreampy_csv(bass_result, "adoption_for_dreampy.csv")
"""

from __future__ import annotations

import csv
import dataclasses

from ..model import ClosedEconomy, ModelParams, ModelResult
from .bass import BassModel, BassParams, BassResult


def _assert_no_schedule(model_params: ModelParams) -> None:
    """Raise if model_params already carries years/adoption_fracs.

    Both Pipeline and TwoStagePipeline inject these from the adoption model.
    Passing pre-filled values would be silently overwritten, so we fail fast.
    """
    if model_params.years or model_params.adoption_fracs:
        raise ValueError(
            "model_params.years and adoption_fracs must be empty — "
            "the pipeline fills them from the adoption model output."
        )

__all__ = [
    "PipelineResult",
    "Pipeline",
    "TwoStageBassParams",
    "TwoStagePipelineResult",
    "TwoStagePipeline",
    "to_dreampy_table",
    "to_dreampy_csv",
]


# ── Result container ──────────────────────────────────────────────────────────

@dataclasses.dataclass
class PipelineResult:
    """Combined output of a Bass + ClosedEconomy pipeline run.

    Attributes
    ----------
    bass : BassResult
        Full Bass diffusion output, including per-year adoption fractions,
        peak year, and peak adoption rate.
    welfare : ModelResult
        Full ClosedEconomy output, including per-year surplus flows and
        NPV of producer surplus, consumer surplus, and total welfare.

    Quick access
    ------------
    result.welfare.npv_W     -- NPV of total welfare gains (USD)
    result.welfare.npv_PS    -- NPV of producer surplus (USD)
    result.welfare.npv_CS    -- NPV of consumer surplus (USD)
    result.bass.peak_year    -- year of highest new adoption rate
    result.bass.adoption_fracs  -- risk-adjusted adoption schedule used
    """
    bass:    BassResult
    welfare: ModelResult


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    """Full CREAMpy pipeline: Bass diffusion + DREAM welfare (Path A).

    Combines a Bass adoption model with a DREAM Closed Economy welfare
    model.  The Bass model generates the year-by-year adoption schedule;
    the welfare model translates that schedule into NPV of producer and
    consumer surplus.

    ``model_params`` must NOT include ``years`` or ``adoption_fracs`` —
    those are filled in from the Bass result.  Pass all other ModelParams
    fields (K, epsilon, eta, P0, Q0, discount_rate, base_year, scenario,
    shift_type, price_growth, qty_growth).

    Raises
    ------
    ValueError
        If ``model_params.years`` or ``model_params.adoption_fracs`` are
        non-empty (they would be overwritten, which is an error).

    Example
    -------
    >>> from creampy import ModelParams
    >>> from creampy.adoption import BassParams
    >>> from creampy.adoption.pipeline import Pipeline
    >>> bass = BassParams(p=0.01, q=0.40, ceiling=0.70, ptrs=0.80,
    ...                   t0=2025, years=list(range(2025, 2046)))
    >>> model = ModelParams(K=0.13, epsilon=0.5, eta=-0.5,
    ...                     P0=200.0, Q0=1_000_000.0,
    ...                     discount_rate=0.05, base_year=2025)
    >>> result = Pipeline(bass, model).run()
    >>> print(f"NPV_W = USD {result.welfare.npv_W:,.0f}")
    """

    def __init__(self, bass_params: BassParams, model_params: ModelParams) -> None:
        _assert_no_schedule(model_params)
        self.bass_params  = bass_params
        self.model_params = model_params

    def run(self) -> PipelineResult:
        """Execute both models and return combined results.

        Execution order:
          1. Run BassModel to produce the adoption schedule.
          2. Inject schedule into ModelParams.
          3. Run ClosedEconomy with the completed params.
        """
        bass_result = BassModel(self.bass_params).run()

        full_params = dataclasses.replace(
            self.model_params,
            years          = bass_result.years,
            adoption_fracs = bass_result.adoption_fracs,  # risk-adjusted
        )

        welfare_result = ClosedEconomy(full_params).run()
        return PipelineResult(bass=bass_result, welfare=welfare_result)


# ── Two-stage Bass pipeline ───────────────────────────────────────────────────

@dataclasses.dataclass
class TwoStageBassParams:
    """Parameters for a two-stage Bass diffusion model.

    Stage 1 — intermediary adoption (mills, distributors, manufacturers).
    Stage 2 — consumer/end-user adoption, gated by Stage 1 coverage.

    The Stage 2 effective ceiling at time t is:

        ceiling_series_2[t] = A_int(t) * ceiling_con

    where A_int(t) is the Stage 1 cumulative adoption fraction.  When no
    intermediaries have adopted (A_int = 0), the consumer market is fully
    closed.  As intermediaries come on board the accessible consumer market
    grows proportionally, reaching ceiling_con at full intermediary saturation.

    Typical ggo-hips use cases
    --------------------------
    - Nutrition fortification: mills adopt (Stage 1) → households buy
      fortified product through those mills (Stage 2)
    - Carbon crediting: project developers adopt MRV standard (Stage 1) →
      smallholders join carbon programmes through those developers (Stage 2)
    - Index insurance: insurance companies offer product (Stage 1) →
      farmers buy policies through those companies (Stage 2)

    Parameters
    ----------
    p_int, q_int : float
        Bass coefficients for the intermediary (Stage 1) population.
    ceiling_int : float
        Maximum intermediary adoption fraction (0, 1].
    p_con, q_con : float
        Bass coefficients for the consumer/end-user (Stage 2) population.
    ceiling_con : float
        Maximum consumer adoption fraction given full intermediary coverage.
        The effective ceiling at time t = A_int(t) * ceiling_con.
    ptrs : float
        Probability of technical and regulatory success.  Applied to
        Stage 2 output only — Stage 1 is a mechanical gate, not a product.
    t0 : int
        Launch year.  Stage 1 starts here; Stage 2 cannot start before t0.
    years : list of int
        Projection years for both stages.
    """
    p_int:       float
    q_int:       float
    ceiling_int: float
    p_con:       float
    q_con:       float
    ceiling_con: float
    t0:          int
    years:       list[int]
    ptrs:        float = 1.0


@dataclasses.dataclass
class TwoStagePipelineResult:
    """Output of a two-stage Bass + welfare pipeline run.

    Attributes
    ----------
    stage1 : BassResult
        Intermediary adoption schedule.
    stage2 : BassResult
        Consumer adoption schedule (time-varying ceiling from Stage 1).
    welfare : ModelResult
        NPV of producer and consumer surplus from ClosedEconomy.
    """
    stage1:  BassResult
    stage2:  BassResult
    welfare: ModelResult

    @property
    def adoption_fracs(self) -> list[float]:
        """Stage 2 risk-adjusted fracs used in the welfare model."""
        return self.stage2.adoption_fracs

    @property
    def years(self) -> list[int]:
        return self.stage2.years


class TwoStagePipeline:
    """Two-stage Bass diffusion pipeline: intermediary → consumer → welfare.

    Stage 1 runs a standard Bass model on the intermediary population.
    Stage 2 runs Bass on the consumer population with a time-varying ceiling
    equal to Stage 1 cumulative adoption × ceiling_con.  The welfare model
    receives Stage 2's risk-adjusted adoption schedule.

    ``model_params`` must NOT include ``years`` or ``adoption_fracs``
    (Pipeline fills them from Stage 2).

    Example
    -------
    >>> from creampy import ModelParams
    >>> from creampy.adoption.pipeline import TwoStageBassParams, TwoStagePipeline
    >>> ts = TwoStageBassParams(
    ...     p_int=0.02, q_int=0.35, ceiling_int=0.60,
    ...     p_con=0.005, q_con=0.30, ceiling_con=0.75,
    ...     ptrs=0.80, t0=2025, years=list(range(2025, 2046)),
    ... )
    >>> model = ModelParams(K=0.12, epsilon=0.5, eta=-0.5,
    ...                     P0=800.0, Q0=500_000.0,
    ...                     discount_rate=0.05, base_year=2025)
    >>> result = TwoStagePipeline(ts, model).run()
    >>> print(f"NPV_W = USD {result.welfare.npv_W:,.0f}")
    """

    def __init__(self, params: TwoStageBassParams,
                 model_params: ModelParams | None = None) -> None:
        if model_params is not None:
            _assert_no_schedule(model_params)
        self.params       = params
        self.model_params = model_params

    def run_stages(self) -> tuple[BassResult, BassResult]:
        """Run both Bass stages and return (stage1, stage2) without welfare.

        Use this when you only need the adoption schedule — e.g. inside the
        Monte Carlo runner where welfare parameters are sampled separately.
        """
        par = self.params
        stage1 = BassModel(BassParams(
            p=par.p_int, q=par.q_int, ceiling=par.ceiling_int,
            ptrs=1.0, t0=par.t0, years=par.years,
        )).run()
        ceiling_series_2 = [A * par.ceiling_con for A in stage1.cumulative_fracs]
        stage2 = BassModel(BassParams(
            p=par.p_con, q=par.q_con,
            ceiling=par.ceiling_con,
            ceiling_series=ceiling_series_2,
            ptrs=par.ptrs,
            t0=par.t0, years=par.years,
        )).run()
        return stage1, stage2

    def run(self) -> TwoStagePipelineResult:
        """Execute both Bass stages and the welfare model.

        Requires model_params to be set at construction time.
        For adoption-only runs call run_stages() instead.
        """
        if self.model_params is None:
            raise ValueError(
                "model_params is required for run(). "
                "Pass it at construction or call run_stages() for adoption only."
            )
        stage1, stage2 = self.run_stages()
        full_params = dataclasses.replace(
            self.model_params,
            years=stage2.years,
            adoption_fracs=stage2.adoption_fracs,
        )
        return TwoStagePipelineResult(
            stage1=stage1, stage2=stage2,
            welfare=ClosedEconomy(full_params).run(),
        )



# ── PATH B: DREAMpy export helpers ───────────────────────────────────────────

def to_dreampy_table(
    bass_result: BassResult,
    *,
    risk_adjusted: bool = True,
) -> dict[int, float]:
    """Export Bass adoption schedule as a year → fraction mapping (Path B).

    Returns a dict suitable for pasting into DREAMpy's adoption schedule
    worksheet.  Use ``risk_adjusted=True`` (default) for the ptrs-scaled
    fractions (Column C in a standard DREAMpy template).  Use
    ``risk_adjusted=False`` for the raw unadjusted fractions (Column B),
    in which case you should set DREAMpy's own ptrs field to match
    ``bass_result.params.ptrs``.

    Parameters
    ----------
    bass_result : BassResult
        Output from BassModel.run().
    risk_adjusted : bool
        If True (default), returns cumulative_frac * ptrs.
        If False, returns cumulative_frac (unadjusted).

    Returns
    -------
    dict[int, float]
        {year: adoption_fraction} for all years in the Bass result.
        Fractions for years before t0 are 0.0.

    DREAMpy template mapping
    ------------------------
    In DREAMpy v2.2.3 Closed Economy Excel template:
      - Navigate to the adoption schedule sheet (name varies by template version)
      - Column A: Year  (enter the keys of this dict)
      - Column B: Raw adoption fraction  (use risk_adjusted=False)
      - Column C: Risk-adjusted adoption  (use risk_adjusted=True, default)

    The welfare calculation in DREAMpy uses the risk-adjusted column.
    If DREAMpy applies ptrs internally (some versions do), use
    risk_adjusted=False and set DREAMpy's ptrs field to bass_result.params.ptrs.
    """
    if risk_adjusted:
        return {r.year: r.risk_adj_frac for r in bass_result.year_results}
    return {r.year: r.cumul_frac for r in bass_result.year_results}


def to_dreampy_csv(
    bass_result: BassResult,
    path: str,
    *,
    delimiter: str = ",",
) -> None:
    """Write Bass adoption schedule to a CSV file for DREAMpy import (Path B).

    The output CSV has three columns:

      year, adoption_raw, adoption_risk_adjusted

    where:
      adoption_raw            = cumulative Bass fraction (unadjusted)
      adoption_risk_adjusted  = adoption_raw * ptrs

    Import this file into DREAMpy's adoption schedule sheet, or use it
    as a reference when entering values manually.

    Parameters
    ----------
    bass_result : BassResult
        Output from BassModel.run().
    path : str
        Destination file path (e.g. "adoption_for_dreampy.csv").
    delimiter : str
        CSV delimiter (default comma). Use "\\t" for tab-separated.

    Notes
    -----
    - DREAMpy's adoption schedule is typically entered as annual *new*
      adoption rates, not cumulative fractions.  Inspect your template
      before deciding whether to use Column B (raw cumulative) or to
      compute annual new rates from it.
    - This function writes cumulative fractions. To convert to annual
      new rates: new_frac(t) = cumul_frac(t) - cumul_frac(t-1).
    """
    par = bass_result.params
    header_comment = (
        f"# CREAMpy Bass diffusion adoption schedule\n"
        f"# p={par.p}, q={par.q}, ceiling={par.ceiling}, "
        f"ptrs={par.ptrs}, t0={par.t0}\n"
        f"# adoption_risk_adjusted = adoption_raw * ptrs\n"
        f"# For DREAMpy: use adoption_risk_adjusted as the adoption input,\n"
        f"# or use adoption_raw and set DREAMpy's ptrs to {par.ptrs}.\n"
    )
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(header_comment)
        writer = csv.writer(fh, delimiter=delimiter)
        writer.writerow(["year", "adoption_raw", "adoption_risk_adjusted"])
        for r in bass_result.year_results:
            writer.writerow([r.year, f"{r.cumul_frac:.6f}", f"{r.risk_adj_frac:.6f}"])
