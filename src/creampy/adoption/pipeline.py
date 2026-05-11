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

__all__ = [
    "PipelineResult",
    "Pipeline",
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
        if model_params.years or model_params.adoption_fracs:
            raise ValueError(
                "model_params.years and model_params.adoption_fracs must be empty "
                "(or default) when using Pipeline — they are filled from the Bass result. "
                "Pass a ModelParams constructed without those two fields."
            )
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
