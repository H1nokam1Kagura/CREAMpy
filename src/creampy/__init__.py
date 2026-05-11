"""
CREAMpy — Crops Research Economic Adoption Model, Python edition.

A complete open-source Python pipeline for agricultural research impact evaluation:

  Bass diffusion  -->  DREAM Closed Economy welfare  -->  NPV of surplus

Two entry points:
  - ``creampy.adoption.Pipeline``  — full chain (Bass + welfare) in one call
  - ``creampy.ClosedEconomy``      — welfare model only (supply your own adoption)

Two export paths:
  - Path A (in-repo): Pipeline.run() -> PipelineResult  (no external tools)
  - Path B (DREAMpy): to_dreampy_table / to_dreampy_csv -> paste into DREAMpy Excel

Quick start:
  from creampy import ModelParams
  from creampy.adoption import BassParams
  from creampy.adoption.pipeline import Pipeline

  bass = BassParams(p=0.01, q=0.40, ceiling=0.70, ptrs=0.80,
                    t0=2025, years=list(range(2025, 2046)))
  model = ModelParams(K=0.13, epsilon=0.5, eta=-0.5, P0=200.0, Q0=1_000_000.0,
                      discount_rate=0.05, base_year=2025)
  result = Pipeline(bass, model).run()
  print(f"NPV welfare: USD {result.welfare.npv_W:,.0f}")
"""

from .model import (
    ClosedEconomy,
    ModelParams,
    ModelResult,
    YearResult,
    k_from_yield_gain,
    __version__,
)
from .monte_carlo import (
    sample,
    MCBassParams,
    MCModelParams,
    MCTwoStageParams,
    MCNetworkPlatformParams,
    MCResult,
    run_bass_welfare_mc,
    run_two_stage_welfare_mc,
    run_platform_welfare_mc,
)

__all__ = [
    # Welfare model
    "ClosedEconomy",
    "ModelParams",
    "ModelResult",
    "YearResult",
    "k_from_yield_gain",
    "__version__",
    # Monte Carlo
    "sample",
    "MCBassParams",
    "MCModelParams",
    "MCTwoStageParams",
    "MCNetworkPlatformParams",
    "MCResult",
    "run_bass_welfare_mc",
    "run_two_stage_welfare_mc",
    "run_platform_welfare_mc",
    # Adoption sub-package importable as creampy.adoption.*
]
