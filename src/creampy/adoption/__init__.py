"""CREAMpy adoption sub-package — Bass diffusion model and pipeline."""

from .bass import BassModel, BassParams, BassResult, BassYearResult
from .seir import SeirModel, SeirParams, SeirResult, SeirYearResult
from .pipeline import (
    Pipeline,
    PipelineResult,
    TwoStageBassParams,
    TwoStagePipeline,
    TwoStagePipelineResult,
    to_dreampy_csv,
    to_dreampy_table,
)

__all__ = [
    # Bass model
    "BassParams",
    "BassYearResult",
    "BassResult",
    "BassModel",
    # SEIR + vaccination programme model
    "SeirParams",
    "SeirYearResult",
    "SeirResult",
    "SeirModel",
    # Single-stage pipeline (Bass → welfare)
    "Pipeline",
    "PipelineResult",
    # Two-stage pipeline (intermediary Bass → consumer Bass → welfare)
    "TwoStageBassParams",
    "TwoStagePipeline",
    "TwoStagePipelineResult",
    # DREAMpy export (Path B)
    "to_dreampy_table",
    "to_dreampy_csv",
]
