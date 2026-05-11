"""CREAMpy adoption sub-package — Bass diffusion model and pipeline."""

from .bass import BassModel, BassParams, BassResult, BassYearResult
from .pipeline import Pipeline, PipelineResult, to_dreampy_csv, to_dreampy_table

__all__ = [
    # Bass model
    "BassParams",
    "BassYearResult",
    "BassResult",
    "BassModel",
    # Pipeline
    "Pipeline",
    "PipelineResult",
    # DREAMpy export (Path B)
    "to_dreampy_table",
    "to_dreampy_csv",
]
