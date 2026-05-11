# Changelog

All notable changes to CREAMpy are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

**`creampy.adoption.network_platform` — two-sided network-effects platform model**
- `NetworkPlatformModel` class: 7-equation coupled system (provider Bass +
  quality function + critical mass gate + downstream farmer Bass + cross-side
  spillover feedback)
- `NetworkPlatformParams` dataclass: p_p, q_p, M_p (provider side); sigma,
  lambda_q, S_crit (network/quality); p_f, q_f, M_f_max, ptrs (farmer side);
  N_p0 seed support
- `NetworkPlatformResult` / `NetworkPlatformYearResult`: per-year provider
  coverage S(t), quality Q(t), effective farmer market, farmer adoption; plus
  peak_year, crit_mass_year
- 9 analytical validation cases covering all degenerate cases including the
  diffusion-peak timing test that guards the q_f normalisation invariant

**Evaluation harness**
- `eval/eval_network_platform.py`: 9 analytical tests + grid-search calibration
  against M-Pesa Kenya synthetic data
- `eval/eval_network_platform.ps1`: PS1 orchestrator with three phases; optional
  GSMA real-data extraction via openpyxl
- `eval/data/mpesa_kenya_synthetic.csv`: 2007–2020 reference dataset from CBK /
  GSMA published reports; structural analog for LMIC platform calibration

### Fixed
- `NetworkPlatformModel._farmer_step`: imitation term `q_f` now uses normalised
  adoption fraction rather than absolute count — without the fix, any q_f > 0
  caused the entire farmer market to saturate within two years of launch

---

## [1.0.0] — 2026-05-11

Initial public release.

### Added

**`creampy` — DREAM Closed Economy welfare model**
- `ClosedEconomy` class: discrete annual partial-equilibrium surplus model
- `ModelParams` dataclass: K, ε, η, P0, Q0, adoption schedule, discount rate, base year, scenario, shift type, price/qty growth
- `ModelResult` / `YearResult` dataclasses: per-year ΔPS/ΔCS/ΔW flows and NPV totals
- K-shift (parallel cost reduction) and J-shift (horizontal yield-augmenting) welfare partitions
- `k_from_yield_gain()` helper: converts proportional yield gain to K coefficient
- CLI (`python -m creampy`): `--validate`, `--k`/`--yield-gain`, `--price`, `--qty`, `--shift-type`, `--out-json`
- 12 closed-form analytical validation cases; all pass to float precision

**`creampy.adoption` — Bass diffusion adoption model**
- `BassModel` class: discrete annual Bass diffusion with ceiling and ptrs
- `BassParams` dataclass: p (innovation), q (imitation), ceiling, ptrs, t0, years
- `BassResult` / `BassYearResult`: per-year new_frac, cumul_frac, risk_adj_frac; peak_year
- 9 analytical validation cases including pure-innovation closed-form, ceiling enforcement, and peak timing

**`creampy.adoption.pipeline` — full pipeline and DREAMpy export**
- `Pipeline` class: Bass → ClosedEconomy in one call (Path A, fully in Python)
- `to_dreampy_table()`: year → adoption fraction dict for DREAMpy Excel import (Path B)
- `to_dreampy_csv()`: adoption schedule CSV with column documentation for DREAMpy

**Infrastructure**
- `Dockerfile` and `docker-compose.yml`: containerised, Excel-free deployment
- `eval/eval_vs_dreampy.py`: two-mode cross-validation harness against IFPRI DREAMpy v2.2.3
- `CITATION.cff`: academic citation metadata
- GitHub Actions CI: Python 3.9 / 3.11 / 3.12 matrix

### Implementation notes
- Derived from first principles: Alston-Norton-Pardey (1995) Ch. 4–5 and Wood et al. (2001)
- Cross-validated against IFPRI DREAMpy v2.2.3 on four canonical test vectors; divergence < 0.5 %
- Zero runtime dependencies; Python ≥ 3.9; `openpyxl` optional for DREAMpy eval mode only
