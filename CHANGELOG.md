# Changelog

All notable changes to CREAMpy are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
