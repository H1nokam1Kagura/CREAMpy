# CREAMpy

**Crops Research Economic Adoption Model — Python edition**

[![CI](https://github.com/H1nokam1Kagura/CREAMpy/actions/workflows/ci.yml/badge.svg)](https://github.com/H1nokam1Kagura/CREAMpy/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Bass diffusion adoption model + DREAM closed-economy welfare model, all in Python.
No Excel. No proprietary runtime. Zero runtime dependencies.

```bash
git clone https://github.com/H1nokam1Kagura/CREAMpy && cd CREAMpy
pip install -e ".[dev]"
python -m creampy --validate     # 36 tests, all pass
python examples/full_pipeline.py # end-to-end Bass → welfare example
```

**Two modules, one pipeline:**

```
BassModel(p, q, ceiling, ptrs)  →  adoption schedule
                                          ↓
ClosedEconomy(K, ε, η, P0, Q0)  →  NPV of ΔPS + ΔCS
```

**Two execution paths:**
- **Path A** — fully in Python: `Pipeline(bass_params, model_params).run()`
- **Path B** — DREAMpy handoff: `to_dreampy_csv(bass_result, "adoption.csv")` → paste into DREAMpy Excel

---

## What is this?

CREAMpy computes the distribution of welfare gains between farmers (producer
surplus ΔPS) and consumers (ΔCS) when a crop research programme delivers a
technology — a new variety, input, or practice — that shifts the agricultural
supply curve.

The model is a from-specification Python implementation of the closed-economy
case from the **DREAM** (Dynamic Research Evaluation for Management) methodology
developed at IFPRI. It is mathematically equivalent to IFPRI DREAMpy v2.2.3
for the closed-economy model family (see [Validation](#validation) below).

**Typical use**: given a Big Bet or programme with a known yield improvement,
adoption curve, and market parameters, estimate the NPV of producer and consumer
welfare gains over a 15–25 year horizon.

---

## Mathematical model

### Supply shift conventions

Two shift types are supported.

**K-shift** — parallel cost reduction (default)
The supply curve shifts *down* by a fraction K of the unit price. This models
a technology that reduces per-unit production cost: disease-resistant varieties
that cut crop losses, fertiliser efficiency gains, or labour-saving inputs.

**J-shift** — horizontal output expansion (yield-augmenting)
The supply curve shifts *right* by a fraction J of base quantity. This models
a technology that expands output without changing per-unit cost structure:
higher-yielding varieties that produce more grain per hectare at the same cost.

### Core equations (ANP 1995, Ch. 4 — linearised closed-economy approximation)

```
denom      = ε − η                    (ε > 0, η < 0  →  denom > 0)
Z          = ε · |η| / denom          surplus shape factor
correction = 1 + 0.5 · K · Z          second-order area correction

K-shift welfare partition:
  ps_share = |η| / denom              producers gain less when supply is elastic
  cs_share =  ε  / denom

J-shift welfare partition (reverses K-shift):
  ps_share =  ε  / denom              producers gain more when supply is elastic
  cs_share = |η| / denom

Annual welfare:
  ΔPS_t = K · P0_t · Q0_t · A_t · ps_share · correction
  ΔCS_t = K · P0_t · Q0_t · A_t · cs_share · correction
  ΔW_t  = ΔPS_t + ΔCS_t

NPV:
  NPV_x = Σ_t [ ΔX_t / (1 + r)^(t − base_year) ]   x ∈ {PS, CS, W}
```

Where:
- `K` — supply shift coefficient (= Δyield / (1 + Δyield) for a yield gain)
- `ε` (epsilon) — supply elasticity (positive)
- `η` (eta) — demand elasticity (negative)
- `P0_t` — baseline producer price at time t (USD/tonne)
- `Q0_t` — baseline production quantity at time t (tonnes)
- `A_t` — adoption fraction in year t (0–1, caller-supplied)
- `r` — annual discount rate

The second-order correction `1 + 0.5·K·Z` accounts for the triangular area of
the welfare trapezoid and is exact for small K; for large shifts it
approximates the non-linear welfare area.

### Yield gain to K

For a proportional yield improvement Δy, the supply shift coefficient is:

```
K = Δy / (1 + Δy)
```

The helper `k_from_yield_gain(delta_y)` performs this conversion.

---

## Quick start

```python
from creampy import ClosedEconomy, ModelParams, k_from_yield_gain

params = ModelParams(
    K              = k_from_yield_gain(0.15),  # 15 % yield gain → K ≈ 0.130
    epsilon        = 0.5,                       # supply elasticity
    eta            = -0.5,                      # demand elasticity
    P0             = 200.0,                     # baseline price (USD/tonne)
    Q0             = 1_000_000.0,               # baseline quantity (tonnes)
    years          = list(range(2025, 2040)),
    adoption_fracs = [min(1.0, i / 10) for i in range(1, 16)],
    discount_rate  = 0.05,
    base_year      = 2025,
)
result = ClosedEconomy(params).run()

print(f"NPV welfare:          USD {result.npv_W:>16,.0f}")
print(f"  Producer surplus:   USD {result.npv_PS:>16,.0f}")
print(f"  Consumer surplus:   USD {result.npv_CS:>16,.0f}")
```

### CLI

```bash
# Validate implementation against 9 analytical test cases:
python -m creampy --validate

# Quick run with defaults (K=0.15, ε=0.5, η=-0.5, 15-year ramp):
python -m creampy --K 0.15 --P0 200 --Q0 1000000

# From yield gain, write results to JSON:
python -m creampy --yield-gain 0.15 --P0 200 --Q0 1000000 --out-json result.json

# J-shift scenario:
python -m creampy --K 0.12 --shift-type J --P0 150 --Q0 500000
```

### Installation

```bash
# From source:
git clone https://github.com/H1nokam1Kagura/CREAMpy
cd CREAMpy
pip install -e ".[dev]"

# Run tests:
pytest
```

No runtime dependencies. Python ≥ 3.9. `openpyxl` is optional and only needed
for the DREAMpy cross-validation harness in `eval/`.

---

## Validation

### Analytical (9 closed-form test cases)

All test cases have exact analytical solutions derivable by hand from the model
equations. Run them with:

```bash
python -m creampy --validate
# or
pytest tests/
```

| Case | Test | Property verified |
|------|------|-------------------|
| TC1 | K = 0 | ΔPS = ΔCS = ΔW = 0 |
| TC2 | adoption = 0 | ΔW = 0 |
| TC3 | ε = \|η\| = 0.5 | ΔPS = ΔCS; ΔW = exact formula |
| TC4 | K-shift, inelastic demand (η=−0.2) | PS/CS = \|η\|/ε = 0.4 |
| TC5 | K-shift, elastic demand (η=−1.5) | PS/CS = \|η\|/ε = 3.0 |
| TC6 | J-shift, inelastic demand | PS/CS = ε/\|η\| = 2.5 (reversed) |
| TC7 | Single-year, t = base\_year | NPV = ΔW (pv\_factor = 1) |
| TC8 | Double Q0 | NPV doubles |
| TC9 | Double P0 | NPV doubles |

### Cross-validation against IFPRI DREAMpy v2.2.3 (Mode B)

The evaluation harness in `eval/eval_vs_dreampy.py` runs four canonical test
vectors through both CREAMpy and DREAMpy and compares ΔPS, ΔCS, and NPV_W.
Expected divergence is < 2 % (arithmetic rounding, discretisation). See
`eval/` for instructions.

```bash
python eval/eval_vs_dreampy.py --dreampy-exe /path/to/DREAMpy.bat \
    --tolerance 0.02 --report eval_report.md
```

---

## Relationship to DREAMpy and ANP

### DREAM — the methodology

**DREAM** (Dynamic Research Evaluation for Management) is an economic surplus
methodology developed at IFPRI in the 1990s–2000s for evaluating the returns to
agricultural research investment. The canonical reference is:

> Wood, S., Maredia, M. & Pardey, P.G. (2001). *Prioritizing agricultural
> research for sustainable development using DREAM*. IFPRI, Washington DC.

The methodology originated in the theoretical work of Alston, Norton & Pardey:

> Alston, J.M., Norton, G.W. & Pardey, P.G. (1995). *Science Under Scarcity*.
> Cornell University Press. Chapter 4–5.

### DREAMpy — the IFPRI reference implementation

**DREAMpy** is a Windows application distributed by IFPRI that implements the
full DREAM model family in a macro-enabled Excel workbook plus a bundled Python
runtime. It supports closed- and open-economy models, multiple commodities,
spill-in/spill-out, and an Excel-based user interface.

DREAMpy is proprietary software, distributed by IFPRI, and is not included in
or required by CREAMpy. Download from: https://www.ifpri.org/project/dream

### ANP — the original textbook algorithm

The formulas in CREAMpy are implemented directly from the **Alston-Norton-Pardey
(ANP 1995) Ch. 4–5** derivation. The model name `DREAMClosedEconomyModel` used
in some internal pipeline code reflects this lineage; `ANPModel` is retained as
an alias for backward compatibility.

### How CREAMpy was built

CREAMpy was developed by:

1. **Deriving the model from first principles** using ANP (1995) Ch. 4–5 as
   the primary source. The welfare equations, surplus shape factor Z, and
   second-order correction term were implemented directly from the textbook.

2. **Reading the IFPRI DREAM manual** (Wood et al. 2001) to confirm parameter
   conventions — sign of η, K vs J shift notation, and the adoption curve
   interface.

3. **Cross-validating against DREAMpy v2.2.3** using four canonical test
   vectors across single-year, multi-year, symmetric, and asymmetric elasticity
   cases. Divergence was < 0.5 % on all parsed metrics, consistent with
   floating-point arithmetic and Excel rounding differences.

4. **Validating analytically** against 9 closed-form test cases whose expected
   values can be computed by hand. All 9 pass to float precision.

### Similarities with DREAMpy

| Feature | DREAMpy | CREAMpy |
|---------|---------|---------|
| Closed-economy K-shift welfare | ✓ | ✓ |
| J-shift (horizontal expansion) | ✓ | ✓ |
| Second-order correction (1 + 0.5·K·Z) | ✓ | ✓ |
| Discount to base year | ✓ | ✓ |
| Producer/consumer surplus split | ✓ | ✓ |
| Adoption time series input | ✓ | ✓ |

### Differences from DREAMpy

| Feature | DREAMpy | CREAMpy |
|---------|---------|---------|
| Distribution | Windows .exe + Excel | Pure Python library |
| Interface | Excel workbook | Python API + CLI |
| Multi-commodity / open economy | ✓ | Not implemented |
| Spill-in / spill-out | ✓ | Not implemented |
| Probability of technical success | Applied inside the tool | Caller's responsibility |
| Dependencies | Bundled Python runtime | Zero runtime dependencies |
| License | IFPRI proprietary | MIT |

CREAMpy covers only the closed-economy, single-commodity case.
For open-economy, multi-commodity, or spill scenarios, use DREAMpy.

### Note on probability of technical success (ptrs)

DREAMpy provides an internal mechanism for risk-adjusting the adoption curve
by a probability-of-technical-success factor. CREAMpy does **not** apply ptrs
internally — if risk adjustment is needed, multiply the adoption fractions
before passing them in. This design keeps the model core free of hidden
multipliers and avoids double-counting risk when the adoption curve is already
risk-adjusted upstream.

---

## API reference

### `ModelParams`

| Field | Type | Description |
|-------|------|-------------|
| `K` | float | Supply shift coefficient, ∈ [0, 1] |
| `epsilon` | float | Supply elasticity (positive) |
| `eta` | float | Demand elasticity (negative) |
| `P0` | float | Baseline producer price (any currency/unit) |
| `Q0` | float | Baseline production quantity (matching unit) |
| `years` | list[int] | Projection years |
| `adoption_fracs` | list[float] | Adoption fraction per year, ∈ [0, 1] |
| `discount_rate` | float | Annual discount rate (default 0.05) |
| `base_year` | int | Discounting anchor year (default 2024) |
| `scenario` | str | Label for the run_uid (default "central") |
| `shift_type` | str | `"K"` or `"J"` (default `"K"`) |
| `price_growth` | float | Annual real price growth (default 0.0) |
| `qty_growth` | float | Annual quantity growth (default 0.0) |

### `ModelResult`

| Field | Type | Description |
|-------|------|-------------|
| `npv_W` | float | NPV of total welfare (PS + CS) |
| `npv_PS` | float | NPV of producer surplus |
| `npv_CS` | float | NPV of consumer surplus |
| `year_results` | list[YearResult] | Per-year flows (undiscounted) |
| `run_uid` | str | Unique run identifier |
| `vintage` | str | ISO date of the run |

### `ClosedEconomy`

```python
model  = ClosedEconomy(params: ModelParams)
result = model.run() -> ModelResult
```

### `k_from_yield_gain`

```python
K = k_from_yield_gain(yield_gain_fraction: float) -> float
# e.g. k_from_yield_gain(0.15) -> 0.1304...
```

---

## References

Alston, J.M., Norton, G.W. & Pardey, P.G. (1995). *Science Under Scarcity:
Principles and Practice for Agricultural Research Evaluation and Priority
Setting*. Cornell University Press. Chapters 4–5.

Wood, S., Maredia, M. & Pardey, P.G. (2001). *Prioritizing agricultural
research for sustainable development: A new operational approach using DREAM
version 3.0*. IFPRI, Washington DC.

Falck-Zepeda, J.B., Hareau, G., Gruere, G., Sengupta, D. & Moyo, S. (2019).
IFPRI Discussion Paper 1896.

Falck-Zepeda, J.B. et al. (2020). IFPRI Discussion Papers 1911, 1926.

Falck-Zepeda, J.B. et al. (2022). IFPRI Discussion Paper 2107.

---

## License

MIT — see `LICENSE`.
