"""
CREAMpy full pipeline examples.
================================
Shows both execution paths and a side-by-side comparison.

Run: python examples/full_pipeline.py
"""

from creampy import ModelParams
from creampy.adoption import BassModel, BassParams
from creampy.adoption.pipeline import Pipeline, to_dreampy_table, to_dreampy_csv


# ── Shared parameters ─────────────────────────────────────────────────────────

BASS = BassParams(
    p       = 0.01,                        # low external influence (extension)
    q       = 0.40,                        # moderate peer learning
    ceiling = 0.70,                        # 70% structural ceiling
    ptrs    = 0.80,                        # 80% probability technology succeeds
    t0      = 2027,                        # first year of market availability
    years   = list(range(2025, 2046)),
)

# ModelParams WITHOUT years/adoption_fracs — Pipeline fills those from Bass.
MODEL = ModelParams(
    K             = 0.13,          # from k_from_yield_gain(0.15)
    epsilon       = 0.50,
    eta           = -0.50,
    P0            = 200.0,         # USD/tonne base price
    Q0            = 1_000_000.0,   # tonnes base production
    discount_rate = 0.05,
    base_year     = 2025,
    scenario      = "central",
)


# ── Example 1: PATH A — fully in Python ──────────────────────────────────────

def example_path_a():
    """Run Bass diffusion + DREAM welfare entirely within CREAMpy."""
    print("=== PATH A: CREAMpy full pipeline (Bass + DREAM welfare) ===\n")

    result = Pipeline(BASS, MODEL).run()

    b = result.bass
    w = result.welfare

    print(f"Bass adoption curve")
    print(f"  p={b.params.p}, q={b.params.q}, "
          f"ceiling={b.params.ceiling}, ptrs={b.params.ptrs}")
    print(f"  Launch year: {b.params.t0}")
    print(f"  Peak new adoption: {b.peak_new_frac:.3%} in {b.peak_year}")
    print(f"  Adoption at year 10 ({b.params.t0+9}): "
          f"{next(r.risk_adj_frac for r in b.year_results if r.year == b.params.t0+9):.3%}")
    print()
    print(f"DREAM welfare")
    print(f"  K={w.params.K:.4f}  epsilon={w.params.epsilon}  eta={w.params.eta}")
    print(f"  NPV welfare:           USD {w.npv_W:>14,.0f}")
    print(f"    Producer surplus:    USD {w.npv_PS:>14,.0f}")
    print(f"    Consumer surplus:    USD {w.npv_CS:>14,.0f}")
    print(f"  run_uid: {w.run_uid}")
    print()
    print(f"  {'Year':4}  {'Adoption':8}  {'dPS (USD)':>16}  {'dCS (USD)':>16}")
    for r in w.year_results[::3]:      # every 3rd year
        print(f"  {r.year:4d}  {r.adoption:8.3%}  {r.dPS:>16,.0f}  {r.dCS:>16,.0f}")
    print()


# ── Example 2: PATH B — Bass here, DREAMpy for welfare ────────────────────────

def example_path_b():
    """Generate adoption schedule for DREAMpy's Excel welfare model."""
    print("=== PATH B: Bass adoption export for DREAMpy ===\n")

    bass_result = BassModel(BASS).run()

    # Table for pasting into DREAMpy Excel template
    table = to_dreampy_table(bass_result, risk_adjusted=True)
    print("Adoption schedule (risk-adjusted) — paste into DREAMpy 'Adoption' sheet:")
    print(f"  {'Year':4}  {'Adoption':8}")
    for year, frac in list(table.items())[2:15]:    # years around the ramp
        print(f"  {year:4d}  {frac:.4f}")
    print(f"  ... ({len(table)} years total)\n")

    # CSV export
    csv_path = "adoption_for_dreampy.csv"
    to_dreampy_csv(bass_result, csv_path)
    print(f"Full adoption schedule written to: {csv_path}")
    print("  Import into DREAMpy v2.2.3 Closed Economy template:")
    print("  1. Open your DREAMpy .xlsm template")
    print("  2. Navigate to the adoption schedule sheet")
    print("  3. Paste Column A (year) and Column C (adoption_risk_adjusted)")
    print("     OR paste Column B (adoption_raw) and set DREAMpy's ptrs"
          f"= {BASS.ptrs}")
    print()
    print("After DREAMpy runs the welfare model, compare NPV_W with Path A result")
    print("(expected divergence < 2%, from arithmetic rounding differences).")
    print()


# ── Example 3: Parameter sensitivity — ceiling and ptrs ───────────────────────

def example_sensitivity():
    """Show NPV sensitivity to ceiling and ptrs assumptions."""
    print("=== Sensitivity: ceiling x ptrs impact on NPV_W ===\n")
    print(f"  {'ceiling':8}  {'ptrs':6}  {'Peak year':10}  {'NPV_W (USD)':>16}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*10}  {'-'*16}")

    for ceiling in [0.50, 0.65, 0.80]:
        for ptrs in [0.60, 0.80, 1.00]:
            bass = BassParams(p=BASS.p, q=BASS.q, ceiling=ceiling, ptrs=ptrs,
                              t0=BASS.t0, years=BASS.years)
            res = Pipeline(bass, MODEL).run()
            print(f"  {ceiling:8.0%}  {ptrs:6.0%}  "
                  f"{res.bass.peak_year:10d}  {res.welfare.npv_W:>16,.0f}")
    print()


if __name__ == "__main__":
    example_path_a()
    example_path_b()
    example_sensitivity()
