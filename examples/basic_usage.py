"""
CREAMpy usage examples.

Run any section independently:
  python examples/basic_usage.py
"""

from creampy import ClosedEconomy, ModelParams, k_from_yield_gain


# ── Example 1: Minimal one-year calculation ───────────────────────────────────

def example_one_year():
    """Symmetric market, full adoption, single year — verify by hand."""
    params = ModelParams(
        K             = 0.15,         # 15 % cost reduction
        epsilon       = 0.5,          # supply elasticity
        eta           = -0.5,         # demand elasticity
        P0            = 200.0,        # USD/tonne base price
        Q0            = 1_000_000.0,  # tonnes base production
        years         = [2025],
        adoption_fracs= [1.0],
        discount_rate = 0.05,
        base_year     = 2025,
    )
    result = ClosedEconomy(params).run()
    print("=== Example 1: single year ===")
    yr = result.year_results[0]
    print(f"  ΔPS = USD {yr.dPS:>14,.0f}")
    print(f"  ΔCS = USD {yr.dCS:>14,.0f}")
    print(f"  ΔW  = USD {yr.dW:>14,.0f}")
    print(f"  NPV_W (same, pv_factor=1) = USD {result.npv_W:,.0f}\n")


# ── Example 2: Multi-year projection with S-curve adoption ───────────────────

def example_multi_year():
    """15-year projection with a linear adoption ramp over 10 years."""
    start, end = 2025, 2040
    years = list(range(start, end + 1))
    # Linear ramp reaching full adoption by year 10, then plateau
    adoption = [min(1.0, (i + 1) / 10) for i in range(len(years))]

    params = ModelParams(
        K             = k_from_yield_gain(0.15),   # 15 % yield gain → K ≈ 0.130
        epsilon       = 0.5,
        eta           = -0.5,
        P0            = 200.0,
        Q0            = 1_000_000.0,
        years         = years,
        adoption_fracs= adoption,
        discount_rate = 0.05,
        base_year     = 2025,
        scenario      = "central",
    )
    result = ClosedEconomy(params).run()
    print("=== Example 2: 15-year projection ===")
    print(f"  K (from 15% yield gain) = {params.K:.4f}")
    print(f"  NPV of welfare:           USD {result.npv_W:>14,.0f}")
    print(f"  NPV producer surplus:     USD {result.npv_PS:>14,.0f}")
    print(f"  NPV consumer surplus:     USD {result.npv_CS:>14,.0f}")
    print(f"  run_uid: {result.run_uid}\n")
    print(f"  {'Year':4s}  {'Adoption':8s}  {'ΔPS (USD)':>16s}  {'ΔCS (USD)':>16s}")
    for r in result.year_results:
        print(f"  {r.year:4d}  {r.adoption:8.2f}  {r.dPS:>16,.0f}  {r.dCS:>16,.0f}")
    print()


# ── Example 3: K-shift vs J-shift comparison ─────────────────────────────────

def example_shift_comparison():
    """
    Compare K-shift (cost reduction) vs J-shift (yield augmenting) welfare split.

    With ε=0.5, |η|=0.3:
      K-shift: ps_share = 0.3/0.8 = 0.375  (producers get less)
      J-shift: ps_share = 0.5/0.8 = 0.625  (producers get more)
    """
    common = dict(
        K=0.10, epsilon=0.5, eta=-0.3,
        P0=200.0, Q0=1_000_000.0,
        years=[2025], adoption_fracs=[1.0],
        discount_rate=0.05, base_year=2025,
    )
    k_result = ClosedEconomy(ModelParams(**common, shift_type="K")).run()
    j_result = ClosedEconomy(ModelParams(**common, shift_type="J")).run()

    print("=== Example 3: K-shift vs J-shift ===")
    print(f"  {'':10s}  {'NPV_PS':>14s}  {'NPV_CS':>14s}  {'PS share':>10s}")
    for label, r in [("K-shift", k_result), ("J-shift", j_result)]:
        share = r.npv_PS / r.npv_W
        print(f"  {label:10s}  {r.npv_PS:>14,.0f}  {r.npv_CS:>14,.0f}  {share:>9.1%}")
    print()


if __name__ == "__main__":
    example_one_year()
    example_multi_year()
    example_shift_comparison()
