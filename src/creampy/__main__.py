"""CLI entry point: python -m creampy [options]"""

from __future__ import annotations

import argparse
import json
import sys

from .model import ClosedEconomy, ModelParams, __version__, k_from_yield_gain


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="creampy",
        description=(
            "CREAMpy -- DREAM Closed Economy partial-equilibrium surplus model.\n"
            "Computes producer surplus (dPS), consumer surplus (dCS), and their\n"
            "NPV from a research-induced supply shift."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--validate", action="store_true",
                   help="Run 9 closed-form analytical test cases and exit (no other args needed).")
    p.add_argument("--k", type=float, default=None,
                   help="Supply shift coefficient in [0, 1]. "
                        "Use --yield-gain as an alternative.")
    p.add_argument("--yield-gain", type=float, default=None,
                   help="Proportional yield gain (e.g. 0.15 for 15%%). "
                        "Converted to k via k = gy/(1+gy).")
    p.add_argument("--epsilon", type=float, default=0.5,
                   help="Supply elasticity, positive (default 0.5).")
    p.add_argument("--eta", type=float, default=-0.5,
                   help="Demand elasticity, negative (default -0.5).")
    p.add_argument("--price", type=float, default=200.0,
                   help="Baseline producer price P0, any currency/tonne (default 200).")
    p.add_argument("--qty", type=float, default=1_000_000.0,
                   help="Baseline production quantity Q0, tonnes (default 1 000 000).")
    p.add_argument("--start-year",    type=int,   default=2025)
    p.add_argument("--end-year",      type=int,   default=2040)
    p.add_argument("--ramp-years",    type=int,   default=10,
                   help="Years to ramp from zero to full adoption (default 10). "
                        "Must be >= 1.")
    p.add_argument("--discount-rate", type=float, default=0.05,
                   help="Annual discount rate in [0, 1) (default 0.05). "
                        "Use 0 for undiscounted sum.")
    p.add_argument("--base-year",     type=int,   default=2025,
                   help="Discounting anchor year (default 2025). "
                        "Flows in this year have pv_factor=1.")
    p.add_argument("--shift-type",    choices=["K", "J"], default="K",
                   help="Supply shift convention: K=cost-reduction (default), "
                        "J=yield-augmenting output expansion.")
    p.add_argument("--scenario",      default="central",
                   help="Scenario label attached to run_uid (default 'central').")
    p.add_argument("--out-json",      default=None,
                   help="Write full results JSON to this path instead of printing.")
    p.add_argument("--version",       action="version", version=__version__)
    return p.parse_args(argv)


def _make_adoption(start: int, end: int, ramp_years: int) -> tuple[list[int], list[float]]:
    """Build a linear-ramp-then-plateau adoption curve.

    Generates ``ramp_years`` years of linearly increasing adoption (from
    1/ramp_years to 1.0), then holds at 1.0 for the remaining years.

    Raises ValueError if ``ramp_years < 1``.
    """
    if ramp_years < 1:
        raise ValueError(f"ramp_years must be >= 1, got {ramp_years}")
    years = list(range(start, end + 1))
    fracs = [min(1.0, (i + 1) / ramp_years) for i in range(len(years))]
    return years, fracs


def run_validation() -> bool:
    """Nine closed-form analytical test cases with known exact solutions."""
    from .model import ClosedEconomy, ModelParams
    import dataclasses

    passed = failed = 0

    def near(a: float, b: float, tol: float = 1e-6) -> bool:
        return abs(a - b) / max(abs(b), 1.0) < tol

    def check(name: str, got: float, expected: float, tol: float = 1e-6) -> None:
        nonlocal passed, failed
        if near(got, expected, tol):
            print(f"  PASS  {name}")
            passed += 1
        else:
            pct = 100 * abs(got - expected) / max(abs(expected), 1.0)
            print(f"  FAIL  {name}  got={got:.8g}  expected={expected:.8g}  ({pct:.4f}%)")
            failed += 1

    W = ClosedEconomy._welfare
    print("\n=== CREAMpy: 9-case analytical validation ===\n")

    # TC1  K=0 -> all zeros
    dPS, dCS, dW = W(0.0, 0.5, -0.5, 200.0, 1e6, 1.0, "K")
    check("TC1  K=0: dPS=0", dPS, 0.0)
    check("TC1  K=0: dCS=0", dCS, 0.0)
    check("TC1  K=0: dW=0",  dW,  0.0)

    # TC2  adoption=0 -> all zeros
    dPS, dCS, dW = W(0.15, 0.5, -0.5, 200.0, 1e6, 0.0, "K")
    check("TC2  adoption=0: dW=0", dW, 0.0)

    # TC3  symmetric elasticities (eps=|eta|=0.5) -> dPS == dCS, dW exact
    dPS, dCS, dW = W(0.15, 0.5, -0.5, 200.0, 1e6, 1.0, "K")
    check("TC3  symmetric: dPS==dCS", dPS, dCS, tol=1e-9)
    Z_sym = 0.5 * 0.5 / 1.0
    expected_dW = 0.15 * 200.0 * 1e6 * (1.0 + 0.5 * 0.15 * Z_sym)
    check("TC3  dW exact formula", dW, expected_dW)

    # TC4  inelastic demand eta=-0.2: PS/CS = |eta|/eps = 0.2/0.5
    dPS4, dCS4, _ = W(0.01, 0.5, -0.2, 100.0, 1e6, 1.0, "K")
    check("TC4  K-shift inelastic PS/CS = |eta|/eps", dPS4 / dCS4, 0.2 / 0.5, tol=1e-3)

    # TC5  elastic demand eta=-1.5: PS/CS = 1.5/0.5 = 3.0
    dPS5, dCS5, _ = W(0.01, 0.5, -1.5, 100.0, 1e6, 1.0, "K")
    check("TC5  K-shift elastic  PS/CS = |eta|/eps", dPS5 / dCS5, 1.5 / 0.5, tol=1e-3)

    # TC6  J-shift reverses share: PS/CS = eps/|eta|
    dPS6, dCS6, _ = W(0.01, 0.5, -0.2, 100.0, 1e6, 1.0, "J")
    check("TC6  J-shift: PS/CS = eps/|eta|", dPS6 / dCS6, 0.5 / 0.2, tol=1e-3)

    # TC7  single-year NPV at base_year -> pv_factor=1 -> NPV==dW
    p7 = ModelParams(K=0.10, epsilon=0.5, eta=-0.5, P0=100.0, Q0=1e6,
                     years=[2024], adoption_fracs=[1.0],
                     discount_rate=0.05, base_year=2024)
    r7 = ClosedEconomy(p7).run()
    Z7 = 0.5 * 0.5 / 1.0
    expected_npv7 = 0.10 * 100.0 * 1e6 * (1.0 + 0.5 * 0.10 * Z7)
    check("TC7  single-year NPV == dW", r7.npv_W, expected_npv7)

    # TC8  linearity: doubling Q0 doubles NPV
    p8a = ModelParams(K=0.10, epsilon=0.5, eta=-0.5, P0=100.0, Q0=1e6,
                      years=[2024], adoption_fracs=[1.0],
                      discount_rate=0.05, base_year=2024)
    p8b = dataclasses.replace(p8a, Q0=2e6)
    r8a = ClosedEconomy(p8a).run()
    r8b = ClosedEconomy(p8b).run()
    check("TC8  linearity: 2xQ0 -> 2xNPV", r8b.npv_W, 2.0 * r8a.npv_W)

    # TC9  linearity: doubling P0 doubles NPV
    p9  = dataclasses.replace(p8a, P0=200.0)
    r9  = ClosedEconomy(p9).run()
    check("TC9  linearity: 2xP0 -> 2xNPV", r9.npv_W, 2.0 * r8a.npv_W)

    print(f"\n  {passed} passed / {failed} failed\n")
    return failed == 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.validate:
        return 0 if run_validation() else 1

    # Resolve K
    if args.k is not None:
        K = args.k
    elif args.yield_gain is not None:
        K = k_from_yield_gain(args.yield_gain)
    else:
        print("ERROR: supply --k or --yield-gain", file=sys.stderr)
        return 2

    try:
        years, adoption_fracs = _make_adoption(args.start_year, args.end_year, args.ramp_years)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    params = ModelParams(
        K=K, epsilon=args.epsilon, eta=args.eta,
        P0=args.price, Q0=args.qty,
        years=years, adoption_fracs=adoption_fracs,
        discount_rate=args.discount_rate, base_year=args.base_year,
        scenario=args.scenario, shift_type=args.shift_type,
    )
    result = ClosedEconomy(params).run()

    print(f"run_uid : {result.run_uid}")
    print(f"NPV_W   : USD {result.npv_W:>18,.0f}")
    print(f"NPV_PS  : USD {result.npv_PS:>18,.0f}")
    print(f"NPV_CS  : USD {result.npv_CS:>18,.0f}")

    if args.out_json:
        out = {
            "run_uid": result.run_uid,
            "vintage": result.vintage,
            "npv_W":  result.npv_W,
            "npv_PS": result.npv_PS,
            "npv_CS": result.npv_CS,
            "params": {
                "K": params.K, "epsilon": params.epsilon, "eta": params.eta,
                "P0": params.P0, "Q0": params.Q0,
                "shift_type": params.shift_type,
                "discount_rate": params.discount_rate,
                "base_year": params.base_year,
                "scenario": params.scenario,
            },
            "year_results": [
                {"year": r.year, "adoption": r.adoption,
                 "dPS": r.dPS, "dCS": r.dCS, "dW": r.dW}
                for r in result.year_results
            ],
        }
        with open(args.out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"Results written to {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
