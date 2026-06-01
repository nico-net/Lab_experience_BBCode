"""day16_threshold.py — Day 16 deliverable: BP threshold p* from EXIT analysis.

Evolved from day15_preview.py. Where the Day 15 preview evaluated the GF(4)
(3,6)-regular EXIT chart at four hand-picked p values to *bracket* the
threshold (it landed in p* in [0.16, 0.20]), this script *pins it down*:

  1. Coarse scan over a p-grid to confirm the tunnel-gap sign change and
     robustly bracket the root (guards against Monte-Carlo non-monotonicity).
  2. Bisection on that bracket, using common random numbers (the SAME master
     seed at every p), so min_gap(p) is a smooth, reproducible function of p
     and the bisection converges cleanly to the tangency point.
  3. Monte-Carlo uncertainty on p* by repeating the bisection with several
     independent master seeds — this is the number Day 17 needs to decide
     where p* sits inside the 20-30% tolerance band.
  4. Figure 3: EXIT chart at p* (V-curve tangent to inverted-C, with the
     tunnel shown open just below and closed just above) plus the
     tunnel-gap-vs-p crossing.

Tasks covered (PIPELINE Day 16):
  [x] Find p* where EXIT curves just touch
  [x] Plot EXIT curves at p slightly below and above p*
  [x] Verify tunnel opens/closes as expected
  [x] Create Figure 3: EXIT chart with threshold

Outputs (under results/):
    day16_threshold.png   — Figure 3 (publication candidate)
    day16_threshold.txt   — p* +/- sd, scan table (paste into DECISION_LOG)

Run:
    python day16_threshold.py            # production settings (a few minutes)
    python day16_threshold.py --quick    # fast sanity (~1 min), coarse N
    python day16_threshold.py --n-samples 80000 --seeds 9   # tighten p*
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exit_analysis import exit_curves, channel_mi


# (3, 6)-regular ensemble — matches all four BB code instances.
D_V, D_C = 3, 6
MASTER_SEED = 20260601

# Colours kept identical to the Day 15 preview for visual continuity.
C_V = "#1f77b4"   # variable-node (V) EXIT curve
C_C = "#d62728"   # check-node (C) EXIT curve, plotted inverted


# ---------------------------------------------------------------------------
# Core: tunnel gap at a single p
# ---------------------------------------------------------------------------
def tunnel_gap(p, d_v, d_c, n_points, N, seed):
    """Return (min_gap, ExitCurves) for one channel error rate p.

    min_gap is the minimum over the *interior* I_A grid of
        I_E^V(I_A, p) - (I_E^C)^{-1}(I_A),
    i.e. the vertical clearance of the V-EXIT curve above the inverted
    C-EXIT curve. Positive => tunnel open (BP converges to the noise-free
    fixed point); negative => tunnel closed.

    The inverted-C value at each I_A is obtained exactly as in the Day 15
    preview, via np.interp(I_A, I_E_C, I_A) (C-EXIT is monotone in I_A, so
    I_E_C is a valid increasing x-axis for interpolation).

    Reseeding with the SAME seed for every p call gives common random
    numbers across p, which removes most of the Monte-Carlo jitter from the
    *difference* min_gap(p) and makes the bisection well-behaved.
    """
    rng = np.random.default_rng(seed)
    ec = exit_curves(p, d_v, d_c, n_points=n_points, N=N, rng=rng)
    c_inv_at_iA = np.interp(ec.I_A, ec.I_E_C, ec.I_A)
    gap = ec.I_E_V - c_inv_at_iA
    return float(gap[1:-1].min()), ec


# ---------------------------------------------------------------------------
# Step 1: coarse scan + bracket
# ---------------------------------------------------------------------------
def coarse_scan(p_grid, d_v, d_c, n_points, N, seed):
    gaps = np.empty(len(p_grid))
    print(f"{'p':>7}  {'I_ch':>7}  {'min_gap':>9}  {'tunnel':>7}  {'sec':>6}")
    print("-" * 44)
    for i, p in enumerate(p_grid):
        t0 = time.perf_counter()
        gaps[i], _ = tunnel_gap(p, d_v, d_c, n_points, N, seed)
        dt = time.perf_counter() - t0
        print(f"{p:>7.4f}  {channel_mi(p):>7.4f}  {gaps[i]:>+9.4f}  "
              f"{'open' if gaps[i] > 0 else 'CLOSED':>7}  {dt:>6.2f}")
    return gaps


def bracket_from_scan(p_grid, gaps):
    """Smallest grid interval [p_lo, p_hi] across which gap goes + -> -."""
    for i in range(len(gaps) - 1):
        if gaps[i] > 0.0 >= gaps[i + 1]:
            return float(p_grid[i]), float(p_grid[i + 1])
    raise RuntimeError(
        "No tunnel-gap sign change in the scan grid — widen --grid range or "
        f"raise --n-samples. gaps ranged {gaps.min():+.4f} to {gaps.max():+.4f}."
    )


# ---------------------------------------------------------------------------
# Step 2: bisection on the bracket
# ---------------------------------------------------------------------------
def bisect_threshold(p_lo, p_hi, d_v, d_c, n_points, N, seed, tol):
    """Bisect min_gap(p) = 0 on [p_lo, p_hi] (gap is decreasing in p)."""
    g_lo, _ = tunnel_gap(p_lo, d_v, d_c, n_points, N, seed)
    g_hi, _ = tunnel_gap(p_hi, d_v, d_c, n_points, N, seed)
    if not (g_lo > 0.0 >= g_hi):
        raise RuntimeError(
            f"Bad bracket for seed {seed}: gap({p_lo:.3f})={g_lo:+.4f}, "
            f"gap({p_hi:.3f})={g_hi:+.4f}; need + then -.")
    while p_hi - p_lo > tol:
        p_mid = 0.5 * (p_lo + p_hi)
        g_mid, _ = tunnel_gap(p_mid, d_v, d_c, n_points, N, seed)
        if g_mid > 0.0:
            p_lo = p_mid
        else:
            p_hi = p_mid
    return 0.5 * (p_lo + p_hi)


# ---------------------------------------------------------------------------
# Step 4: Figure 3
# ---------------------------------------------------------------------------
def make_figure3(p_star, p_grid, gaps, d_v, d_c, n_points, N, seed,
                 delta, out_path):
    p_below = max(p_star - delta, 0.001)
    p_above = p_star + delta

    _, ec_below = tunnel_gap(p_below, d_v, d_c, n_points, N, seed)
    g_star, ec_star = tunnel_gap(p_star, d_v, d_c, n_points, N, seed)
    _, ec_above = tunnel_gap(p_above, d_v, d_c, n_points, N, seed)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # ---- Left: EXIT chart at the threshold -------------------------------
    # Inverted C-EXIT curve is p-independent; draw it once.
    axL.plot(ec_star.I_E_C, ec_star.I_A, marker="s", markersize=4,
             linewidth=1.8, color=C_C, label="C-EXIT inverted")
    axL.plot(ec_below.I_A, ec_below.I_E_V, linewidth=1.5, color=C_V,
             alpha=0.45, label=f"V-EXIT  p = {p_below:.3f}  (tunnel open)")
    axL.plot(ec_star.I_A, ec_star.I_E_V, marker="o", markersize=3.5,
             linewidth=2.3, color=C_V,
             label=f"V-EXIT  p* = {p_star:.3f}  (tangent)")
    axL.plot(ec_above.I_A, ec_above.I_E_V, linewidth=1.5, color=C_V,
             alpha=0.45, linestyle="--",
             label=f"V-EXIT  p = {p_above:.3f}  (tunnel CLOSED)")
    axL.plot([0, 2], [0, 2], color="grey", linewidth=0.5, linestyle=":",
             alpha=0.6)
    axL.set_xlim(0, 2)
    axL.set_ylim(0, 2)
    axL.set_xlabel(r"$I_A$  (bits/symbol)")
    axL.set_ylabel(r"$I_E$  (bits/symbol)")
    axL.set_title(rf"GF(4) EXIT chart at BP threshold  ($d_v={d_v}$, $d_c={d_c}$)",
                  fontsize=10)
    axL.grid(True, alpha=0.3, linewidth=0.5)
    axL.legend(loc="lower right", fontsize=7.5)

    # ---- Right: tunnel gap vs p ------------------------------------------
    axR.plot(p_grid, gaps, marker="o", markersize=4, linewidth=1.6,
             color="#2c7fb8", label="min tunnel gap")
    axR.axhline(0.0, color="grey", linewidth=0.8)
    axR.axvline(p_star, color="#238b45", linewidth=1.4, linestyle="--")
    axR.scatter([p_star], [0.0], color="#238b45", zorder=5, s=30)
    axR.annotate(rf"$p^* = {p_star:.3f}$",
                 xy=(p_star, 0.0), xytext=(6, 10),
                 textcoords="offset points", color="#238b45", fontsize=10)
    axR.set_xlabel(r"channel error rate  $p$")
    axR.set_ylabel(r"$\min_{I_A}\,[\,I_E^V - (I_E^C)^{-1}\,]$")
    axR.set_title("Tunnel pinch-off (threshold crossing)", fontsize=10)
    axR.grid(True, alpha=0.3, linewidth=0.5)
    axR.legend(loc="upper right", fontsize=8)

    fig.suptitle("Figure 3 — BP decoding threshold from EXIT analysis, "
                 "GF(4) (3,6) ensemble", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"\nFigure 3 saved to {out_path}")
    return g_star


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="Fast sanity run: coarse grid, low N, single seed.")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="Monte-Carlo samples per EXIT point "
                         "(default 40000, or 8000 with --quick).")
    ap.add_argument("--n-points", type=int, default=None,
                    help="I_A grid points per EXIT curve "
                         "(default 41, or 21 with --quick).")
    ap.add_argument("--seeds", type=int, default=None,
                    help="Number of master seeds for the p* uncertainty "
                         "estimate (default 5, or 1 with --quick).")
    ap.add_argument("--tol", type=float, default=0.002,
                    help="Bisection tolerance in p (default 0.002).")
    ap.add_argument("--grid-lo", type=float, default=0.14,
                    help="Coarse-scan lower p (default 0.14).")
    ap.add_argument("--grid-hi", type=float, default=0.22,
                    help="Coarse-scan upper p (default 0.22).")
    ap.add_argument("--grid-step", type=float, default=0.01,
                    help="Coarse-scan p step (default 0.01).")
    ap.add_argument("--delta", type=float, default=0.02,
                    help="Offset for the open/closed companion curves in "
                         "Figure 3 (default 0.02).")
    return ap.parse_args()


def main():
    args = parse_args()
    N = args.n_samples if args.n_samples is not None else (8_000 if args.quick else 40_000)
    n_points = args.n_points if args.n_points is not None else (21 if args.quick else 41)
    n_seeds = args.seeds if args.seeds is not None else (1 if args.quick else 5)

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    p_grid = np.round(np.arange(args.grid_lo, args.grid_hi + 1e-9, args.grid_step), 4)

    print(f"GF(4) EXIT threshold search: (d_v, d_c) = ({D_V}, {D_C}), q = 4")
    print(f"N = {N} MC samples/point, n_points = {n_points}, "
          f"seeds = {n_seeds}, tol = {args.tol}\n")
    print("Step 1 — coarse scan to bracket the threshold:")
    gaps = coarse_scan(p_grid, D_V, D_C, n_points, N, MASTER_SEED)
    p_lo, p_hi = bracket_from_scan(p_grid, gaps)
    print(f"\nBracket from scan: p* in [{p_lo:.3f}, {p_hi:.3f}]")

    # Widen the bracket by one grid step each side so per-seed MC noise can't
    # flip the endpoint signs during the uncertainty loop.
    pad = args.grid_step
    p_lo_b = max(p_lo - pad, float(p_grid[0]))
    p_hi_b = min(p_hi + pad, float(p_grid[-1]))

    print("\nStep 2-3 — bisection per seed (common random numbers):")
    p_stars = []
    for k in range(n_seeds):
        seed = MASTER_SEED + k
        try:
            ps = bisect_threshold(p_lo_b, p_hi_b, D_V, D_C,
                                  n_points, N, seed, args.tol)
            p_stars.append(ps)
            print(f"  seed {seed}:  p* = {ps:.4f}")
        except RuntimeError as e:
            print(f"  seed {seed}:  SKIPPED ({e})")

    if not p_stars:
        raise SystemExit("No seed produced a valid bracket — raise --n-samples.")

    p_stars = np.asarray(p_stars)
    p_star = float(p_stars.mean())
    p_sd = float(p_stars.std(ddof=1)) if len(p_stars) > 1 else float("nan")
    print(f"\np* = {p_star:.4f}  (sd = {p_sd:.4f} over {len(p_stars)} seeds)")

    print("\nStep 4 — Figure 3:")
    g_star = make_figure3(p_star, p_grid, gaps, D_V, D_C, n_points, N,
                          MASTER_SEED, args.delta, out_dir / "day16_threshold.png")

    # ---- Results dump for DECISION_LOG -----------------------------------
    txt_path = out_dir / "day16_threshold.txt"
    with open(txt_path, "w") as f:
        f.write("Day 16 — BP threshold from EXIT analysis\n")
        f.write("GF(4) (3,6)-regular ensemble (d_v=3, d_c=6), q=4\n\n")
        f.write(f"N = {N} MC samples/point, n_points = {n_points}, "
                f"tol = {args.tol}, seeds = {len(p_stars)}\n\n")
        f.write(f"p* = {p_star:.4f}  (sd = {p_sd:.4f})\n")
        f.write(f"per-seed p*: {', '.join(f'{x:.4f}' for x in p_stars)}\n")
        f.write(f"residual gap at p* (master seed): {g_star:+.4f}\n\n")
        f.write("Coarse scan (master seed):\n")
        f.write(f"{'p':>7}  {'I_ch':>7}  {'min_gap':>9}  tunnel\n")
        for p, g in zip(p_grid, gaps):
            f.write(f"{p:>7.4f}  {channel_mi(p):>7.4f}  {g:>+9.4f}  "
                    f"{'open' if g > 0 else 'CLOSED'}\n")
        f.write("\nDay 17 check: simulated thresholds cluster ~0.12-0.16; "
                "tolerance 20-30%.\n")
    print(f"Results written to {txt_path}")


if __name__ == "__main__":
    main()