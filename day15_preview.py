"""day15_preview.py — sanity-preview the EXIT chart at four p values."""
from __future__ import annotations
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exit_analysis import exit_curves, channel_mi


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    # (3, 6)-regular ensemble — matches all four BB codes.
    d_v, d_c = 3, 6

    # Bracket the simulated threshold (~0.13).
    p_values = [0.05, 0.13, 0.16, 0.20]
    n_pts = 21
    N = 20_000
    rng = np.random.default_rng(20260601)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2), sharey=True, sharex=True)
    print(f"Generating EXIT curves for (d_v, d_c) = ({d_v}, {d_c}), "
          f"q = 4, N = {N} samples per point")
    print(f"{'p':>6}  {'I_ch':>6}  {'V(0)':>6}  {'V(2)':>6}  "
          f"{'C(0)':>6}  {'C(2)':>6}  {'min_gap':>8}  {'tunnel':>7}  {'sec':>6}")
    print("-" * 75)

    for ax, p in zip(axes, p_values):
        t0 = time.perf_counter()
        ec = exit_curves(p, d_v, d_c, n_points=n_pts, N=N, rng=rng)
        elapsed = time.perf_counter() - t0

        # Plot V curve forward and C curve inverted on same axes.
        ax.plot(ec.I_A, ec.I_E_V, marker="o", linewidth=1.8,
                color="#1f77b4", label="V-EXIT  $I_E^V(I_A, p)$")
        # The inverted C curve is plotted with x = I_E^C(I_A) on horizontal
        # and y = I_A on vertical (so we read the role of the C function
        # as "what I_A_c do we need to produce a given I_E_c").
        ax.plot(ec.I_E_C, ec.I_A, marker="s", linewidth=1.8,
                color="#d62728", label="C-EXIT inverted")

        # Tunnel: V curve must stay strictly above inverted-C curve to admit
        # BP convergence. The two curves always meet at (log_2 q, log_2 q),
        # so we check the OPEN interior of I_A ∈ (0, log_2 q).
        c_inv_at_iA = np.interp(ec.I_A, ec.I_E_C, ec.I_A)
        gap = ec.I_E_V - c_inv_at_iA
        tunnel_open = bool(np.all(gap[1:-1] > 0))
        min_gap = float(gap[1:-1].min())
        ax.set_title(f"p = {p:.2f}   "
                     f"({'tunnel open' if tunnel_open else 'tunnel CLOSED'}, "
                     f"min gap = {min_gap:+.3f})",
                     fontsize=10)
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xlabel(r"$I_A$  (bits/symbol)")
        if ax is axes[0]:
            ax.set_ylabel(r"$I_E$  (bits/symbol)")
        ax.plot([0, 2], [0, 2], color="grey", linewidth=0.5,
                linestyle=":", alpha=0.6)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if ax is axes[0]:
            ax.legend(loc="lower right", fontsize=8)

        print(f"{p:>6.3f}  {channel_mi(p):>6.3f}  "
              f"{ec.I_E_V[0]:>6.3f}  {ec.I_E_V[-1]:>6.3f}  "
              f"{ec.I_E_C[0]:>6.3f}  {ec.I_E_C[-1]:>6.3f}  "
              f"{min_gap:>+8.4f}  "
              f"{'open' if tunnel_open else 'CLOSED':>7}  {elapsed:>6.2f}")

    fig.suptitle(r"GF(4) EXIT chart preview: (3, 6)-regular ensemble, "
                 r"$d_v = 3$, $d_c = 6$",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out_dir / "day15_exit_preview.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nPreview figure saved to {out_path}")


if __name__ == "__main__":
    main()