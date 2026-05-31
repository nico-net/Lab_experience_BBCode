"""day13_14_make_figure2.py — Figure 2: BP FER waterfall on all four codes."""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path):
    by_code: dict[str, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            name = row["code_name"]
            if name not in by_code:
                by_code[name] = {
                    "n": int(row["n"]), "k": int(row["k"]),
                    "p": [], "fer": [], "ci_lo": [], "ci_hi": [],
                    "trials": [], "fails": [], "ser": [],
                }
            d = by_code[name]
            d["p"].append(float(row["p"]))
            d["fer"].append(float(row["fer"]))
            d["ci_lo"].append(float(row["fer_ci_low"]))
            d["ci_hi"].append(float(row["fer_ci_high"]))
            d["trials"].append(int(row["num_trials"]))
            d["fails"].append(int(row["num_frame_errors"]))
            d["ser"].append(float(row["ser"]))
    for d in by_code.values():
        for k in ("p", "fer", "ci_lo", "ci_hi", "ser"):
            d[k] = np.asarray(d[k], dtype=float)
        d["trials"] = np.asarray(d["trials"], dtype=int)
        d["fails"] = np.asarray(d["fails"], dtype=int)
    return by_code


def make_figure(by_code: dict, out_path: Path) -> None:
    order = ["tiny", "small", "medium", "large"]
    colors = {"tiny": "#1f77b4", "small": "#2ca02c",
              "medium": "#ff7f0e", "large": "#d62728"}
    markers = {"tiny": "o", "small": "s", "medium": "D", "large": "^"}

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    # For each code, plot FER vs p with asymmetric error bars from Wilson CIs.
    # Zero-fail cells get plotted at the Wilson upper bound as an "≤" upper
    # limit (downward arrow), matching the FER-curve convention in DF Figs 5-9.
    for name in order:
        d = by_code[name]
        n, k = d["n"], d["k"]
        rate = k / n
        fer = d["fer"]
        ci_lo = d["ci_lo"]
        ci_hi = d["ci_hi"]
        # Split into "fer > 0" (plotted as point + error bar) and "fer == 0"
        # (plotted as upper-limit caret at ci_hi).
        meas = fer > 0
        zerr_lo = np.maximum(fer - ci_lo, 1e-12)
        zerr_hi = np.maximum(ci_hi - fer, 1e-12)
        ax.errorbar(
            d["p"][meas], fer[meas],
            yerr=[zerr_lo[meas], zerr_hi[meas]],
            marker=markers[name], color=colors[name], linestyle="-",
            linewidth=1.5, markersize=6, capsize=3,
            label=f"{name}  (n={n}, k={k}, R={rate:.3f})"
        )
        # Upper-limit triangles for zero-fail cells (downward-pointing).
        if (~meas).any():
            ax.scatter(d["p"][~meas], ci_hi[~meas],
                       marker="v", color=colors[name], s=40,
                       facecolors="none", linewidths=1.2)

    # Reference: null-decoder FER  =  1 - (1 - p)^n  per code (dashed thin lines).
    p_smooth = np.logspace(np.log10(0.0008), np.log10(0.32), 100)
    for name in order:
        n = by_code[name]["n"]
        null_fer = 1 - (1 - p_smooth) ** n
        ax.plot(p_smooth, null_fer, color=colors[name],
                linewidth=0.8, alpha=0.35, linestyle="--")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.0008, 0.32)
    ax.set_ylim(5e-4, 1.2)
    ax.set_xlabel("Channel error rate $p$ (QSC)")
    ax.set_ylabel("Frame Error Rate (FER)")
    ax.set_title("BP-FFT decoder, F$_4$-linear BB codes (max_iters=50)")
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    # Footnote-style note in lower-left.
    ax.text(0.0009, 7e-4,
            "▽ : Wilson upper bound for 0 frame errors\n"
            "dashed : null-decoder analytical FER $= 1{-}(1{-}p)^n$",
            fontsize=7.5, color="#444",
            verticalalignment="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Figure written to {out_path} (.png and .pdf)")


def summary_table(by_code: dict) -> str:
    order = ["tiny", "small", "medium", "large"]
    p_values = sorted(by_code[order[0]]["p"])
    lines = [f"{'p':>7}" + "".join(f"  {n:>14}" for n in order)]
    lines.append("-" * (7 + 16 * len(order)))
    for i, p in enumerate(p_values):
        cells = []
        for name in order:
            d = by_code[name]
            fer = d["fer"][i]
            tr = d["trials"][i]
            fails = d["fails"][i]
            cells.append(f"{fer:.4f} ({fails}/{tr})")
        lines.append(f"{p:>7.3f}" + "".join(f"  {c:>14}" for c in cells))
    return "\n".join(lines)


def main():
    here = Path(__file__).resolve().parent
    csv_path = here / "results" / "day13_14_bp_sweep.csv"
    fig_path = here / "results" / "figure2_bp_fer.png"

    by_code = load_csv(csv_path)
    make_figure(by_code, fig_path)

    print()
    print("FER summary table (frame errors / trials):")
    print(summary_table(by_code))


if __name__ == "__main__":
    main()
