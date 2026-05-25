"""
day6_fer_curves.py — PIPELINE Days 6-7 deliverable.

Generates FER (and SER, BER) vs. p curves for the Nishimori Metropolis decoder
on all four BB code instances, saving raw data and a draft Figure 2.

Usage
-----
    python day6_fer_curves.py --smoke           # ~30 seconds, tiny only, sanity check
    python day6_fer_curves.py --full            # full sweep, 4 codes × 10 p-values
    python day6_fer_curves.py --code small      # only the named code (full settings)
    python day6_fer_curves.py --trials 2000 --sweeps 300  # override defaults

Outputs (under results/)
------------------------
    day6_metropolis_sweep.json    — full raw EvalResults
    day6_metropolis_sweep.txt     — human-readable table
    day6_metropolis_sweep.csv     — flat CSV for spreadsheet inspection
    day6_metropolis_fer.png       — draft Figure 2 (FER vs p, log y)
    day6_metropolis_ser_ber.png   — auxiliary SER/BER curves

The script is reentrant: per-cell results are streamed to JSON as they finish,
so an interrupted run leaves partial data on disk. Re-running with the same
seed and parameters reproduces the exact numbers cell-for-cell.

Validation acceptance (PIPELINE Day 6-7):
    "Do curves look sensible?" — monotone in p, ordered by code size at
    low-to-moderate p (larger n protects better below threshold), with
    statistically resolved separation between consecutive p-values at the
    waterfall.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from bb_constructor import ALL_INSTANCES, make_tiny, make_small, make_medium, make_large
from evaluation import EvalResult, evaluate
from decoder_metropolis import NishimoriMetropolisDecoder

# ---------------------------------------------------------------------------
# Defaults (PIPELINE Day 6-7)
# ---------------------------------------------------------------------------
DEFAULT_P_VALUES_FULL = [
    0.005, 0.010, 0.020, 0.030, 0.050, 0.070, 0.100, 0.130, 0.160, 0.200,
]
DEFAULT_P_VALUES_SMOKE = [0.01, 0.05, 0.10]

DEFAULT_TRIALS_FULL = 1000        # PIPELINE: "1000+ trials per data point"
DEFAULT_TRIALS_SMOKE = 30
DEFAULT_SWEEPS = 200              # matches Day 5 benchmark canonical config
DEFAULT_SEED = 20260526
RESULTS_DIR = Path(__file__).resolve().parent / "results"

CODE_FACTORIES = {
    "tiny": make_tiny,
    "small": make_small,
    "medium": make_medium,
    "large": make_large,
}


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------
def run_cell(code, p: float, num_trials: int, num_sweeps: int, seed: int) -> EvalResult:
    """One (code, decoder, p) Monte-Carlo cell."""
    dec = NishimoriMetropolisDecoder(p=p, num_sweeps=num_sweeps)
    # Decoder identity is "metropolis_<sweeps>" so the JSON record carries it.
    decoder_name = f"metropolis_T1.00_sweeps{num_sweeps}"
    r = evaluate(code, dec, p=p, num_trials=num_trials, seed=seed,
                 decoder_name=decoder_name)
    # Stash extra config for downstream plotting / reproducibility.
    r.extra.update({
        "num_sweeps": num_sweeps,
        "T": dec.T,
        "gamma": dec.gamma,
        "mu": dec.mu,
    })
    return r


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_streaming(results: list, json_path: Path) -> None:
    """Atomically write the current results list to JSON (overwrite each cell)."""
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    tmp.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(json_path)


def write_text_table(results: list, txt_path: Path, args) -> None:
    lines = [
        "Day 6 — Metropolis FER vs p sweep",
        f"  seed: {args.seed}    sweeps: {args.sweeps}    trials/cell: {args.trials}",
        f"  decoder: NishimoriMetropolis (Sourlas-Nishimori, T=1)",
        "",
        f"{'code':>7} {'n':>4} {'k':>4} {'p':>7} {'trials':>7} "
        f"{'FER':>8} {'FER_lo':>8} {'FER_hi':>8} {'SER':>8} {'BER':>8} {'sec':>7}",
        "-" * 92,
    ]
    for r in results:
        lines.append(
            f"{r.code_name:>7} {r.n:>4} {r.k:>4} {r.p:>7.4f} {r.num_trials:>7} "
            f"{r.fer:>8.4f} {r.fer_ci_low:>8.4f} {r.fer_ci_high:>8.4f} "
            f"{r.symbol_error_rate:>8.4f} {r.bit_error_rate:>8.4f} {r.elapsed_sec:>7.1f}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(results: list, csv_path: Path) -> None:
    headers = [
        "code_name", "n", "k", "p", "num_trials", "num_frame_errors",
        "fer", "fer_ci_low", "fer_ci_high",
        "symbol_error_rate", "symbol_ci_low", "symbol_ci_high",
        "bit_error_rate", "bit_ci_low", "bit_ci_high",
        "elapsed_sec", "seed", "decoder_name",
    ]
    rows = [",".join(headers)]
    for r in results:
        rows.append(",".join(str(getattr(r, h)) for h in headers))
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Figure 2 draft
# ---------------------------------------------------------------------------
def make_fer_figure(results: list, png_path: Path) -> None:
    """FER vs p, one line per code, log-y. Skip if matplotlib unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not installed — skipping PNG")
        return

    # Group by code, preserving the order the codes appear in results.
    by_code: dict[str, list[EvalResult]] = {}
    for r in results:
        by_code.setdefault(r.code_name, []).append(r)
    for name in by_code:
        by_code[name].sort(key=lambda r: r.p)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {"tiny": "C3", "small": "C0", "medium": "C2", "large": "C1"}
    markers = {"tiny": "o", "small": "s", "medium": "^", "large": "D"}
    for name, rs in by_code.items():
        ps = np.array([r.p for r in rs])
        fers = np.array([r.fer for r in rs])
        lo = np.array([r.fer_ci_low for r in rs])
        hi = np.array([r.fer_ci_high for r in rs])
        # asymmetric error bars; clip the lower at a small floor for log plot
        floor = 1e-4
        fers_plot = np.maximum(fers, floor)
        yerr_lo = fers_plot - np.maximum(lo, floor)
        yerr_hi = np.maximum(hi - fers_plot, 0)
        n = rs[0].n if rs else "?"
        k = rs[0].k if rs else "?"
        ax.errorbar(
            ps, fers_plot, yerr=[yerr_lo, yerr_hi],
            label=f"{name} (n={n}, k={k})",
            marker=markers.get(name, "o"), color=colors.get(name, "k"),
            linewidth=1.4, markersize=5, capsize=2,
        )
    ax.set_xlabel("Symbol error rate $p$")
    ax.set_ylabel("Frame error rate")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-4)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Metropolis decoder — FER vs $p$ on BB(F$_4$) codes")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def make_ser_ber_figure(results: list, png_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    by_code: dict[str, list[EvalResult]] = {}
    for r in results:
        by_code.setdefault(r.code_name, []).append(r)
    for name in by_code:
        by_code[name].sort(key=lambda r: r.p)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, rs in by_code.items():
        ps = np.array([r.p for r in rs])
        ser = np.array([r.symbol_error_rate for r in rs])
        ber = np.array([r.bit_error_rate for r in rs])
        ax1.plot(ps, ser, "-o", label=name, markersize=4)
        ax2.plot(ps, ber, "-o", label=name, markersize=4)
    # Channel SER baseline = p
    ps_all = np.linspace(0, max(r.p for r in results), 50)
    ax1.plot(ps_all, ps_all, "--", color="gray", linewidth=1, label="channel (p)")
    ax2.plot(ps_all, ps_all * 2 / 3, "--", color="gray", linewidth=1,
             label="channel ((2/3)·p)")
    for ax, title in [(ax1, "Symbol error rate"), (ax2, "Bit error rate")]:
        ax.set_xlabel("Channel p")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true",
                      help="Tiny sanity run (~30s) — tiny code, 3 p-values, 30 trials.")
    mode.add_argument("--full", action="store_true",
                      help="Full production sweep (default if neither flag given).")
    ap.add_argument("--code", choices=list(CODE_FACTORIES.keys()),
                    action="append", default=None,
                    help="Restrict to this code (repeatable). Default: all four.")
    ap.add_argument("--trials", type=int, default=None,
                    help="Override trial count per cell.")
    ap.add_argument("--sweeps", type=int, default=DEFAULT_SWEEPS,
                    help=f"Metropolis sweeps per trial (default {DEFAULT_SWEEPS}).")
    ap.add_argument("--p", type=float, action="append", default=None,
                    help="Override p-values (repeatable). Default: 10-point sweep.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    smoke = args.smoke

    # Mode-derived defaults
    if args.p is None:
        args.p = DEFAULT_P_VALUES_SMOKE if smoke else DEFAULT_P_VALUES_FULL
    if args.trials is None:
        args.trials = DEFAULT_TRIALS_SMOKE if smoke else DEFAULT_TRIALS_FULL

    if args.code:
        code_names = args.code
    else:
        code_names = ["tiny"] if smoke else list(CODE_FACTORIES.keys())

    args.out_dir.mkdir(exist_ok=True)
    tag = "smoke" if smoke else "full"
    json_path = args.out_dir / f"day6_metropolis_sweep_{tag}.json"
    txt_path = args.out_dir / f"day6_metropolis_sweep_{tag}.txt"
    csv_path = args.out_dir / f"day6_metropolis_sweep_{tag}.csv"
    fer_png = args.out_dir / f"day6_metropolis_fer_{tag}.png"
    aux_png = args.out_dir / f"day6_metropolis_ser_ber_{tag}.png"

    # Build codes lazily (large takes ~1s to construct because of gf4_rank).
    codes = [CODE_FACTORIES[name]() for name in code_names]

    n_cells = sum(len(args.p) for _ in codes)
    print(f"Day 6 FER sweep ({tag})")
    print(f"  codes:    {[c.name for c in codes]}")
    print(f"  p_values: {args.p}")
    print(f"  trials:   {args.trials}     sweeps: {args.sweeps}     seed: {args.seed}")
    print(f"  out:      {args.out_dir}")
    print(f"  total cells: {n_cells}")
    print()

    results: list[EvalResult] = []
    t_start = time.perf_counter()
    cell_idx = 0
    for code in codes:
        print(f"--- {code.name} (n={code.n}, k={code.k}) ---")
        for p in args.p:
            cell_idx += 1
            t0 = time.perf_counter()
            r = run_cell(code, p=p, num_trials=args.trials,
                         num_sweeps=args.sweeps, seed=args.seed)
            results.append(r)
            elapsed = time.perf_counter() - t0
            print(
                f"  [{cell_idx:>2}/{n_cells}] p={p:.4f}  "
                f"FER={r.fer:.4f} [{r.fer_ci_low:.4f}, {r.fer_ci_high:.4f}]  "
                f"frames_err={r.num_frame_errors:>4}/{r.num_trials}  "
                f"({elapsed:.1f}s)"
            )
            # Stream every cell to disk so partial runs aren't lost.
            write_streaming(results, json_path)

    total = time.perf_counter() - t_start
    print(f"\nTotal: {total:.1f}s = {total/60:.1f}min for {n_cells} cells")

    write_text_table(results, txt_path, args)
    write_csv(results, csv_path)
    make_fer_figure(results, fer_png)
    make_ser_ber_figure(results, aux_png)

    print(f"\nFiles written:")
    for p in [json_path, txt_path, csv_path, fer_png, aux_png]:
        if p.exists():
            print(f"  {p}")

    # Sensibility checks (PIPELINE Day 6-7 GO/NO-GO)
    print("\nSensibility checks (Day 6 GO/NO-GO):")
    by_code: dict[str, list] = {}
    for r in results:
        by_code.setdefault(r.code_name, []).append(r)
    all_ok = True
    for name, rs in by_code.items():
        rs.sort(key=lambda r: r.p)
        fers = [r.fer for r in rs]
        monotone = all(b >= a - 1e-9 for a, b in zip(fers, fers[1:]))
        # CI-strict monotone: each consecutive pair has CIs that order correctly.
        strict = True
        for a, b in zip(rs, rs[1:]):
            if a.fer_ci_low > b.fer_ci_high:  # later p has lower CI than earlier
                strict = False
                break
        tag = "PASS" if monotone else "FAIL"
        print(f"  {name:>7}: FER monotone in p: {tag}  "
              f"(FER {fers[0]:.3f} → {fers[-1]:.3f}, CI-strict: {'yes' if strict else 'no'})")
        if not monotone:
            all_ok = False

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())