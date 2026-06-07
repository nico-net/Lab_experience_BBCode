"""day18_finegrid_sweep.py — dense BP FER re-sweep near threshold (Day 18 front half).

Day 17 left the simulated thresholds soft: the FER=0.5 crossing was interpolated
across a 0.05-wide gap ([0.10, 0.15]) at only 300-500 trials/point, giving ~±0.01
crossings — too loose for a clean finite-size-scaling collapse. This script fixes
both problems for the m=6 family (small / medium / large; tiny is excluded — it is
off-family (3,3) and its high-p FER is contaminated by the small-codeword-space
lucky-guess effect, so it must not enter the collapse fit):

  - Fine p-grid bracketing the crossing region: 0.08 .. 0.16.
  - ~2000 trials in the transition band (FER half-width ~±0.02 -> crossing ~±0.003).

Mechanics reused verbatim from day13_14_bp_sweep.py (QSC channel, BPFFTDecoder,
all-zero transmitted codeword, Wilson 95% CIs, identical CSV schema) so the Day 18
collapse fitter can read this file and the original interchangeably.

Two upgrades over the Day 13-14 sweep:
  1. Deterministic, independent per-cell seeding via numpy SeedSequence. (The
     original used hash((seed, name, p)), which is salted per-process for strings
     unless PYTHONHASHSEED is pinned — so its "reproducible" cells were not actually
     reproducible across runs. SeedSequence([master, code_idx, p_idx]) is.)
  2. Resumable: completed (code, p) cells already in the CSV are skipped, and rows
     are appended (not rewritten), so Ctrl-C / closing the laptop loses nothing.

Estimated wall time ~1.5-2 h (small ~2.5 min/pt, medium ~5, large ~6 at 2000
trials; lower p is cheaper). Resume any time by re-running.

Run:
    python day18_finegrid_sweep.py
    python day18_finegrid_sweep.py --include-tiny   # add tiny for plotting only
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bb_constructor import make_tiny, make_small, make_medium, make_large
from channel import QSC
from decoder_bp import BPFFTDecoder

MASTER_SEED = 20260601
FIELDS = ["code_name", "n", "k", "p", "num_trials", "num_frame_errors",
          "fer", "fer_ci_low", "fer_ci_high", "num_sym_errors", "ser",
          "elapsed_sec"]


def wilson_interval(successes: int, trials: int, z: float = 1.96):
    if trials == 0:
        return (0.0, 1.0)
    n = trials
    p_hat = successes / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, center - half)
    hi = 1.0 if successes == trials else min(1.0, center + half)
    return (lo, hi)


@dataclass
class Cell:
    code_name: str
    n: int
    k: int
    p: float
    num_trials: int
    num_frame_errors: int
    fer: float
    fer_ci_low: float
    fer_ci_high: float
    num_sym_errors: int
    ser: float
    elapsed_sec: float


def run_cell(code, p, n_trials, seed, max_iters=50) -> Cell:
    dec = BPFFTDecoder(p=p, max_iters=max_iters)
    qsc = QSC(p)
    rng = np.random.default_rng(seed)
    zero = np.zeros(code.n, dtype=np.uint8)
    fails = 0
    sym = 0
    t0 = time.perf_counter()
    for _ in range(n_trials):
        recv, _ = qsc.transmit(zero, rng)
        out = dec(code, recv)
        d = int((out != zero).sum())
        sym += d
        if d > 0:
            fails += 1
    elapsed = time.perf_counter() - t0
    lo, hi = wilson_interval(fails, n_trials)
    return Cell(code.name, code.n, code.k, p, n_trials, fails,
                fails / n_trials, lo, hi, sym, sym / (n_trials * code.n), elapsed)


def trials_for_p(p: float) -> int:
    # Heavy sampling through the transition band; lighter in the low-p approach.
    return 1500 if p < 0.10 else 2000


def cell_seed(code_idx: int, p_idx: int) -> int:
    ss = np.random.SeedSequence([MASTER_SEED, code_idx, p_idx])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def load_done(path: Path) -> set:
    done = set()
    if path.exists():
        with path.open() as f:
            for row in csv.DictReader(f):
                done.add((row["code_name"], round(float(row["p"]), 6)))
    return done


def ensure_header(path: Path):
    if (not path.exists()) or path.stat().st_size == 0:
        with path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def append_cell(path: Path, cell: Cell):
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writerow(
            {k: getattr(cell, k) for k in FIELDS})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--include-tiny", action="store_true",
                    help="Also sweep tiny (for the FER plot only — NOT the collapse fit).")
    ap.add_argument("--out", type=str, default=None,
                    help="Output CSV (default ./results/day18_finegrid_sweep.csv).")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = Path(args.out) if args.out else out_dir / "day18_finegrid_sweep.csv"

    p_grid = [0.08, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
    makers = [make_small, make_medium, make_large]
    if args.include_tiny:
        makers = [make_tiny] + makers
    codes = [m() for m in makers]

    done = load_done(csv_path)
    ensure_header(csv_path)

    total = len(codes) * len(p_grid)
    remaining = total - sum(
        (c.name, round(p, 6)) in done for c in codes for p in p_grid)
    print(f"Day 18 fine-grid re-sweep — {len(codes)} codes × {len(p_grid)} p")
    print(f"p-grid: {p_grid}")
    print(f"family: {', '.join(c.name for c in codes)}  (collapse fit uses "
          f"small/medium/large only)")
    print(f"CSV: {csv_path}")
    print(f"already done: {total - remaining}/{total}; to run: {remaining}\n")
    hdr = (f"{'code':>7}  {'p':>5}  {'n_tr':>5}  {'fails':>5}  {'FER':>7}  "
           f"{'CI95':>17}  {'sec':>7}  {'ETA':>8}")
    print(hdr)
    print("-" * len(hdr))

    t_start = time.perf_counter()
    n_run = 0
    try:
        for ci, code in enumerate(codes):
            for pi, p in enumerate(p_grid):
                if (code.name, round(p, 6)) in done:
                    continue
                n_tr = trials_for_p(p)
                cell = run_cell(code, p, n_tr, cell_seed(ci, pi))
                append_cell(csv_path, cell)
                n_run += 1
                avg = (time.perf_counter() - t_start) / n_run
                eta = avg * (remaining - n_run)
                ci_str = f"[{cell.fer_ci_low:.3f},{cell.fer_ci_high:.3f}]"
                print(f"{code.name:>7}  {p:>5.3f}  {n_tr:>5}  {cell.num_frame_errors:>5}  "
                      f"{cell.fer:>7.4f}  {ci_str:>17}  {cell.elapsed_sec:>7.1f}  "
                      f"{eta/60:>6.1f}m")
    except KeyboardInterrupt:
        print(f"\nInterrupted — {n_run} new cells saved to {csv_path}. "
              f"Re-run to resume; finished cells are skipped.")
        return

    print(f"\nDone. {n_run} new cells written. Full fine grid in {csv_path}.")
    print("Next: Day 18 collapse fitter reads this CSV (+ the coarse day13_14 CSV "
          "for the tails) to extract p_c(infinity) ± CI.")


if __name__ == "__main__":
    main()