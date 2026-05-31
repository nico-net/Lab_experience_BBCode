"""day13_14_bp_sweep.py — full BP FER waterfall on all four code sizes.

PIPELINE Day 13-14 deliverable:
  - BP on tiny / small / medium / large at p ∈ [0.001, 0.30] log-spaced
  - ≥ 1000 trials per point (relaxed at high p where BP is expensive and FER
    is already near 1 — Wilson CIs are still well-controlled with 300-500 trials)
  - Raw data saved to CSV after every cell so partial completion is usable
  - Figure 2 (FER vs p, one curve per code) generated at the end

Run with `python day13_14_bp_sweep.py`. Estimated wall time ~12-15 min.
Tolerates KeyboardInterrupt cleanly: prints what's done and exits.
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bb_constructor import (BBCode, make_tiny, make_small, make_medium, make_large)
from channel import QSC
from decoder_bp import BPFFTDecoder


# ---------------------------------------------------------------------------
# Wilson 95% interval (matches evaluation.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# One cell evaluation
# ---------------------------------------------------------------------------
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


def run_cell(code: BBCode, p: float, n_trials: int, seed: int,
             max_iters: int = 50) -> Cell:
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
    fer = fails / n_trials
    lo, hi = wilson_interval(fails, n_trials)
    return Cell(
        code_name=code.name, n=code.n, k=code.k,
        p=p, num_trials=n_trials,
        num_frame_errors=fails, fer=fer, fer_ci_low=lo, fer_ci_high=hi,
        num_sym_errors=sym, ser=sym / (n_trials * code.n),
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# Adaptive trial-count schedule. At low p we want enough trials to bound the
# FER below the noise floor; at high p, BP is expensive and FER is high so
# the Wilson CI tightens quickly.
# ---------------------------------------------------------------------------
def trials_for_p(p: float) -> int:
    if p <= 0.005:   return 2000
    if p <= 0.02:    return 1500
    if p <= 0.06:    return 1000
    if p <= 0.10:    return 500
    return 300


# ---------------------------------------------------------------------------
# Sweep + incremental save
# ---------------------------------------------------------------------------
def write_csv(rows, path: Path):
    fields = ["code_name", "n", "k", "p", "num_trials", "num_frame_errors",
              "fer", "fer_ci_low", "fer_ci_high", "num_sym_errors", "ser",
              "elapsed_sec"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in rows:
            w.writerow({k: getattr(c, k) for k in fields})


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "day13_14_bp_sweep.csv"

    # Log-spaced p grid. 10 points spanning 2.5 decades.
    p_grid = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04, 0.06, 0.10, 0.15, 0.20]
    codes = [make_tiny(), make_small(), make_medium(), make_large()]
    seed = 20260601

    # Quick total-cell estimate so progress prints are meaningful.
    total_cells = len(codes) * len(p_grid)
    cell_idx = 0
    rows: list[Cell] = []
    t_start = time.perf_counter()
    print(f"Day 13-14 BP FER sweep — {len(codes)} codes × {len(p_grid)} p-values "
          f"= {total_cells} cells")
    print(f"p-grid: {p_grid}")
    print(f"max_iters: 50,  seed: {seed}")
    print(f"writing to: {csv_path}")
    print()
    header = (f"{'code':>6}  {'p':>6}  {'n_tr':>5}  {'fails':>5}  {'FER':>8}  "
              f"{'CI':>16}  {'SER':>8}  {'sec':>7}  {'ms/dec':>8}  {'ETA':>7}")
    print(header)
    print("-" * len(header))

    try:
        for code in codes:
            for p in p_grid:
                cell_idx += 1
                n_tr = trials_for_p(p)
                # Seed sub-stream by (code, p) so cells are reproducible / independent.
                cell_seed = hash((seed, code.name, p)) & 0xFFFFFFFF
                cell = run_cell(code, p, n_tr, cell_seed)
                rows.append(cell)
                # Incremental save after every cell.
                write_csv(rows, csv_path)
                # Per-cell timing → ETA.
                elapsed = time.perf_counter() - t_start
                eta = elapsed * (total_cells - cell_idx) / cell_idx
                ms_per = cell.elapsed_sec / cell.num_trials * 1000
                print(
                    f"{cell.code_name:>6}  {p:>6.3f}  {n_tr:>5}  "
                    f"{cell.num_frame_errors:>5}  {cell.fer:>8.4f}  "
                    f"[{cell.fer_ci_low:.3f},{cell.fer_ci_high:.3f}]  "
                    f"{cell.ser:>8.4f}  {cell.elapsed_sec:>7.1f}  "
                    f"{ms_per:>8.2f}  {eta:>6.0f}s"
                )
                sys.stdout.flush()
    except KeyboardInterrupt:
        print(f"\nInterrupted after {cell_idx}/{total_cells} cells.")
        print(f"Partial results saved to {csv_path}.")
        return

    print()
    print(f"Done — {cell_idx}/{total_cells} cells in "
          f"{time.perf_counter() - t_start:.1f}s. Saved to {csv_path}.")


if __name__ == "__main__":
    main()
