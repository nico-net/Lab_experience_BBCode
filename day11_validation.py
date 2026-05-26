"""day11_validation.py — ML decoder validation on the tiny code.

PIPELINE Day 11 deliverable: brute-force ML on smallest code, p ∈ {0.001, 0.01}.

Also runs a head-to-head ML vs BP-FFT preview that will form the heart of
the Day 12 BP-vs-ML validation: at each p, count
  (a) ML failures (frames where ML cannot recover the transmitted codeword),
  (b) BP failures (frames where BP cannot recover it),
  (c) BP-but-not-ML failures = lower bound on BP's loss to optimal decoding.
ML failures are channel-limited (the noise pattern is uncorrectable for any
decoder); (b) - (a) is purely BP's suboptimality.
"""
from __future__ import annotations

import time
import numpy as np

from bb_constructor import make_tiny
from channel import QSC
from decoder_bp import BPFFTDecoder
from decoder_ml import MLDecoder


def fer_run(code, decoder, p, n_trials, seed):
    rng = np.random.default_rng(seed)
    qsc = QSC(p)
    zero = np.zeros(code.n, dtype=np.uint8)
    fails = 0
    sym_errors = 0
    t0 = time.perf_counter()
    for _ in range(n_trials):
        recv, _ = qsc.transmit(zero, rng)
        out = decoder(code, recv)
        diffs = int((out != zero).sum())
        sym_errors += diffs
        if diffs > 0:
            fails += 1
    return fails, sym_errors, time.perf_counter() - t0


def head_to_head(code, p, n_trials, seed):
    """
    Decode the same noise realisations with both ML and BP and tally:
        ml_fail   = ML did not return the transmitted codeword,
        bp_fail   = BP did not return the transmitted codeword,
        only_bp   = BP failed but ML succeeded (BP's suboptimality).
    All three are integer counts over n_trials with the same seed.
    """
    rng = np.random.default_rng(seed)
    qsc = QSC(p)
    ml = MLDecoder(p=p)
    bp = BPFFTDecoder(p=p, max_iters=50)
    zero = np.zeros(code.n, dtype=np.uint8)
    ml_fail = bp_fail = only_bp = 0
    t0 = time.perf_counter()
    for _ in range(n_trials):
        recv, _ = qsc.transmit(zero, rng)
        out_ml = ml(code, recv)
        out_bp = bp(code, recv)
        ml_bad = not np.array_equal(out_ml, zero)
        bp_bad = not np.array_equal(out_bp, zero)
        ml_fail += int(ml_bad)
        bp_fail += int(bp_bad)
        only_bp += int(bp_bad and not ml_bad)
    return ml_fail, bp_fail, only_bp, time.perf_counter() - t0


def main():
    code = make_tiny()
    print("=" * 88)
    print(f"Day 11 ML validation — code {code.name!r}, n={code.n}, k={code.k}, d=6")
    print("=" * 88)

    # ---- 1. ML alone, at the two PIPELINE-specified noise rates ----
    print(f"\nML FER on tiny code:")
    print(f"  {'p':>7}  {'trials':>7}  {'fails':>7}  {'FER':>10}  {'SER':>10}  "
          f"{'sec':>7}  {'ms/dec':>8}")
    for p, n_trials in ((0.001, 2000), (0.01, 2000)):
        ml = MLDecoder(p=p)
        fails, sym, elapsed = fer_run(code, ml, p, n_trials, seed=20260601)
        fer = fails / n_trials
        ser = sym / (n_trials * code.n)
        print(f"  {p:>7.4f}  {n_trials:>7}  {fails:>7}  {fer:>10.5f}  "
              f"{ser:>10.5f}  {elapsed:>7.2f}  {elapsed/n_trials*1000:>8.2f}")

    # ---- 2. Head-to-head ML vs BP at a range of p ----
    print(f"\nML vs BP head-to-head (same noise realisations, tiny code):")
    print(f"  {'p':>7}  {'trials':>7}  {'ML_fail':>8}  {'BP_fail':>8}  "
          f"{'only_BP':>8}  {'note':<30}  {'sec':>7}")
    for p, n_trials in ((0.001, 2000), (0.005, 2000), (0.01, 2000),
                        (0.02, 1000), (0.05, 1000), (0.10, 500)):
        ml_fail, bp_fail, only_bp, elapsed = head_to_head(
            code, p, n_trials, seed=20260601
        )
        if only_bp == 0:
            note = "BP matches ML exactly"
        else:
            gap = only_bp / n_trials
            note = f"BP suboptimal in {only_bp}/{n_trials} ({100*gap:.2f}%)"
        print(f"  {p:>7.4f}  {n_trials:>7}  {ml_fail:>8}  {bp_fail:>8}  "
              f"{only_bp:>8}  {note:<30}  {elapsed:>7.2f}")

    print("\nInterpretation:")
    print("  ML_fail counts channel-limited failures (uncorrectable noise patterns).")
    print("  BP_fail >= ML_fail since BP is at best as good as ML.")
    print("  only_BP isolates BP's suboptimality — the Day 12 quantity of interest.")


if __name__ == "__main__":
    main()
