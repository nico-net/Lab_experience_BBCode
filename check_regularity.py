"""check_regularity.py — confirm the four BB codes are (d_v, d_c) = (3, 6)-regular.

The Day 16 EXIT threshold p* = 0.175 is the threshold of the (3,6)-REGULAR
ensemble over GF(4): the only thing about the codes that enters the EXIT
calculation is the degree pair (d_v, d_c) = (3, 6) hard-coded in
day16_threshold.py. That number is the correct asymptotic target for a given
code only if that code's Tanner graph really has constant column weight 3
(variable-node degree) and constant row weight 6 (check-node degree).

This script verifies that for tiny / small / medium / large by reading the
degree distribution directly off H = [A | B]. It does not rely on the
BBCode.row_weight()/col_weight() helpers, which assert uniformity and would
raise rather than report an irregular profile.

Run:
    python check_regularity.py
"""
from __future__ import annotations
from collections import Counter

import numpy as np

from bb_constructor import ALL_INSTANCES

EXPECTED_DV = 3   # column weight  -> variable-node degree
EXPECTED_DC = 6   # row weight     -> check-node degree


def weight_dist(weights) -> Counter:
    """Counter {weight: how_many_nodes_have_it}."""
    return Counter(int(w) for w in weights)


def fmt_dist(dist: Counter) -> str:
    """e.g. {3: 18} -> '3:18'  (weight:count, comma-separated if mixed)."""
    return ", ".join(f"{w}:{c}" for w, c in sorted(dist.items()))


def main() -> None:
    codes = [make() for make in ALL_INSTANCES]

    print(f"Checking (d_v, d_c) = ({EXPECTED_DV}, {EXPECTED_DC}) regularity "
          f"of H = [A | B] over F_4")
    print("(weight:count — e.g. '3:18' means 18 columns of weight 3)\n")

    header = (f"{'code':>8}  {'n':>5}  {'checks':>6}  "
              f"{'col wt (d_v)':>16}  {'row wt (d_c)':>16}  verdict")
    print(header)
    print("-" * len(header))

    all_regular = True
    for code in codes:
        H = code.H
        col_w = (H != 0).sum(axis=0)   # length n  (one per variable node)
        row_w = (H != 0).sum(axis=1)   # length num_checks (one per check)

        col_dist = weight_dist(col_w)
        row_dist = weight_dist(row_w)

        ok = (set(col_dist) == {EXPECTED_DV}) and (set(row_dist) == {EXPECTED_DC})
        all_regular &= ok

        print(f"{code.name:>8}  {code.n:>5}  {code.num_checks:>6}  "
              f"{fmt_dist(col_dist):>16}  {fmt_dist(row_dist):>16}  "
              f"{'OK (3,6)' if ok else 'IRREGULAR'}")

    print()
    if all_regular:
        print("PASS — all four codes are (3,6)-regular. The Day 16 ensemble "
              "threshold p* = 0.175 is the common asymptotic target for every "
              "instance; the four simulated thresholds differ from it only by "
              "finite-length effects (Day 17).")
    else:
        print("FAIL — at least one code is IRREGULAR (see distribution above). "
              "A single (3,6) regular-ensemble threshold is NOT the right "
              "asymptotic prediction for it. Use an irregular-EXIT treatment "
              "with the actual degree distribution before the Day 17 comparison.")

    # Context only: the (3,6) design rate is 1 - d_v/d_c = 0.5. Actual rates
    # sit slightly above 0.5 because H has redundant rows (rank < num_checks).
    # This does not affect the tunnel-pinch threshold, but Day 17 may cite it.
    print("\nRate context (design rate of a (3,6) ensemble = 1 - 3/6 = 0.500):")
    print(f"{'code':>8}  {'k':>4}  {'n':>5}  {'rank':>5}  {'actual R = k/n':>14}")
    for code in codes:
        print(f"{code.name:>8}  {code.k:>4}  {code.n:>5}  "
              f"{code.rank:>5}  {code.rate:>14.3f}")


if __name__ == "__main__":
    main()