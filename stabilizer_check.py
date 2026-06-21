"""stabilizer_check.py — Day 21 prerequisite for Section VII.

Resolves the open question from the d<=6 discussion: is the structural
weight-6 codeword

    c = [B.e_j ; A.e_j]   (j = 0, ..., ell*m - 1)

a STABILIZER of the quantum CSS code QC(A,B), or a genuine LOGICAL operator?

Background
----------
H = [A | B] is both the classical parity-check matrix studied in this paper
AND (per Bravyi 2024) the X-stabilizer check matrix of a quantum CSS code,
with Z-stabilizers H^Z = [B^T | A^T]. AB = BA (commuting shifts) forces
c = [B.e ; A.e] to satisfy H.c = 0 for every e -- proved in DECISION_LOG,
Day 18. So c is always a classical codeword of weight exactly 6, which is
why the classical distance is capped at d <= 6 for the whole weight-3
bicycle family.

The quantum distance is NOT the classical ker(H) distance: a quantum LOGICAL
operator must lie in ker(H^X) minus rowspace(H^Z) -- i.e. it must commute with
every X-check (already guaranteed, since c in ker(H)) AND must NOT be
expressible as a combination of Z-stabilizer rows (i.e. NOT in rowspace(H^Z)).
If c IS in rowspace(H^Z), it is a stabilizer -- physically trivial, and it
does not lower-bound the quantum distance. If c is NOT in rowspace(H^Z),
it is a genuine logical operator and the quantum distance is also <= 6.

This is exactly the test needed before Section VII can honestly state
whether the quantum distance escapes the classical ceiling.

Method
------
Quotient-membership test over GF(4) via rank:
    c in rowspace(M)   <=>   rank([M; c]) == rank(M)
implemented with the existing gf4_rref / gf4_rank from gf4_lib.py (no new
linear algebra -- reuses the exact machinery already validated by the
project's own unit tests).

Checked EXHAUSTIVELY over all e_j (j = 0, ..., ell*m - 1) for every code, not
just a sample of shift positions. The Day 18 theorem guarantees every e_j
gives a weight-6 codeword in ker(H); this script confirms the corresponding
membership-in-rowspace(H^Z) verdict for literally all of them, so the paper
claim can be stated without a "representative sample" qualifier.

rank(H^Z) is computed once per code (it doesn't depend on j) and reused for
every membership test, since gf4_rref/gf4_rank cost grows with matrix size
and recomputing it ell*m times would be wasteful for the larger codes.

Run:
    python stabilizer_check.py                 # exhaustive, all codes
    python stabilizer_check.py --codes large    # just one code, for a quick rerun
    python stabilizer_check.py --quiet          # suppress per-row table, summary only
"""
from __future__ import annotations
import argparse
import time

import numpy as np

from gf4_lib import gf4_rref, gf4_rank
from bb_constructor import ALL_INSTANCES


def build_HZ(code) -> np.ndarray:
    """H^Z = [B^T | A^T], the Z-stabilizer check matrix (Bravyi 2024, eq. 2)."""
    return np.concatenate([code.B.T, code.A.T], axis=1).astype(np.uint8)


def in_rowspace_given_rank(M: np.ndarray, rank_M: int, c: np.ndarray) -> tuple[bool, int]:
    """c in rowspace(M) over GF(4)? Caller supplies the precomputed rank(M)."""
    stacked = np.vstack([M, c.reshape(1, -1)])
    rank_Mc = gf4_rank(stacked)
    return (rank_Mc == rank_M), rank_Mc


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes", default=None,
                    help="Comma-separated subset of code names (default: all).")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-position rows; print only per-code summary.")
    args = ap.parse_args()

    from gf4_lib import gf4_matvec

    only = set(c.strip() for c in args.codes.split(",")) if args.codes else None

    print("Day 21 prerequisite: is c = [B.e_j ; A.e_j] a stabilizer or a logical?")
    print("Test: c in rowspace(H^Z) over GF(4), via rank([H^Z; c]) == rank(H^Z)")
    print("EXHAUSTIVE over all j = 0..ell*m-1 for every code.\n")

    if not args.quiet:
        hdr = (f"{'code':>8}  {'j':>4}  {'wt(c)':>6}  {'rank(HZ;c)':>11}  verdict")
        print(hdr)
        print("-" * len(hdr))

    summary = {}
    for make in ALL_INSTANCES:
        code = make()
        if only and code.name not in only:
            continue
        t0 = time.perf_counter()
        HZ = build_HZ(code)
        rank_HZ = gf4_rank(HZ)
        m = code.B.shape[1]  # = ell*m, number of shift positions / basis vectors

        n_stab = 0
        n_logical = 0
        bad_weights = []
        for j in range(m):
            ej = np.zeros(m, dtype=np.uint8)
            ej[j] = 1
            Be = gf4_matvec(code.B, ej)
            Ae = gf4_matvec(code.A, ej)
            c = np.concatenate([Be, Ae])

            # sanity: c must be in ker(H) -- the Day 18 theorem, checked for every j
            Hc = gf4_matvec(code.H, c)
            assert np.all(Hc == 0), \
                f"c not in ker(H) for {code.name}, j={j} -- construction error"
            wt = int((c != 0).sum())
            if wt != 6:
                bad_weights.append((j, wt))

            is_stab, rank_HZc = in_rowspace_given_rank(HZ, rank_HZ, c)
            if is_stab:
                n_stab += 1
            else:
                n_logical += 1
                tag = "LOGICAL"
                print(f"{code.name:>8}  {j:>4}  {wt:>6}  {rank_HZc:>11}  {tag}")

            if not args.quiet and is_stab:
                # Only print stabilizer rows when not in quiet mode, to keep
                # output readable for the larger codes (144 -> 144 rows).
                tag = "STABILIZER"
                print(f"{code.name:>8}  {j:>4}  {wt:>6}  {rank_HZc:>11}  {tag}")

        dt = time.perf_counter() - t0
        summary[code.name] = (n_stab, n_logical, m, bad_weights, dt)
        if not args.quiet:
            print()

    print("=" * 72)
    print("SUMMARY (exhaustive over all shift positions)")
    print("=" * 72)
    for name, (n_stab, n_logical, m, bad_weights, dt) in summary.items():
        if bad_weights:
            print(f"{name:>8}: WARNING — {len(bad_weights)}/{m} positions had "
                  f"wt(c) != 6: {bad_weights[:5]}{'...' if len(bad_weights) > 5 else ''}")
        if n_logical == 0:
            print(f"{name:>8}: ALL {m}/{m} positions are STABILIZERS "
                  f"({dt:.2f}s) -> the classical d<=6 ceiling does NOT bound "
                  f"the quantum distance for this code. Holds for every j, "
                  f"not just a sample.")
        elif n_stab == 0:
            print(f"{name:>8}: ALL {m}/{m} positions are LOGICAL ({dt:.2f}s) "
                  f"-> quantum d <= 6 too for this code.")
        else:
            print(f"{name:>8}: MIXED — {n_stab}/{m} stabilizer, {n_logical}/{m} "
                  f"logical ({dt:.2f}s) -> position-dependent; report the split, "
                  f"do not generalize to 'all' or 'none'.")

    print()
    print("This table is exhaustive (every j checked, not a sample) and can be")
    print("cited in Section VII / the Day 21 DECISIONS_LOG entry without a")
    print("'representative positions' qualifier.")


if __name__ == "__main__":
    main()