"""candidate_distance_scan.py — screen BB code families for GROWING distance.

Day 18's data collapse failed because tiny/small/medium/large are all pinned at
d <= 6 regardless of ell: increasing the size buys no extra protection, so the
family has no finite-size-scaling trend to collapse. Before re-running any
simulation on a new family, this script answers the only question that matters:
does d grow with ell?

It reports, per code:
  - n, k, rank (via the existing BBCode machinery)
  - a STRUCTURAL flag: for the x- and y-supports of A and B, the size of the
    cyclic subgroup the support generates. If a support sits inside a PROPER
    subgroup of Z_ell (resp. Z_m), that forces structured low-weight codewords
    and is the root cause of the d-ceiling (e.g. the {0,4,8} order-3 subgroup of
    Z_12 found on Day 6). A healthy code wants "full" on every support.
  - an UPPER BOUND on the classical d of ker(H), found the Day-3 way: minimum
    basis row weight + low-weight message enumeration + uniform random sampling.
    (Exact only when 4^k is small enough to brute-force.)

A family is collapse-viable iff this d-bound INCREASES across the sizes. If it
stays flat (e.g. 6, 6, 6), the family is distance-capped and no amount of extra
simulation will produce a clean collapse.

This is a screen, not a proof: the d-bound is an upper bound (random search can
miss the true minimum), but a flat upper bound across sizes is already decisive
evidence of a ceiling, and a structural subgroup flag explains why.

Run:
    python candidate_distance_scan.py                 # the current 4 codes
    python candidate_distance_scan.py --family current_scaled
    python candidate_distance_scan.py --family candidate --n-random 50000
"""
from __future__ import annotations
import argparse
from math import comb, gcd
from typing import List, Tuple

import numpy as np

from gf4_lib import ONE, OMEGA, OMEGA2
from bb_constructor import BBCode, ALL_INSTANCES

NZ = np.array([ONE, OMEGA, OMEGA2], dtype=np.uint8)

# ---- GF(4) multiplication table (addition is XOR in this representation) ----
def _mul(a, b):
    if a == 0 or b == 0:
        return 0
    if a == ONE:
        return b
    if b == ONE:
        return a
    if a == OMEGA and b == OMEGA:
        return OMEGA2
    if a == OMEGA2 and b == OMEGA2:
        return OMEGA
    return ONE  # OMEGA * OMEGA2


_MAX = int(max(0, ONE, OMEGA, OMEGA2))
MULT = np.zeros((_MAX + 1, _MAX + 1), dtype=np.uint8)
for _a in (0, ONE, OMEGA, OMEGA2):
    for _b in (0, ONE, OMEGA, OMEGA2):
        MULT[_a, _b] = _mul(_a, _b)


def encode_batch(U: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Encode messages U (M x k) against basis G (k x n) over F_4 -> (M x n)."""
    M, k = U.shape
    C = np.zeros((M, G.shape[1]), dtype=np.uint8)
    for i in range(k):
        C ^= MULT[U[:, i][:, None], G[i][None, :]]
    return C


def min_nonzero_weight(C: np.ndarray) -> int:
    w = (C != 0).sum(axis=1)
    w = w[w > 0]
    return int(w.min()) if w.size else 0


def sample_sparse(k: int, weight: int, count: int, rng) -> np.ndarray:
    """count messages of exactly `weight` nonzeros at random positions/values."""
    if count <= 0 or weight > k:
        return np.zeros((0, k), dtype=np.uint8)
    pos = np.argsort(rng.random((count, k)), axis=1)[:, :weight]
    vals = NZ[rng.integers(0, 3, size=(count, weight))]
    U = np.zeros((count, k), dtype=np.uint8)
    np.put_along_axis(U, pos, vals, axis=1)
    return U


def estimate_distance(code: BBCode, n_random: int, rng) -> Tuple[int, str]:
    G = code.codeword_basis()              # k x n, rows are codewords
    k, n = G.shape
    if k == 0:
        return 0, "trivial"

    # Exact brute force when 4^k is cheap.
    if k <= 11:
        msgs = np.array(np.meshgrid(*[[0, ONE, OMEGA, OMEGA2]] * k,
                                    indexing="ij")).reshape(k, -1).T.astype(np.uint8)
        return min_nonzero_weight(encode_batch(msgs, G)), "exact"

    best = int((G != 0).sum(axis=1).min())  # weight-1 == basis row weights

    # Full weight-2 combos if cheap, else sampled; weight-3 sampled.
    w2 = comb(k, 2) * 9
    U2 = (_all_weight2(k) if w2 <= 200_000
          else sample_sparse(k, 2, 40_000, rng))
    best = min(best, min_nonzero_weight(encode_batch(U2, G)))
    best = min(best, min_nonzero_weight(
        encode_batch(sample_sparse(k, 3, 40_000, rng), G)))

    # Uniform-random full-weight messages (catches non-sparse low-weight words).
    if n_random > 0:
        U = NZ[rng.integers(0, 3, size=(n_random, k))]
        mask = rng.random((n_random, k)) < 0.5
        U = np.where(mask, U, 0).astype(np.uint8)
        best = min(best, min_nonzero_weight(encode_batch(U, G)))
    return best, "<= (upper bound)"


def _all_weight2(k: int) -> np.ndarray:
    ii, jj = np.triu_indices(k, k=1)
    combos = [(sa, sb) for sa in NZ for sb in NZ]
    U = np.zeros((len(ii) * len(combos), k), dtype=np.uint8)
    row = 0
    for sa, sb in combos:
        U[row:row + len(ii), ii] = sa
        U[row:row + len(ii)][np.arange(len(ii)), :]  # noop, clarity
        U[row:row + len(ii), jj] = sb
        # set per-pair: need per-row scatter
        block = np.zeros((len(ii), k), dtype=np.uint8)
        block[np.arange(len(ii)), ii] = sa
        block[np.arange(len(ii)), jj] = sb
        U[row:row + len(ii)] = block
        row += len(ii)
    return U


def subgroup_size(exponents, modulus: int) -> int:
    """Size of the cyclic subgroup of Z_modulus that the support generates."""
    diffs = [(e - exponents[0]) % modulus for e in exponents]
    g = 0
    for d in diffs:
        g = gcd(g, d)
    g = gcd(g, modulus)
    return modulus if g == 0 else modulus // g


def structural_flag(code: BBCode) -> str:
    tags = []
    for label, terms, mod in (("Ax", code.A_terms, code.ell),
                              ("Ay", code.A_terms, code.m),
                              ("Bx", code.B_terms, code.ell),
                              ("By", code.B_terms, code.m)):
        idx = 0 if label.endswith("x") else 1
        exps = sorted({t[idx] % mod for t in terms})
        s = subgroup_size(exps, mod)
        if s < mod:
            tags.append(f"{label}:sub({s}/{mod})")
    return "full" if not tags else " ".join(tags)


# ---------------------------------------------------------------------------
# Families to scan
# ---------------------------------------------------------------------------
def family_current() -> List[BBCode]:
    return [make() for make in ALL_INSTANCES]


def family_current_scaled() -> List[BBCode]:
    """Current small-code A/B pattern, ell swept at m=6 — shows the flat ceiling."""
    fam = []
    for ell in (6, 9, 12, 15, 18):
        fam.append(BBCode(
            ell=ell, m=6,
            A_terms=[(3, 0, ONE), (0, 1, OMEGA), (0, 2, OMEGA2)],
            B_terms=[(0, 3, ONE), (1, 0, OMEGA), (2, 0, OMEGA2)],
            name=f"cs_ell{ell}"))
    return fam


def family_desub() -> List[BBCode]:
    """(a) Current construction with EVERY trapped support de-subgrouped.

    The scan showed the ceiling is driven by supports stuck in proper subgroups
    (By:sub(2/6) on all three; Ax:sub on each). Here both polynomials use
    supports that read 'full': A's x-term moves from x^3 to x^1 (gcd(1,ell)=1),
    and B's y-term moves from y^3 to y^1 (gcd(1,6)=1). So:
        A = x + w*y + w^2*y^2     (x-support {0,1}, y-support {0,1,2})
        B = y + w*x + w^2*x^2     (y-support {0,1}, x-support {0,1,2})
    Whether this lifts the classical d above 6 is exactly what the scan tells you.
    To try other full supports, edit the exponents below (keep gcd(exp,mod)=1)."""
    fam = []
    for ell in (6, 9, 12):
        fam.append(BBCode(
            ell=ell, m=6,
            A_terms=[(1, 0, ONE), (0, 1, OMEGA), (0, 2, OMEGA2)],
            B_terms=[(0, 1, ONE), (1, 0, OMEGA), (2, 0, OMEGA2)],
            name=f"desub_ell{ell}"))
    return fam


def family_bravyi_lift() -> List[BBCode]:
    """(b) Bravyi Table 1 construction lifted to GF(4) at the matching (ell,m).

    Paper main text: A = x^3 + y + y^2, B = y^3 + x + x^2 (the generic BB
    construction; [[72,12,6]] uses exactly this). w-decorated below.

    NOTE 1: at (6,6) this IS make_small, so it caps at classical d=6 — included
            to make the point explicit, not because it's expected to help.
    NOTE 2: Bravyi's d=6/10/12/18 are QUANTUM CSS distances, not the classical
            d of ker(H) this scan measures; the latter is smaller and is what
            the finite-size collapse needs.
    NOTE 3: some references use B = y^3 + x^2 + x^7 for the [[144]] gross code;
            swap the ell=12 B_terms to [(0,3,ONE),(2,0,OMEGA),(7,0,OMEGA2)] to
            test that variant. Verify exact per-code exponents against the repo
            (github.com/sbravyi/BivariateBicycleCodes) before citing."""
    specs = [(6, 6), (9, 6), (12, 6)]   # [[72]], [[108]], [[144]]
    fam = []
    for ell, m in specs:
        fam.append(BBCode(
            ell=ell, m=m,
            A_terms=[(3, 0, ONE), (0, 1, OMEGA), (0, 2, OMEGA2)],
            B_terms=[(0, 3, ONE), (1, 0, OMEGA), (2, 0, OMEGA2)],
            name=f"brav_ell{ell}"))
    return fam


FAMILIES = {
    "current": family_current,
    "current_scaled": family_current_scaled,
    "desub": family_desub,
    "bravyi_lift": family_bravyi_lift,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=list(FAMILIES), default="current")
    ap.add_argument("--n-random", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260603)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    codes = FAMILIES[args.family]()

    print(f"Family '{args.family}'  (d is an UPPER BOUND unless marked exact)\n")
    hdr = (f"{'name':>10}  {'(l,m)':>8}  {'n':>4}  {'k':>4}  {'rank':>4}  "
           f"{'d_ub':>5}  {'method':>16}  structural support")
    print(hdr)
    print("-" * len(hdr))
    ds, ns = [], []
    for code in codes:
        d, method = estimate_distance(code, args.n_random, rng)
        flag = structural_flag(code)
        print(f"{code.name:>10}  {f'({code.ell},{code.m})':>8}  {code.n:>4}  "
              f"{code.k:>4}  {code.rank:>4}  {d:>5}  {method:>16}  {flag}")
        ds.append(d)
        ns.append(code.n)

    print()
    grows = all(b > a for a, b in zip(ds, ds[1:])) and len(ds) > 1
    flat = len(set(ds)) == 1
    if flat:
        print(f"VERDICT: d is FLAT at {ds[0]} across all sizes — distance-capped "
              "family. No finite-size-scaling collapse exists; do not re-sweep "
              "this family expecting a clean nu.")
    elif grows:
        print("VERDICT: d INCREASES with size — collapse-viable. Worth a fine "
              "BP/Metropolis sweep on this family for the Day 18 collapse.")
    else:
        print(f"VERDICT: d is non-monotonic across sizes ({ds}). Not a clean "
              "scaling family; inspect the structural flags above for which "
              "supports fall into proper subgroups.")


if __name__ == "__main__":
    main()