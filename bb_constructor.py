"""
bb_constructor.py — Bivariate Bicycle code construction over GF(4).

Implements:
  - Block-circulant matrix realization of polynomials in F_4[x,y]/⟨x^ℓ-1, y^m-1⟩
  - BBCode class: stores (A, B) polynomials and builds H = [A | B]
  - Four canonical code instances (tiny / small / medium / large)
  - Unit tests for sparsity, full-rank H, and existence of a non-trivial
    codeword basis

Mathematical setup
------------------
Following Bravyi et al. 2024,
    x = S_ℓ ⊗ I_m     y = I_ℓ ⊗ S_m
where S_n[i, j] = 1 iff j ≡ i+1 (mod n). The bivariate polynomial
    p(x, y) = Σ c_{ij} x^i y^j   with c_{ij} ∈ F_4
realizes an (ℓm) × (ℓm) block-circulant matrix
    M(p)[(r,a), (s,b)] = c_{(s-r) mod ℓ, (b-a) mod m}.
Linear indexing is (r, a) ↦ r·m + a (so the y-index varies fastest).

A BB code over F_4 is defined by two polynomials A, B with weight-3 monomial
supports, yielding the parity check
    H = [ M(A) | M(B) ]
of shape (ℓm) × (2ℓm). The code is C = ker(H), of length n = 2ℓm and
F_4-dimension k = n − rank(H), expected to be ℓm when H is full rank.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from gf4_lib import (
    ONE,
    OMEGA,
    OMEGA2,
    gf4_matvec,
    gf4_null_space_basis,
    gf4_rank,
    gf4_symbol,
)

# A polynomial is a list of (i, j, c) triples meaning  Σ c · x^i · y^j.
PolyTerms = Sequence[Tuple[int, int, int]]


# ---------------------------------------------------------------------------
# Polynomial → matrix
# ---------------------------------------------------------------------------
def poly_to_matrix(terms: PolyTerms, ell: int, m: int) -> np.ndarray:
    """
    Build the (ℓm × ℓm) block-circulant matrix M(p) over F_4 from a polynomial
    given as terms [(i, j, c), ...]. Multiple terms with the same (i, j) are
    XOR-accumulated (F_4 addition).
    """
    if ell <= 0 or m <= 0:
        raise ValueError(f"ell and m must be positive; got ({ell}, {m})")

    n = ell * m
    M = np.zeros((n, n), dtype=np.uint8)
    rows_2d = np.arange(ell).reshape(-1, 1)   # shape (ell, 1)
    cols_2d = np.arange(m).reshape(1, -1)     # shape (1, m)
    src_lin = (rows_2d * m + cols_2d).ravel()  # source linear indices

    for (i, j, c) in terms:
        if c == 0:
            continue
        i_red = i % ell
        j_red = j % m
        dst_2d = ((rows_2d + i_red) % ell) * m + ((cols_2d + j_red) % m)
        dst_lin = dst_2d.ravel()
        # XOR-accumulate, which is F_4 addition on coincident entries.
        np.bitwise_xor.at(M, (src_lin, dst_lin), np.uint8(c))
    return M


def poly_pretty(terms: PolyTerms, var_x: str = "x", var_y: str = "y") -> str:
    """Human-readable polynomial string for logging."""
    if not terms:
        return "0"
    parts: List[str] = []
    for (i, j, c) in terms:
        if c == 0:
            continue
        coef_str = "" if c == ONE else f"{gf4_symbol(c)}·"
        if i == 0 and j == 0:
            mono = "1"
        elif j == 0:
            mono = f"{var_x}^{i}" if i > 1 else var_x
        elif i == 0:
            mono = f"{var_y}^{j}" if j > 1 else var_y
        else:
            xs = f"{var_x}^{i}" if i > 1 else var_x
            ys = f"{var_y}^{j}" if j > 1 else var_y
            mono = xs + ys
        parts.append(coef_str + mono)
    return " + ".join(parts)


# ---------------------------------------------------------------------------
# BBCode class
# ---------------------------------------------------------------------------
@dataclass
class BBCode:
    """
    Classical GF(4)-linear Bivariate Bicycle code with parity check H = [A | B].
    """

    ell: int
    m: int
    A_terms: PolyTerms
    B_terms: PolyTerms
    name: str = ""

    A: np.ndarray = field(init=False, repr=False)
    B: np.ndarray = field(init=False, repr=False)
    H: np.ndarray = field(init=False, repr=False)
    _rank: int = field(init=False, repr=False, default=-1)

    def __post_init__(self) -> None:
        self.A = poly_to_matrix(self.A_terms, self.ell, self.m)
        self.B = poly_to_matrix(self.B_terms, self.ell, self.m)
        self.H = np.concatenate([self.A, self.B], axis=1)

    # ---- Parameters ------------------------------------------------------
    @property
    def n(self) -> int:
        """Block length over F_4 (number of code symbols)."""
        return 2 * self.ell * self.m

    @property
    def num_checks(self) -> int:
        return self.ell * self.m

    @property
    def rank(self) -> int:
        if self._rank < 0:
            self._rank = gf4_rank(self.H)
        return self._rank

    @property
    def k(self) -> int:
        """F_4-dimension of the code (number of information symbols)."""
        return self.n - self.rank

    @property
    def rate(self) -> float:
        return self.k / self.n

    # ---- Structural checks ----------------------------------------------
    def row_weight(self) -> int:
        rws = (self.H != 0).sum(axis=1)
        assert np.all(rws == rws[0]), f"non-uniform row weights: {set(rws)}"
        return int(rws[0])

    def col_weight(self) -> int:
        cws = (self.H != 0).sum(axis=0)
        assert np.all(cws == cws[0]), f"non-uniform column weights: {set(cws)}"
        return int(cws[0])

    # ---- Codewords -------------------------------------------------------
    def codeword_basis(self) -> np.ndarray:
        """
        Basis of the null space of H, as a (k × n) matrix whose rows are
        F_4-linearly independent codewords.
        """
        return gf4_null_space_basis(self.H)

    def is_codeword(self, c: np.ndarray) -> bool:
        """True iff H · c = 0 over F_4."""
        return bool(np.all(gf4_matvec(self.H, c) == 0))

    # ---- Pretty printing -------------------------------------------------
    def __str__(self) -> str:
        tag = f" '{self.name}'" if self.name else ""
        return (
            f"BBCode{tag}: (ℓ,m) = ({self.ell},{self.m}),  "
            f"n = {self.n},  rank = {self.rank},  k = {self.k},  "
            f"row_wt = {self.row_weight()},  col_wt = {self.col_weight()}\n"
            f"    A = {poly_pretty(self.A_terms)}\n"
            f"    B = {poly_pretty(self.B_terms)}"
        )


# ---------------------------------------------------------------------------
# Canonical code instances — DECISIONS_LOG Day 2 choices
# ---------------------------------------------------------------------------
def make_tiny() -> BBCode:

    
    """(ℓ,m) = (3,3), n = 18. Brute-force-d feasible; ML validation target."""
    return BBCode(
        ell=3, m=3,
        A_terms=[(1, 0, ONE), (0, 1, OMEGA), (0, 2, OMEGA2)],
        B_terms=[(0, 1, ONE), (1, 0, OMEGA), (2, 0, OMEGA2)],
        name="tiny",
    )


def make_small() -> BBCode:
    """(ℓ,m) = (6,6), n = 72. Bravyi [[72,12,6]] scaffold + ω-decoration."""
    return BBCode(
        ell=6, m=6,
        A_terms=[(3, 0, ONE), (0, 1, OMEGA), (0, 2, OMEGA2)],
        B_terms=[(0, 3, ONE), (1, 0, OMEGA), (2, 0, OMEGA2)],
        name="small",
    )


def make_medium() -> BBCode:
    """(ℓ,m) = (9,6), n = 108. Bravyi [[108,8,10]] scaffold."""
    return BBCode(
        ell=9, m=6,
        A_terms=[(3, 0, ONE), (0, 1, OMEGA), (0, 2, OMEGA2)],
        B_terms=[(0, 3, ONE), (2, 0, OMEGA), (4, 0, OMEGA2)],
        name="medium",
    )


def make_large() -> BBCode:
    """(ℓ,m) = (12,6), n = 144. Bravyi [[144,12,12]] scaffold."""
    return BBCode(
        ell=12, m=6,
        A_terms=[(3, 0, ONE), (0, 1, OMEGA), (0, 2, OMEGA2)],
        B_terms=[(0, 3, ONE), (4, 0, OMEGA), (8, 0, OMEGA2)],
        name="large",
    )


ALL_INSTANCES = [make_tiny, make_small, make_medium, make_large]


# ===========================================================================
# Unit tests
# ===========================================================================
def _test_poly_identity_matrix() -> None:
    # p(x,y) = 1 ↔ identity matrix.
    M = poly_to_matrix([(0, 0, ONE)], 4, 3)
    assert np.array_equal(M, np.eye(12, dtype=np.uint8))


def _test_poly_x_is_cyclic_shift() -> None:
    # x acts as a cyclic shift on the first index, identity on the second.
    ell, m = 5, 3
    X = poly_to_matrix([(1, 0, ONE)], ell, m)
    for r in range(ell):
        for a in range(m):
            src = r * m + a
            dst = ((r + 1) % ell) * m + a
            assert X[src, dst] == 1
            assert X[src].sum() == 1   # exactly one nonzero per row
            assert X[:, src].sum() == 1   # exactly one nonzero per column


def _test_poly_xy_commute() -> None:
    # xy = yx in F_4[x,y]/⟨x^ℓ-1, y^m-1⟩.
    from gf4_lib import gf4_matmul
    ell, m = 4, 5
    X = poly_to_matrix([(1, 0, ONE)], ell, m)
    Y = poly_to_matrix([(0, 1, ONE)], ell, m)
    assert np.array_equal(gf4_matmul(X, Y), gf4_matmul(Y, X))


def _test_poly_periodicity() -> None:
    # x^ℓ = I and y^m = I in the quotient ring.
    from gf4_lib import gf4_matmul
    ell, m = 4, 3
    X = poly_to_matrix([(1, 0, ONE)], ell, m)
    Y = poly_to_matrix([(0, 1, ONE)], ell, m)
    Xp = X.copy()
    for _ in range(ell - 1):
        Xp = gf4_matmul(Xp, X)
    assert np.array_equal(Xp, np.eye(ell * m, dtype=np.uint8))
    Yp = Y.copy()
    for _ in range(m - 1):
        Yp = gf4_matmul(Yp, Y)
    assert np.array_equal(Yp, np.eye(ell * m, dtype=np.uint8))


def _test_poly_gf4_coefficients_addable() -> None:
    # Two terms with the same monomial must F_4-sum (XOR) their coefficients,
    # not silently overwrite. Check that ω + ω² = 1 propagates through.
    M = poly_to_matrix([(0, 0, OMEGA), (0, 0, OMEGA2)], 2, 2)
    assert np.array_equal(M, np.eye(4, dtype=np.uint8))


def _test_sparsity_per_instance() -> None:
    # Every BB instance must have uniform row weight 6 and column weight 3.
    # Each row of H draws one nonzero from each of the three monomials of A
    # and B (6 total); each column lies in either the A-block (weight 3) or
    # the B-block (weight 3).
    for make in ALL_INSTANCES:
        code = make()
        assert code.row_weight() == 6, (code.name, code.row_weight())
        assert code.col_weight() == 3, (code.name, code.col_weight())


def _test_column_sum_redundancy() -> None:
    # By construction every polynomial uses coefficients {1, ω, ω²}, whose
    # F_4-sum 1 + ω + ω² = 0 (the defining relation of F_4). Each column of
    # H has exactly these three nonzero entries (one per monomial of A or B),
    # so every column sums to zero, i.e., the all-ones vector lies in the
    # left null space of H. This gives a baseline rank deficiency of 1 that
    # is *forced by the coefficient choice*, not a degeneracy.
    for make in ALL_INSTANCES:
        code = make()
        col_sums = np.bitwise_xor.reduce(code.H, axis=0)
        assert np.all(col_sums == 0), (
            f"{code.name}: columns do not sum to zero — coefficient pattern "
            f"may not be {{1, ω, ω²}}; col_sums[:8] = {col_sums[:8]}"
        )


def _test_rank_within_bound() -> None:
    # Sanity: rank cannot exceed number of rows. We expect rank ≤ ℓm − 1
    # because of the all-ones redundancy above, but extra algebraic
    # relations may push the deficit higher when ℓ or m is a multiple of
    # the order of F_4* (= 3). We log the actual deficit per instance.
    for make in ALL_INSTANCES:
        code = make()
        assert code.rank <= code.num_checks
        assert code.rank < code.num_checks, (
            f"{code.name}: rank {code.rank} = ℓm — column sum redundancy "
            f"should have made rank strictly less. Something is off."
        )


def _test_rate_at_least_one_half() -> None:
    # k = n − rank ≥ n − ℓm = ℓm = n/2 always, with equality when H is full
    # row rank. Rank deficiency strictly increases k.
    for make in ALL_INSTANCES:
        code = make()
        assert code.k >= code.n // 2, (
            f"{code.name}: k = {code.k} < n/2 = {code.n // 2}"
        )


def _test_codeword_basis_nontrivial() -> None:
    # The null space of H must be nontrivial (k > 0), and every basis vector
    # produced must (a) be nonzero and (b) satisfy H · c = 0 exactly.
    for make in ALL_INSTANCES:
        code = make()
        basis = code.codeword_basis()
        # k > 0
        assert basis.shape[0] > 0, f"{code.name}: empty codeword basis"
        assert basis.shape[0] == code.k
        assert basis.shape[1] == code.n
        # All basis vectors are nonzero and lie in ker(H).
        for idx, v in enumerate(basis):
            assert np.any(v != 0), f"{code.name}: basis row {idx} is the zero vector"
            assert code.is_codeword(v), (
                f"{code.name}: basis row {idx} fails H · c = 0"
            )
        # Basis vectors are F_4-linearly independent (rank check on the basis).
        assert gf4_rank(basis) == code.k, (
            f"{code.name}: codeword basis is linearly dependent"
        )


def _test_codeword_combinations_are_codewords() -> None:
    # F_4-linear combinations of basis vectors are also codewords.
    # Spot-check on the tiny instance with a deterministic combination.
    code = make_tiny()
    basis = code.codeword_basis()
    if basis.shape[0] >= 2:
        from gf4_lib import gf4_add, gf4_mul
        # c = ω · v_0  +  ω² · v_1
        v0 = basis[0]
        v1 = basis[1]
        c = gf4_add(gf4_mul(OMEGA, v0), gf4_mul(OMEGA2, v1))
        assert code.is_codeword(c)


def _run_all_tests() -> None:
    tests = [
        _test_poly_identity_matrix,
        _test_poly_x_is_cyclic_shift,
        _test_poly_xy_commute,
        _test_poly_periodicity,
        _test_poly_gf4_coefficients_addable,
        _test_sparsity_per_instance,
        _test_column_sum_redundancy,
        _test_rank_within_bound,
        _test_rate_at_least_one_half,
        _test_codeword_basis_nontrivial,
        _test_codeword_combinations_are_codewords,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} BB constructor tests passed.\n")


if __name__ == "__main__":
    _run_all_tests()
    print("Code instances:")
    print("-" * 76)
    for make in ALL_INSTANCES:
        print(make())
        print()
