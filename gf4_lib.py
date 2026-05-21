"""
gf4_lib.py — Arithmetic over GF(4) for the BB code project.

Field definition
----------------
F_4 = F_2[ω] / ⟨ω² + ω + 1⟩, so ω² = ω + 1 and ω³ = 1.

Representation
--------------
Each element a + bω with a, b ∈ {0,1} is encoded as the integer

    (b << 1) | a   ∈   {0, 1, 2, 3}.

Concretely:
    0   ↔ 0b00 = 0
    1   ↔ 0b01 = 1
    ω   ↔ 0b10 = 2
    ω²  ↔ 0b11 = 3   (since ω² = 1 + ω)

This makes addition the bitwise XOR of the 2-bit encodings (characteristic 2),
and lets every routine accept either Python scalars or NumPy arrays of
unsigned integers without any special casing.

Author: BB-codes paper project, Day 1
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Element constants
# ---------------------------------------------------------------------------
ZERO: int = 0
ONE: int = 1
OMEGA: int = 2          # ω
OMEGA2: int = 3         # ω² = ω + 1

ELEMENTS: tuple[int, int, int, int] = (ZERO, ONE, OMEGA, OMEGA2)

# ---------------------------------------------------------------------------
# Precomputed tables
# ---------------------------------------------------------------------------
# Multiplication table: _MUL[a, b] = a * b in F_4.
# Derived from ω² = ω + 1 (so ω² ↔ 3) and ω³ = 1:
#   ω·ω   = ω²        = 3
#   ω·ω²  = ω³        = 1
#   ω²·ω² = ω⁴ = ω·ω³ = ω = 2
_MUL_TABLE: np.ndarray = np.array(
    [
        [0, 0, 0, 0],   # 0 · *
        [0, 1, 2, 3],   # 1 · *
        [0, 2, 3, 1],   # ω · *
        [0, 3, 1, 2],   # ω² · *
    ],
    dtype=np.uint8,
)

# Multiplicative inverses: F_4* is cyclic of order 3, ω³ = 1.
# Therefore  1⁻¹ = 1,  ω⁻¹ = ω²,  (ω²)⁻¹ = ω.
# Index 0 is a sentinel; calling code must never request inv(0).
_INV_TABLE: np.ndarray = np.array([0, 1, 3, 2], dtype=np.uint8)

# Discrete log / antilog with base ω.
#   ω⁰ = 1,  ω¹ = ω,  ω² = ω².
# _LOG[x] is defined for x ∈ {1, 2, 3}; _LOG[0] is sentinel 255.
_LOG_TABLE: np.ndarray = np.array([255, 0, 1, 2], dtype=np.uint8)
_EXP_TABLE: np.ndarray = np.array([1, 2, 3], dtype=np.uint8)  # ω^k for k = 0,1,2


# ---------------------------------------------------------------------------
# Basic arithmetic — scalar- and array-friendly
# ---------------------------------------------------------------------------
def gf4_add(a, b):
    """Addition in F_4. Equivalent to XOR of the 2-bit representations."""
    return np.bitwise_xor(a, b).astype(np.uint8, copy=False)


def gf4_sub(a, b):
    """Subtraction. In characteristic 2 this is identical to addition."""
    return np.bitwise_xor(a, b).astype(np.uint8, copy=False)


def gf4_mul(a, b):
    """Multiplication in F_4 via lookup table. Broadcasts over arrays."""
    a_arr = np.asarray(a, dtype=np.uint8)
    b_arr = np.asarray(b, dtype=np.uint8)
    return _MUL_TABLE[a_arr, b_arr]


def gf4_inv(a):
    """Multiplicative inverse in F_4. Raises on any zero input."""
    a_arr = np.asarray(a, dtype=np.uint8)
    if np.any(a_arr == 0):
        raise ZeroDivisionError("0 has no multiplicative inverse in F_4")
    return _INV_TABLE[a_arr]


def gf4_div(a, b):
    """Division a / b in F_4. Raises on any zero divisor."""
    return gf4_mul(a, gf4_inv(b))


def gf4_pow(a, n: int):
    """
    Exponentiation a^n in F_4 for a scalar or array a and an integer n.

    Conventions:
        - 0^0 returns 1 (the convenient algebraic convention).
        - 0^n = 0 for n > 0.
        - 0^n for n < 0 raises ZeroDivisionError.
        - For nonzero a, exponent reduces mod 3 since |F_4*| = 3.
    """
    a_arr = np.asarray(a, dtype=np.uint8)

    if n == 0:
        return np.ones_like(a_arr, dtype=np.uint8)

    if n < 0:
        if np.any(a_arr == 0):
            raise ZeroDivisionError("Cannot raise 0 to a negative power")
        a_arr = _INV_TABLE[a_arr]
        n = -n

    n_red = n % 3
    if n_red == 0:
        # a^n = 1 for nonzero a, 0 for zero a.
        return (a_arr != 0).astype(np.uint8)

    result = a_arr.copy()
    for _ in range(n_red - 1):
        result = _MUL_TABLE[result, a_arr]
    return result


def gf4_trace(a):
    """
    Trace map Tr : F_4 → F_2, Tr(x) = x + x².

    Values:  Tr(0) = 0,  Tr(1) = 0,  Tr(ω) = 1,  Tr(ω²) = 1.

    With our representation x = (b<<1) | a, the trace is simply the high bit b.
    """
    return ((np.asarray(a, dtype=np.uint8) >> 1) & 1).astype(np.uint8, copy=False)


def gf4_conjugate(a):
    """
    The nontrivial Galois automorphism of F_4 / F_2, x ↦ x².

    Values:  0 ↦ 0,  1 ↦ 1,  ω ↦ ω²,  ω² ↦ ω.
    Implemented as a swap of the two nonzero non-identity elements.
    """
    a_arr = np.asarray(a, dtype=np.uint8)
    # Swap 2 ↔ 3, leave 0 and 1 alone.
    return np.where(a_arr == 2, 3, np.where(a_arr == 3, 2, a_arr)).astype(np.uint8)


# ---------------------------------------------------------------------------
# Pair / bit helpers (useful for I/O, debugging, and BP message indexing)
# ---------------------------------------------------------------------------
def gf4_from_pair(low, high):
    """Build x = low + high·ω from F_2 bits low, high ∈ {0,1}."""
    low_arr = np.asarray(low, dtype=np.uint8) & 1
    high_arr = np.asarray(high, dtype=np.uint8) & 1
    return ((high_arr << 1) | low_arr).astype(np.uint8, copy=False)


def gf4_to_pair(a):
    """Inverse of gf4_from_pair: return (low, high) bits of x = low + high·ω."""
    a_arr = np.asarray(a, dtype=np.uint8)
    return (a_arr & 1).astype(np.uint8), ((a_arr >> 1) & 1).astype(np.uint8)


def gf4_symbol(a) -> str:
    """Human-readable symbol for a scalar F_4 element (for logging)."""
    return {0: "0", 1: "1", 2: "ω", 3: "ω²"}[int(a)]


# ---------------------------------------------------------------------------
# Linear algebra over F_4 (lightweight; full matrix ops live in bb_constructor)
# ---------------------------------------------------------------------------
def gf4_matmul(A, B):
    """
    Matrix product A · B over F_4.
    A: shape (m, k), B: shape (k, n). Returns shape (m, n) in uint8.

    Uses XOR-summation of table-multiplied entries; vectorized over the inner
    dimension. Adequate for the parity-check matrix sizes we will work with.
    """
    A_arr = np.asarray(A, dtype=np.uint8)
    B_arr = np.asarray(B, dtype=np.uint8)
    if A_arr.ndim != 2 or B_arr.ndim != 2 or A_arr.shape[1] != B_arr.shape[0]:
        raise ValueError(
            f"Incompatible shapes for gf4_matmul: {A_arr.shape} · {B_arr.shape}"
        )
    # Broadcast multiplication then XOR-reduce along the shared axis.
    # Result[i, j] = XOR_k MUL[A[i,k], B[k,j]].
    prod = _MUL_TABLE[A_arr[:, :, None], B_arr[None, :, :]]  # (m, k, n)
    return np.bitwise_xor.reduce(prod, axis=1).astype(np.uint8)


def gf4_matvec(M, v):
    """
    Matrix-vector product M · v over F_4.

    M: shape (r, n), v: shape (n,). Returns shape (r,) in uint8.
    Implemented as table-lookup multiplication followed by XOR-reduction.
    """
    M_arr = np.asarray(M, dtype=np.uint8)
    v_arr = np.asarray(v, dtype=np.uint8)
    if M_arr.ndim != 2 or v_arr.ndim != 1 or M_arr.shape[1] != v_arr.shape[0]:
        raise ValueError(
            f"Incompatible shapes for gf4_matvec: {M_arr.shape} · {v_arr.shape}"
        )
    prod = _MUL_TABLE[M_arr, v_arr[None, :]]            # (r, n)
    return np.bitwise_xor.reduce(prod, axis=1).astype(np.uint8)


def gf4_rref(M):
    """
    Reduce M (uint8 array of F_4 elements) to reduced row echelon form.

    Returns (R, pivot_cols) where R is in RREF (each pivot is 1, and the
    pivot column has zeros elsewhere) and pivot_cols is the list of column
    indices containing pivots, in increasing order.

    Standard Gauss-Jordan over F_4. The matrix is copied; M is not mutated.
    """
    R = np.asarray(M, dtype=np.uint8).copy()
    n_rows, n_cols = R.shape
    pivot_cols: list[int] = []
    r = 0  # next row to fill with a pivot

    for c in range(n_cols):
        if r >= n_rows:
            break
        # Find a pivot row at or below row r with a nonzero in column c.
        pivot_row = -1
        for rr in range(r, n_rows):
            if R[rr, c] != 0:
                pivot_row = rr
                break
        if pivot_row < 0:
            continue  # column c has no pivot; move on
        # Swap pivot up.
        if pivot_row != r:
            R[[r, pivot_row]] = R[[pivot_row, r]]
        # Normalize the pivot row so the leading entry is 1.
        inv = _INV_TABLE[R[r, c]]
        if inv != 1:
            R[r] = _MUL_TABLE[R[r], inv]
        # Eliminate column c from every other row.
        for rr in range(n_rows):
            if rr != r and R[rr, c] != 0:
                factor = R[rr, c]
                R[rr] ^= _MUL_TABLE[R[r], factor]
        pivot_cols.append(c)
        r += 1

    return R, pivot_cols


def gf4_rank(M):
    """Rank of M over F_4 (number of pivots in RREF)."""
    _, pivots = gf4_rref(M)
    return len(pivots)


def gf4_null_space_basis(M):
    """
    Basis of the right null space {v : M·v = 0} of M over F_4.

    Returned as a 2D array of shape (n - rank, n) whose rows are the basis
    vectors. Returns a (0, n) array if M has full column rank.

    For each free column c_f, one basis vector is constructed by setting
    x[c_f] = 1, x[c_g] = 0 for every other free column c_g, and reading off
    the pivot variables from the RREF rows. In characteristic 2 the pivot
    value is just R[i, c_f] (no sign flip needed).
    """
    M_arr = np.asarray(M, dtype=np.uint8)
    n_rows, n_cols = M_arr.shape
    R, pivot_cols = gf4_rref(M_arr)
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n_cols) if c not in pivot_set]

    if not free_cols:
        return np.zeros((0, n_cols), dtype=np.uint8)

    basis = np.zeros((len(free_cols), n_cols), dtype=np.uint8)
    for k, c_f in enumerate(free_cols):
        basis[k, c_f] = 1
        # Pivot rows tell us how the pivot variables depend on the free ones.
        for i, c_p in enumerate(pivot_cols):
            basis[k, c_p] = R[i, c_f]
    return basis


# ===========================================================================
# Unit tests
# ===========================================================================
def _test_addition_table() -> None:
    # F_2-vector-space addition is XOR; verify the full 4×4 table.
    expected = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, 3, 2],
            [2, 3, 0, 1],
            [3, 2, 1, 0],
        ],
        dtype=np.uint8,
    )
    for a in ELEMENTS:
        for b in ELEMENTS:
            assert int(gf4_add(a, b)) == expected[a, b], (a, b)


def _test_multiplication_table() -> None:
    # Spot-check every nonzero relation from ω² = ω + 1 and ω³ = 1.
    assert int(gf4_mul(OMEGA, OMEGA)) == OMEGA2                # ω·ω = ω²
    assert int(gf4_mul(OMEGA, OMEGA2)) == ONE                  # ω·ω² = 1
    assert int(gf4_mul(OMEGA2, OMEGA2)) == OMEGA               # (ω²)² = ω
    # Identity and absorbing element.
    for x in ELEMENTS:
        assert int(gf4_mul(ZERO, x)) == ZERO
        assert int(gf4_mul(ONE, x)) == x
        assert int(gf4_mul(x, ONE)) == x
    # Commutativity (full table).
    for a in ELEMENTS:
        for b in ELEMENTS:
            assert int(gf4_mul(a, b)) == int(gf4_mul(b, a))


def _test_brief_validation() -> None:
    # The exact validation specified in PROJECT_BRIEF / PIPELINE Day 1:
    #     gf4_mul(ω, ω) == gf4_add(ω, 1)        because ω² = ω + 1.
    assert int(gf4_mul(OMEGA, OMEGA)) == int(gf4_add(OMEGA, ONE))


def _test_distributivity() -> None:
    # a · (b + c) = a·b + a·c, exhaustively over F_4³.
    for a in ELEMENTS:
        for b in ELEMENTS:
            for c in ELEMENTS:
                lhs = int(gf4_mul(a, gf4_add(b, c)))
                rhs = int(gf4_add(gf4_mul(a, b), gf4_mul(a, c)))
                assert lhs == rhs, (a, b, c)


def _test_inverses() -> None:
    # x · x⁻¹ = 1 for every nonzero x.
    for x in (ONE, OMEGA, OMEGA2):
        assert int(gf4_mul(x, gf4_inv(x))) == ONE
    # Zero has no inverse.
    try:
        gf4_inv(ZERO)
    except ZeroDivisionError:
        pass
    else:
        raise AssertionError("gf4_inv(0) should raise ZeroDivisionError")


def _test_power() -> None:
    # ω³ = 1, ω⁴ = ω, ω⁻¹ = ω², and the 0^0 = 1 convention.
    assert int(gf4_pow(OMEGA, 3)) == ONE
    assert int(gf4_pow(OMEGA, 4)) == OMEGA
    assert int(gf4_pow(OMEGA, -1)) == OMEGA2
    assert int(gf4_pow(ZERO, 0)) == ONE
    assert int(gf4_pow(ZERO, 5)) == ZERO


def _test_trace() -> None:
    # Hand-computed values: Tr(x) = x + x².
    assert int(gf4_trace(ZERO)) == 0
    assert int(gf4_trace(ONE)) == 0
    assert int(gf4_trace(OMEGA)) == 1
    assert int(gf4_trace(OMEGA2)) == 1
    # F_2-linearity of Tr.
    for a in ELEMENTS:
        for b in ELEMENTS:
            assert int(gf4_trace(gf4_add(a, b))) == (
                int(gf4_trace(a)) ^ int(gf4_trace(b))
            )


def _test_conjugate_is_frobenius() -> None:
    # x² equals the Frobenius/conjugate.
    for x in ELEMENTS:
        assert int(gf4_conjugate(x)) == int(gf4_mul(x, x))


def _test_array_broadcasting() -> None:
    # Verify all ops vectorize correctly over NumPy arrays.
    a = np.array([0, 1, 2, 3, 2], dtype=np.uint8)
    b = np.array([1, 2, 3, 0, 2], dtype=np.uint8)
    sums = gf4_add(a, b)
    prods = gf4_mul(a, b)
    for i in range(len(a)):
        assert int(sums[i]) == int(gf4_add(int(a[i]), int(b[i])))
        assert int(prods[i]) == int(gf4_mul(int(a[i]), int(b[i])))


def _test_pair_roundtrip() -> None:
    # (low, high) ↔ element round-trip on all 4 elements.
    for x in ELEMENTS:
        low, high = gf4_to_pair(x)
        assert int(gf4_from_pair(low, high)) == x


def _test_matmul_identity() -> None:
    # I · A = A · I = A for the 3×3 GF(4) identity and a small test matrix.
    I = np.eye(3, dtype=np.uint8)
    A = np.array(
        [
            [1, 2, 3],
            [0, 1, 2],
            [3, 0, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(gf4_matmul(I, A), A)
    assert np.array_equal(gf4_matmul(A, I), A)


def _test_matvec_basic() -> None:
    # Hand-computable check: [1, ω] · [ω², 1]ᵀ = 1·ω² + ω·1 = ω² + ω = 1.
    M = np.array([[ONE, OMEGA]], dtype=np.uint8)
    v = np.array([OMEGA2, ONE], dtype=np.uint8)
    assert int(gf4_matvec(M, v)[0]) == ONE
    # Vectorized vs. loop-based matmul agreement on a 4×4 example.
    A = np.array(
        [
            [1, 2, 3, 0],
            [0, 1, 2, 3],
            [3, 0, 1, 2],
            [2, 3, 0, 1],
        ],
        dtype=np.uint8,
    )
    x = np.array([2, 3, 1, 2], dtype=np.uint8)
    direct = gf4_matvec(A, x)
    via_matmul = gf4_matmul(A, x.reshape(-1, 1)).ravel()
    assert np.array_equal(direct, via_matmul)


def _test_rref_simple() -> None:
    # 2x2 identity is already in RREF.
    I = np.eye(2, dtype=np.uint8)
    R, pivots = gf4_rref(I)
    assert pivots == [0, 1]
    assert np.array_equal(R, I)


def _test_rank_rank_deficient() -> None:
    # Two F_4-dependent rows: [1, ω] and [ω, ω²] = ω · [1, ω]. Rank 1.
    M = np.array([[ONE, OMEGA], [OMEGA, OMEGA2]], dtype=np.uint8)
    assert gf4_rank(M) == 1
    # Identity matrices have full rank.
    for n in (1, 3, 5):
        assert gf4_rank(np.eye(n, dtype=np.uint8)) == n
    # Zero matrix has rank 0.
    assert gf4_rank(np.zeros((4, 4), dtype=np.uint8)) == 0


def _test_null_space_residual() -> None:
    # Every null-space basis vector v must satisfy M · v = 0, exactly.
    M = np.array(
        [
            [ONE, OMEGA,  OMEGA2, ONE],
            [ZERO, ONE,   OMEGA,  OMEGA2],
        ],
        dtype=np.uint8,
    )
    basis = gf4_null_space_basis(M)
    # Dimension: 4 columns − 2 pivots = 2 free vars → 2 basis vectors.
    assert basis.shape == (2, 4)
    for v in basis:
        assert np.all(gf4_matvec(M, v) == 0)
    # Basis vectors are F_4-linearly independent: stack and check rank.
    assert gf4_rank(basis) == 2


def _test_null_space_full_rank() -> None:
    # If M has full column rank, null space is trivial (no basis vectors).
    M = np.eye(3, dtype=np.uint8)
    basis = gf4_null_space_basis(M)
    assert basis.shape == (0, 3)


def _run_all_tests() -> None:
    tests = [
        _test_addition_table,
        _test_multiplication_table,
        _test_brief_validation,
        _test_distributivity,
        _test_inverses,
        _test_power,
        _test_trace,
        _test_conjugate_is_frobenius,
        _test_array_broadcasting,
        _test_pair_roundtrip,
        _test_matmul_identity,
        _test_matvec_basic,
        _test_rref_simple,
        _test_rank_rank_deficient,
        _test_null_space_residual,
        _test_null_space_full_rank,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} unit tests passed.")


if __name__ == "__main__":
    _run_all_tests()