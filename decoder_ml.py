"""
decoder_ml.py — Brute-force Maximum-Likelihood decoder for GF(4) BB codes.

Used as the ground-truth reference against which BP-FFT (decoder_bp.py) is
validated on the tiny code at Day 12. Not intended for production decoding:
runtime is O(4^k · n) per received word, so this is only practical for
k ≲ 12.

ML for the Quaternary Symmetric Channel reduces to minimum-Hamming-distance
=========================================================================

For the QSC defined in channel.py, given a received word y and any candidate
codeword c with Hamming distance D = #{i : c_i ≠ y_i} (and A = n − D
agreements),

    log Pr[y | c] = A · log(1 − p) + D · log(p / 3)
                 = n · log(1 − p) − D · μ,
    where      μ = log((1 − p) / (p / 3)).

With μ > 0 (i.e. p < 3/4, the entire useful range of the QSC), maximising
log-likelihood is equivalent to minimising Hamming distance. Under a uniform
prior over codewords MAP = ML, so we report the minimum-Hamming-distance
codeword. Ties (multiple codewords at the same distance) are broken by the
smaller enumeration index — this only affects the post-failure symbol
identity, never the FER count.

The min-Hamming-distance form is preferred to the explicit log-likelihood
sum because (a) it is exact in integer arithmetic, (b) it avoids underflow
in evaluating p^D, (c) it makes the equivalence to bounded-distance decoders
transparent: ML corrects every error pattern of weight ≤ ⌊(d − 1) / 2⌋ for
a code of minimum distance d.

Codeword enumeration
====================
The basis B (k × n over F_4) returned by `code.codeword_basis()` spans the
full code as { m · B : m ∈ F_4^k }, exhaustively as m ranges over the 4^k
F_4-messages. We build the codeword table by doubling: starting from {0},
at step j we replace the current set S by
        S_new = { c + α · B[j] : c ∈ S,  α ∈ F_4 }.
After k steps |S| = 4^k = the full code. Per-step intermediate memory is
4 · |S_old| · n bytes, peaking at the final step at 4 · 4^{k-1} · n. For
the tiny code (k = 10, n = 18) this peak is 19 MB and the final table is
18 MB — comfortable.

API contract
============
Conforms to evaluation.Decoder: callable as `dec(code, received, rng=None)
-> codeword`. The `rng` argument is ignored (ML is deterministic).

Tables are cached by id(code.H), matching the pattern in
decoder_metropolis.py: the harness re-uses one decoder instance across many
received words for a single code, so the (potentially expensive) codeword
enumeration is amortised.

Discrepancy with the pipeline budget
====================================
PIPELINE Day 11 specifies "4^k ≤ 2^14 (k ≤ 7)", which would exclude the
tiny instance (k = 10). However Day 3's `code_params.brute_force_min_distance`
already enumerates 4^10 ≈ 10^6 codewords against this same basis to compute
the exact minimum distance, so the scale is demonstrably tractable. We
default `max_k = 12` (4^12 = 16 M codewords, sufficient for any plausible
ML-validation use case) and reject larger codes at construction. This is
recorded in the Day 11 DECISION_LOG entry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bb_constructor import BBCode
from gf4_lib import _MUL_TABLE, gf4_matvec


def enumerate_codewords(basis: np.ndarray) -> np.ndarray:
    """
    Enumerate all 4^k codewords of the F_4-linear code spanned by `basis`
    (a k × n uint8 matrix). Returns a (4^k, n) uint8 array whose rows are
    the codewords in some deterministic order (the lex-order on the
    coefficient vectors over (0, 1, ω, ω²)).

    Construction is iterative doubling: at each step we replace the current
    table by its "outer sum" with the four F_4-scalings of the next basis
    vector. F_4 addition is XOR on the (b<<1)|a encoding, hence
    `np.bitwise_xor` here.

    Memory: peak intermediate is 4 · 4^{k-1} · n bytes (the last expansion
    step); final table is 4^k · n bytes.
    """
    k, n = basis.shape
    if k == 0:
        return np.zeros((1, n), dtype=np.uint8)

    # Start from the zero codeword and grow by factor of 4 per basis row.
    cw = np.zeros((1, n), dtype=np.uint8)
    for j in range(k):
        b_j = basis[j]                              # (n,)
        # The four F_4-scalings (0·b, 1·b, ω·b, ω²·b) of the current basis row.
        # _MUL_TABLE[c, b_j] broadcasts to (n,) for scalar c.
        b_scaled = np.stack(
            [_MUL_TABLE[c, b_j] for c in range(4)], axis=0
        ).astype(np.uint8)                          # (4, n)
        # Cartesian-product XOR: shape (|cw|, 4, n).
        expanded = np.bitwise_xor(cw[:, None, :], b_scaled[None, :, :])
        cw = expanded.reshape(-1, n)
    return cw


@dataclass
class MLDecoder:
    """
    Maximum-likelihood decoder by brute-force codeword enumeration.

    Parameters
    ----------
    p : float
        Channel error rate. Required to lie in (0, 3/4); above 3/4 the QSC
        ML decision rule inverts (would become maximum Hamming distance)
        and the channel has zero capacity at p = 3/4. The parameter is
        stored only for reporting / consistency with the Decoder protocol —
        the actual decision rule is min-Hamming, valid throughout (0, 3/4).
    max_k : int, default 12
        Refuse to build the codeword table when k exceeds this. Default 12
        corresponds to 4^12 = 16 M codewords (~256 MB at n = 16), the upper
        edge of tractability on a workstation. Caller can raise if they
        know what they're doing. Tiny BB instance is k = 10 (~1 M
        codewords), well inside the default.
    chunk_size : int, default 200_000
        Decoding evaluates Hamming distances in chunks of this many
        codewords at a time. Caps the temporary memory of the comparison
        bool array. Smaller values trade speed for memory; the default is
        a few MB temp.
    name : str, default ""
        Identifier carried into EvalResult.decoder_name. Auto-generated
        from `p` if blank.
    """

    p: float
    max_k: int = 12
    chunk_size: int = 200_000
    name: str = ""

    # Per-code codeword-table cache, keyed by id(code.H) — same pattern as
    # decoder_metropolis._idx_cache.
    _codeword_cache: dict = field(default_factory=dict, repr=False, compare=False)
    # μ = log((1 − p) / (p / 3)); positive iff p < 3/4. Computed for
    # consistency checks even though the decision rule uses Hamming distance.
    mu: float = field(init=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 0.75):
            raise ValueError(
                f"p must be in (0, 3/4) for min-Hamming-distance to equal "
                f"ML; got {self.p}. Above 3/4 the QSC has zero capacity "
                f"and the ML decision rule inverts."
            )
        if self.max_k < 0:
            raise ValueError(f"max_k must be ≥ 0; got {self.max_k}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive; got {self.chunk_size}")
        self.mu = math.log((1.0 - self.p) / (self.p / 3.0))
        if not self.name:
            self.name = f"ml_p{self.p:.4f}"

    # ---- lazy codeword-table construction -------------------------------
    def _get_codewords(self, code: BBCode) -> np.ndarray:
        key = id(code.H)
        cached = self._codeword_cache.get(key)
        if cached is not None:
            return cached
        if code.k > self.max_k:
            raise ValueError(
                f"ML decoder requires 4^k codewords; code {code.name!r} has "
                f"k = {code.k} (4^k = {4 ** code.k:.2e}), exceeding "
                f"max_k = {self.max_k}. Bump max_k to enumerate anyway."
            )
        table = enumerate_codewords(code.codeword_basis())
        # Defensive sanity: every entry must satisfy H · c = 0. We only spot-
        # check the first 8 rows to avoid making construction O(M · M·n).
        for row in table[: min(8, len(table))]:
            if not np.all(gf4_matvec(code.H, row) == 0):
                raise RuntimeError(
                    "enumerate_codewords produced a non-codeword; basis or "
                    "F_4 arithmetic is broken."
                )
        self._codeword_cache[key] = table
        return table

    # ---- public callable ------------------------------------------------
    def __call__(
        self,
        code: BBCode,
        received: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        codewords = self._get_codewords(code)
        y = np.asarray(received, dtype=np.uint8)
        if y.shape != (code.n,):
            raise ValueError(
                f"received has shape {y.shape}, expected ({code.n},)"
            )

        M, n = codewords.shape
        best_dist = n + 1            # strictly larger than any feasible distance
        best_idx = 0
        for start in range(0, M, self.chunk_size):
            end = min(start + self.chunk_size, M)
            # Hamming distance per chunk row. (chunk, n) bool != (n,) broadcast,
            # then sum along axis=1. uint8 != uint8 stays fast in numpy.
            dists = (codewords[start:end] != y).sum(axis=1)
            local_idx = int(np.argmin(dists))
            local_dist = int(dists[local_idx])
            if local_dist < best_dist:
                best_dist = local_dist
                best_idx = start + local_idx
            if best_dist == 0:        # exact match — cannot be improved upon
                break
        return codewords[best_idx].copy()


# ===========================================================================
# Unit tests
# ===========================================================================
def _test_enumerate_count_and_uniqueness() -> None:
    """For the tiny code, enumeration must produce exactly 4^k distinct codewords."""
    from bb_constructor import make_tiny
    code = make_tiny()
    basis = code.codeword_basis()
    table = enumerate_codewords(basis)
    assert table.shape == (4 ** code.k, code.n), table.shape
    # Uniqueness: sort lex and check no duplicate adjacent rows.
    sorted_table = table[np.lexsort(table.T[::-1])]
    diffs = np.any(np.diff(sorted_table, axis=0) != 0, axis=1)
    assert int(diffs.sum()) == 4 ** code.k - 1, (
        f"duplicate codewords detected: {4 ** code.k - 1 - int(diffs.sum())} dupes"
    )


def _test_enumerate_all_in_kernel() -> None:
    """Every enumerated codeword must satisfy H · c = 0 over F_4."""
    from bb_constructor import make_tiny
    code = make_tiny()
    table = enumerate_codewords(code.codeword_basis())
    # We spot-check rather than checking all 10⁶: random sample of 500 rows.
    rng = np.random.default_rng(42)
    sample = rng.choice(table.shape[0], size=500, replace=False)
    for idx in sample:
        assert np.all(gf4_matvec(code.H, table[idx]) == 0), idx


def _test_enumerate_includes_zero_and_basis() -> None:
    """The enumerated table must contain the zero codeword and every basis row."""
    from bb_constructor import make_tiny
    code = make_tiny()
    basis = code.codeword_basis()
    table = enumerate_codewords(basis)
    # Convert each row to a hashable tuple for set membership tests.
    table_set = {bytes(row) for row in table}
    assert bytes(np.zeros(code.n, dtype=np.uint8)) in table_set
    for row in basis:
        assert bytes(row.astype(np.uint8)) in table_set


def _test_zero_received_returns_zero() -> None:
    """y = 0 (zero codeword, no noise) decodes to the zero codeword."""
    from bb_constructor import make_tiny
    code = make_tiny()
    dec = MLDecoder(p=0.05)
    out = dec(code, np.zeros(code.n, dtype=np.uint8))
    assert np.array_equal(out, np.zeros(code.n, dtype=np.uint8))


def _test_codeword_received_returns_itself() -> None:
    """Noise-free transmission of any codeword decodes to that codeword."""
    from bb_constructor import make_tiny
    code = make_tiny()
    dec = MLDecoder(p=0.05)
    for cw in code.codeword_basis()[:5]:
        out = dec(code, cw)
        assert np.array_equal(out, cw), (cw, out)


def _test_every_single_error_corrected() -> None:
    """
    The tiny code has d = 6, so ML must correct every single-symbol error
    deterministically. Verify exhaustively over (n_positions × 3 error values).
    """
    from bb_constructor import make_tiny
    code = make_tiny()
    dec = MLDecoder(p=0.02)
    zero = np.zeros(code.n, dtype=np.uint8)
    failures = []
    for pos in range(code.n):
        for sym in (1, 2, 3):
            received = zero.copy()
            received[pos] = sym
            out = dec(code, received)
            if not np.array_equal(out, zero):
                failures.append((pos, sym, list(out)))
    assert not failures, (
        f"ML failed on {len(failures)} single-error patterns; first: "
        f"{failures[0]}"
    )


def _test_every_two_error_pattern_correctable_by_bdd() -> None:
    """
    The tiny code has d = 6, so any 2-symbol error is within the
    bounded-distance radius ⌊(d - 1)/2⌋ = 2 and must be uniquely corrected.
    We test a random sample of 200 weight-2 patterns for cost reasons.
    """
    from bb_constructor import make_tiny
    code = make_tiny()
    dec = MLDecoder(p=0.02)
    zero = np.zeros(code.n, dtype=np.uint8)
    rng = np.random.default_rng(11)
    n = code.n
    for trial in range(200):
        positions = rng.choice(n, size=2, replace=False)
        syms = rng.integers(1, 4, size=2, dtype=np.uint8)
        received = zero.copy()
        received[positions] = syms
        out = dec(code, received)
        assert np.array_equal(out, zero), (
            f"ML failed on weight-2 error: positions={positions}, syms={syms}, "
            f"output differs at {np.flatnonzero(out)}"
        )


def _test_output_is_always_a_codeword() -> None:
    """For any received word, the ML output must satisfy H · c = 0."""
    from bb_constructor import make_tiny
    from channel import QSC
    code = make_tiny()
    dec = MLDecoder(p=0.1)
    qsc = QSC(0.1)
    rng = np.random.default_rng(99)
    for _ in range(50):
        recv = qsc.sample_noise(code.n, rng)
        out = dec(code, recv)
        assert np.all(gf4_matvec(code.H, out) == 0), (recv, out)


def _test_ml_at_least_as_good_as_bp() -> None:
    """
    ML must beat (or tie) BP in Hamming distance to the true codeword on
    every single trial. Compare across 100 noise realisations at p = 0.05.
    """
    from bb_constructor import make_tiny
    from channel import QSC
    from decoder_bp import BPFFTDecoder
    code = make_tiny()
    ml = MLDecoder(p=0.05)
    bp = BPFFTDecoder(p=0.05, max_iters=50)
    qsc = QSC(0.05)
    rng = np.random.default_rng(13)
    zero = np.zeros(code.n, dtype=np.uint8)
    for trial in range(100):
        recv, _ = qsc.transmit(zero, rng)
        d_ml = int((ml(code, recv) != zero).sum())
        d_bp = int((bp(code, recv) != zero).sum())
        assert d_ml <= d_bp, (
            f"trial {trial}: ML distance {d_ml} worse than BP {d_bp}; "
            f"received = {recv}"
        )


def _test_low_p_low_fer() -> None:
    """At p = 0.001 ML on the tiny code should achieve FER very near 0."""
    from bb_constructor import make_tiny
    from channel import QSC
    code = make_tiny()
    dec = MLDecoder(p=0.001)
    qsc = QSC(0.001)
    rng = np.random.default_rng(7)
    zero = np.zeros(code.n, dtype=np.uint8)
    trials = 500
    failures = sum(
        not np.array_equal(dec(code, qsc.transmit(zero, rng)[0]), zero)
        for _ in range(trials)
    )
    # Expected error count per word: n*p = 18*0.001 = 0.018.
    # Pr[≥3 errors] ≈ (n choose 3) p^3 = 816 * 1e-9 ≈ 8e-7. So FER ≪ 1/500.
    assert failures <= 1, f"FER too high: {failures}/{trials} at p = 0.001"


def _test_reproducibility() -> None:
    """Same input ⇒ same output. ML is deterministic; rng is ignored."""
    from bb_constructor import make_tiny
    code = make_tiny()
    dec = MLDecoder(p=0.05)
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    recv = np.array([1, 0, 2, 3, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
                    dtype=np.uint8)
    out1 = dec(code, recv, rng_a)
    out2 = dec(code, recv, rng_b)
    assert np.array_equal(out1, out2)


def _test_invalid_params_rejected() -> None:
    for bad_p in (0.0, 0.75, 0.9, -0.1, 1.5):
        try:
            MLDecoder(p=bad_p)
        except ValueError:
            continue
        raise AssertionError(f"p = {bad_p} should have been rejected")
    try:
        MLDecoder(p=0.05, max_k=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("max_k = -1 should have been rejected")
    try:
        MLDecoder(p=0.05, chunk_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("chunk_size = 0 should have been rejected")


def _test_max_k_enforced() -> None:
    """A code with k > max_k must trigger a clear error rather than OOM."""
    from bb_constructor import make_tiny
    code = make_tiny()                     # k = 10
    dec = MLDecoder(p=0.05, max_k=5)       # too small
    try:
        dec(code, np.zeros(code.n, dtype=np.uint8))
    except ValueError as e:
        assert "max_k" in str(e), str(e)
    else:
        raise AssertionError("max_k enforcement did not trigger")


def _test_cache_reuses_table() -> None:
    """Calling the decoder twice on the same code must reuse the codeword table."""
    from bb_constructor import make_tiny
    code = make_tiny()
    dec = MLDecoder(p=0.05)
    dec(code, np.zeros(code.n, dtype=np.uint8))
    table_id = id(dec._codeword_cache[id(code.H)])
    dec(code, np.zeros(code.n, dtype=np.uint8))
    assert id(dec._codeword_cache[id(code.H)]) == table_id


def _run_all_tests() -> None:
    tests = [
        _test_enumerate_count_and_uniqueness,
        _test_enumerate_all_in_kernel,
        _test_enumerate_includes_zero_and_basis,
        _test_zero_received_returns_zero,
        _test_codeword_received_returns_itself,
        _test_every_single_error_corrected,
        _test_every_two_error_pattern_correctable_by_bdd,
        _test_output_is_always_a_codeword,
        _test_ml_at_least_as_good_as_bp,
        _test_low_p_low_fer,
        _test_reproducibility,
        _test_invalid_params_rejected,
        _test_max_k_enforced,
        _test_cache_reuses_table,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} ML decoder tests passed.\n")


if __name__ == "__main__":
    _run_all_tests()
