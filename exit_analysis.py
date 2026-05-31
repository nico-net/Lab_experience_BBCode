"""
exit_analysis.py — EXIT chart analysis for (d_v, d_c)-regular GF(4) LDPC codes.

PIPELINE Day 15 deliverable.

Implements the Monte-Carlo density-evolution approach to nonbinary EXIT
charts (Ashikhmin-Kramer-ten Brink generalised to F_q by Bennatan-Burshtein
2006; see also Davey-MacKay 1998 Sec. III for the discrete F_q-LDPC density
evolution). For q = 4 the messages are length-4 probability vectors over
F_4, and the relevant scalar summary of "how informative" a message is
about its underlying symbol is the mutual information I(X; M) ∈ [0, log_2 q]
= [0, 2] bits/symbol.

Algorithm
---------
1.  CHANNEL MI.  For the QSC at rate p:
        I_ch(p) = log_2(q) + (1-p) log_2(1-p) + p log_2(p/(q-1)),
    monotonically decreasing on (0, (q-1)/q). Numerically invertible.

2.  CALIBRATION ENSEMBLE.  Generate N independent messages with mutual
    information equal to a target I_A by drawing them from the QSC with
    p_calib = invert_channel_mi(I_A). By the channel-symmetry of QSC, the
    resulting ensemble of length-q vectors is exchangeable under any
    permutation of the nonzero F_q indices — i.e. effectively averaged
    over iid uniform F_q* edge weights.

3.  VARIABLE-NODE EXIT.  For a degree-d_v variable node, the extrinsic
    output on one edge is the normalised pointwise product of the d_v - 1
    incoming check-to-var messages (each with MI = I_A) and the channel
    message (MI = I_ch(p)). Monte-Carlo over N samples:
        I_E^V(I_A, p) = E[ log_2 q − H(M_out) ].

4.  CHECK-NODE EXIT.  For a degree-d_c check node, the extrinsic output is
    the additive-convolution over F_q of the d_c - 1 incoming messages,
    computed via the Walsh-Hadamard transform (the same machinery as the
    FFT-BP decoder, with edge weight h = 1 because the iid-edge-weight
    averaging happens automatically through the symmetric QSC calibration).

5.  THRESHOLD.  The BP threshold p* is the smallest p such that the V
    curve I_E^V(·, p) and the inverted C curve I_E^C(·)^{-1} touch (i.e.
    the BP iteration tunnel pinches off). Threshold extraction is the
    Day 16 deliverable; this module produces the curves needed for it.

Validity of the iid-uniform-edge-weight assumption for BB codes
---------------------------------------------------------------
Each row of the BB parity-check matrix H = [A | B] contains *exactly* the
three F_4 elements {1, ω, ω²} (one per nonzero monomial in A and B). So the
per-row weight distribution is uniformly supported on F_4*, matching the
iid-uniform assumption (modulo per-row correlations between the three
weights, which are a second-order effect for asymptotic threshold).

References
----------
- A. Ashikhmin, G. Kramer, S. ten Brink, "Extrinsic information transfer
  functions: model and erasure channel properties", IEEE Trans. IT 2004.
- A. Bennatan, D. Burshtein, "Design and analysis of nonbinary LDPC codes
  for arbitrary discrete-memoryless channels", IEEE Trans. IT 2006.
- D. Declercq, M. Fossorier, "Decoding algorithms for nonbinary LDPC codes
  over GF(q)", IEEE Trans. Comm. 2007 (FFT-BP check update used here).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

# Reuse the same Walsh-Hadamard matrix the BP decoder uses, to keep the
# EXIT analysis consistent with the actual check-node update.
from decoder_bp import H4

Q = 4
LOG2_Q = float(np.log2(Q))


# ---------------------------------------------------------------------------
# Mutual information estimator and channel MI
# ---------------------------------------------------------------------------
def mi_of_messages(msgs: np.ndarray, q: int = Q) -> float:
    """
    Estimate I(X; M) in bits/symbol from an array of N message samples
    (shape (N, q)) under the all-zero codeword convention. With X = 0
    fixed and the message ensemble exchangeable under F_q* permutations,
    the Bayes posterior P(X | M) recovers all bits not lost to the
    intrinsic entropy of M, so I(X; M) = log_2(q) - E[H(M)]. H here is
    Shannon entropy in bits.
    """
    msgs = np.clip(msgs, 1e-300, 1.0)
    H = -np.sum(msgs * np.log2(msgs), axis=1)
    return float(np.log2(q) - H.mean())


def channel_mi(p: float, q: int = Q) -> float:
    """
    Channel mutual information of the q-ary symmetric channel with error
    rate p, in bits/symbol. Closed-form:
        I_ch(p) = log_2(q) - H(Y|X)
                = log_2(q) + (1-p) log_2(1-p) + p log_2(p/(q-1)).
    Monotonically decreasing from log_2(q) at p = 0 to 0 at p = (q-1)/q.
    Beyond p = (q-1)/q the channel still has zero mutual information
    (uniform output regardless of input) but the formula would diverge,
    so we clamp.
    """
    if p <= 0.0:
        return float(np.log2(q))
    if p >= (q - 1) / q:
        return 0.0
    return float(
        np.log2(q)
        + (1.0 - p) * np.log2(1.0 - p)
        + p * np.log2(p / (q - 1))
    )


def invert_channel_mi(target_mi: float, q: int = Q) -> float:
    """Find p ∈ (0, (q-1)/q) such that channel_mi(p) = target_mi."""
    if target_mi >= np.log2(q):
        return 0.0
    if target_mi <= 0.0:
        return (q - 1) / q
    f = lambda p: channel_mi(p, q) - target_mi
    return float(brentq(f, 1e-12, (q - 1) / q - 1e-12))


# ---------------------------------------------------------------------------
# Calibration ensemble
# ---------------------------------------------------------------------------
def calibration_messages(
    target_mi: float, N: int, q: int = Q,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw N independent messages with MI = target_mi from the QSC with
    p_calib = invert_channel_mi(target_mi). Under the all-zero-codeword
    convention (X = 0), the received symbol Y is drawn with
        P(Y = 0) = 1 - p_calib,    P(Y = y) = p_calib / (q-1)  for y ≠ 0,
    and the message is the channel likelihood
        m[x] ∝ (1 - p_calib) if x == Y else p_calib / (q - 1).
    Returns a (N, q) float64 array of normalised probability vectors.
    """
    if rng is None:
        rng = np.random.default_rng()
    p_calib = invert_channel_mi(target_mi, q)
    if p_calib > 0.0:
        cat = np.empty(q, dtype=np.float64)
        cat[0] = 1.0 - p_calib
        cat[1:] = p_calib / (q - 1)
        Y = rng.choice(q, size=N, p=cat).astype(np.uint8)
    else:
        Y = np.zeros(N, dtype=np.uint8)
    msgs = np.full(
        (N, q),
        p_calib / (q - 1) if p_calib > 0 else 0.0,
        dtype=np.float64,
    )
    msgs[np.arange(N), Y] = (1.0 - p_calib) if p_calib > 0 else 1.0
    msgs /= msgs.sum(axis=1, keepdims=True)
    return msgs


# ---------------------------------------------------------------------------
# Variable-node EXIT function
# ---------------------------------------------------------------------------
def variable_node_mi(
    I_A: float, p: float, d_v: int,
    N: int = 10000, q: int = Q,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Estimate the mutual information of the extrinsic var→check message
    from a degree-d_v variable node: the normalised pointwise product of
    (d_v - 1) iid incoming check→var messages with MI = I_A and one
    channel message with MI = I_ch(p).

    Endpoints:
        I_E^V(I_A = 0, p) = I_ch(p)              (no info from checks)
        I_E^V(I_A = log_2 q, p) = log_2 q        (perfect info from checks)
    """
    if rng is None:
        rng = np.random.default_rng()
    incoming = calibration_messages(I_A, (d_v - 1) * N, q, rng).reshape(
        N, d_v - 1, q
    )
    channel = calibration_messages(channel_mi(p, q), N, q, rng)
    prod = channel.copy()
    for j in range(d_v - 1):
        prod *= incoming[:, j, :]
    prod = np.clip(prod, 1e-300, np.inf)
    prod /= prod.sum(axis=1, keepdims=True)
    return mi_of_messages(prod, q)


# ---------------------------------------------------------------------------
# Check-node EXIT function (via Walsh-Hadamard transform)
# ---------------------------------------------------------------------------
def check_node_mi(
    I_A: float, d_c: int,
    N: int = 10000, q: int = Q,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Estimate the mutual information of the extrinsic check→var message
    from a degree-d_c check node: the F_q-additive convolution of the
    (d_c - 1) iid incoming var→check messages with MI = I_A. Computed
    via the Walsh-Hadamard transform on the additive group F_2^p ≅ F_q:
        F = WHT(m),    P = product across (d_c - 1) of F,    m_out = WHT^{-1}(P).

    Edge weights are taken = 1 in this routine; the iid-uniform-over-F_q*
    edge-weight assumption is satisfied implicitly because the QSC-based
    calibration ensemble is symmetric over the nonzero F_q indices.

    Endpoints:
        I_E^C(I_A = 0) = 0          (input = uniform → output = uniform)
        I_E^C(I_A = log_2 q) = log_2 q   (input = delta at 0 → output = delta at 0)
    """
    if rng is None:
        rng = np.random.default_rng()
    if q != Q:
        raise NotImplementedError("check_node_mi is F_4 specific; H4 is 4x4")
    incoming = calibration_messages(I_A, (d_c - 1) * N, q, rng).reshape(
        N, d_c - 1, q
    )
    # Forward Walsh-Hadamard: each row gets multiplied by H4 (which is symmetric).
    F = incoming @ H4.T                               # (N, d_c - 1, q)
    P = np.prod(F, axis=1)                            # (N, q)
    # Inverse Walsh-Hadamard: divide by q because H4 · H4 = q · I.
    out = (P @ H4.T) / q                              # (N, q)
    # The convolution of probability vectors is a probability vector in
    # exact arithmetic, but finite precision can yield small negative
    # entries; clip and renormalise per the BP-FFT decoder convention.
    np.maximum(out, 0.0, out=out)
    sums = out.sum(axis=1, keepdims=True)
    zero_rows = sums.ravel() <= 0.0
    sums[zero_rows] = 1.0
    out[zero_rows] = 1.0 / q
    out /= sums
    return mi_of_messages(out, q)


# ---------------------------------------------------------------------------
# Full EXIT curves at a given p
# ---------------------------------------------------------------------------
@dataclass
class ExitCurves:
    """Bundle of arrays describing the V and C EXIT curves at one channel rate."""
    p: float
    d_v: int
    d_c: int
    I_A: np.ndarray            # shape (n_points,)
    I_E_V: np.ndarray          # I_E^V(I_A, p) at each I_A
    I_E_C: np.ndarray          # I_E^C(I_A) at each I_A
    N: int                     # Monte-Carlo budget per point


def exit_curves(
    p: float, d_v: int, d_c: int,
    n_points: int = 21, N: int = 10000, q: int = Q,
    rng: Optional[np.random.Generator] = None,
) -> ExitCurves:
    """Compute the V- and C-EXIT curves on a uniform grid of I_A ∈ [0, log_2 q]."""
    if rng is None:
        rng = np.random.default_rng()
    I_A = np.linspace(0.0, np.log2(q), n_points)
    I_E_V = np.array([
        variable_node_mi(ia, p, d_v, N=N, q=q, rng=rng) for ia in I_A
    ])
    I_E_C = np.array([
        check_node_mi(ia, d_c, N=N, q=q, rng=rng) for ia in I_A
    ])
    return ExitCurves(p=p, d_v=d_v, d_c=d_c, I_A=I_A,
                      I_E_V=I_E_V, I_E_C=I_E_C, N=N)


# ===========================================================================
# Unit tests
# ===========================================================================
def _test_mi_extremes() -> None:
    """Delta msgs have MI = log_2 q; uniform msgs have MI = 0."""
    delta = np.zeros((100, 4)); delta[:, 0] = 1.0
    assert abs(mi_of_messages(delta) - 2.0) < 1e-6
    uniform = np.full((100, 4), 0.25)
    assert abs(mi_of_messages(uniform) - 0.0) < 1e-9


def _test_channel_mi_boundaries() -> None:
    """I_ch(0) = log_2 q, I_ch((q-1)/q) = 0, monotone decreasing."""
    assert abs(channel_mi(0.0) - 2.0) < 1e-9
    assert abs(channel_mi(0.75) - 0.0) < 1e-9
    prev = 2.0
    for p in [0.01, 0.05, 0.1, 0.2, 0.4, 0.6]:
        cur = channel_mi(p)
        assert cur < prev, (p, cur, prev)
        prev = cur


def _test_channel_mi_invert_roundtrip() -> None:
    """invert_channel_mi · channel_mi = identity to ~1e-10."""
    for p in [0.005, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6]:
        mi = channel_mi(p)
        p_back = invert_channel_mi(mi)
        assert abs(p - p_back) < 1e-10, (p, p_back)


def _test_calibration_has_target_mi() -> None:
    """Calibrated message ensemble has the requested MI within MC error."""
    rng = np.random.default_rng(42)
    for target in [0.2, 0.5, 1.0, 1.5, 1.9]:
        msgs = calibration_messages(target, 200_000, rng=rng)
        actual = mi_of_messages(msgs)
        # Monte-Carlo error scales as ~1/sqrt(N); 200k → ~1e-3 precision.
        assert abs(actual - target) < 0.01, (target, actual)


def _test_V_exit_endpoint_zero() -> None:
    """At I_A = 0 (uniform incoming) the variable-node output equals the channel."""
    rng = np.random.default_rng(43)
    for p in [0.05, 0.10, 0.20]:
        expected = channel_mi(p)
        got = variable_node_mi(0.0, p, d_v=3, N=80_000, rng=rng)
        assert abs(got - expected) < 0.02, (p, expected, got)


def _test_V_exit_endpoint_max() -> None:
    """At I_A = log_2 q (perfect incoming), V output is perfect."""
    rng = np.random.default_rng(44)
    got = variable_node_mi(2.0, 0.1, d_v=3, N=10_000, rng=rng)
    assert abs(got - 2.0) < 1e-6, got


def _test_C_exit_endpoints() -> None:
    """C-EXIT endpoints: f(0) = 0, f(log_2 q) = log_2 q."""
    rng = np.random.default_rng(45)
    got0 = check_node_mi(0.0, d_c=6, N=50_000, rng=rng)
    assert got0 < 0.005, got0
    got_max = check_node_mi(2.0, d_c=6, N=10_000, rng=rng)
    assert abs(got_max - 2.0) < 1e-6, got_max


def _test_V_monotonic() -> None:
    """V-EXIT is monotone non-decreasing in I_A."""
    rng = np.random.default_rng(46)
    I_A = np.linspace(0, 2, 9)
    vals = [variable_node_mi(ia, 0.1, 3, N=40_000, rng=rng) for ia in I_A]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 0.005, (vals, a, b)


def _test_C_monotonic() -> None:
    """C-EXIT is monotone non-decreasing in I_A."""
    rng = np.random.default_rng(47)
    I_A = np.linspace(0, 2, 9)
    vals = [check_node_mi(ia, 6, N=40_000, rng=rng) for ia in I_A]
    for a, b in zip(vals, vals[1:]):
        assert b >= a - 0.005, (vals, a, b)


def _test_V_decreasing_in_p() -> None:
    """For fixed I_A, V-EXIT decreases as the channel gets noisier (higher p)."""
    rng = np.random.default_rng(48)
    I_A = 0.5
    vals = [variable_node_mi(I_A, p, 3, N=40_000, rng=rng)
            for p in [0.01, 0.05, 0.1, 0.2]]
    for a, b in zip(vals, vals[1:]):
        assert b <= a + 0.005, vals


def _run_all_tests() -> None:
    tests = [
        _test_mi_extremes,
        _test_channel_mi_boundaries,
        _test_channel_mi_invert_roundtrip,
        _test_calibration_has_target_mi,
        _test_V_exit_endpoint_zero,
        _test_V_exit_endpoint_max,
        _test_C_exit_endpoints,
        _test_V_monotonic,
        _test_C_monotonic,
        _test_V_decreasing_in_p,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} EXIT analysis tests passed.\n")


if __name__ == "__main__":
    _run_all_tests()