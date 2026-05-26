"""
decoder_bp.py — Sum-product Belief Propagation over GF(4) with FFT
check-node update.

Implements the BP-FFT decoder of Declercq & Fossorier, "Decoding algorithms
for nonbinary LDPC codes over GF(q)", IEEE Trans. Comm. 2007 (Section II),
specialized to q = 4. The decoder runs in the probability domain (no log /
EMS approximation) because for q = 4 the q² complexity of probability-domain
BP is already negligible and we want the cleanest possible reference
implementation for ML validation on Day 12.

Algorithm (one BP iteration)
----------------------------
Channel likelihood for each variable v with received symbol y_v:
    L_v(x) = (1 - p)   if x == y_v
           = p / 3     otherwise
(For the QSC defined in channel.py; passed in via the `p` parameter.)

Variable-to-check messages are length-4 probability vectors over F_4.

CHECK-NODE UPDATE (the only non-trivial step). For each check c with
incident edges (v_1, h_1), ..., (v_{d_c}, h_{d_c}):
  1.  Forward edge permutation. For each incoming μ_{v_k → c}, build
          m_k[y] = μ_{v_k → c}[h_k^{-1} · y]
      (multiplication of indices by h_k in the secondary-variable space,
       per Declercq-Fossorier eq. (4)).
  2.  Walsh-Hadamard transform.
          F_k = H4 · m_k
      where H4 is the 4 × 4 ±1 Walsh-Hadamard matrix on the additive group
      of F_4 ≅ (F_2)². This diagonalizes additive convolution on F_4
      (Declercq-Fossorier Proposition + eq. (8)).
  3.  Leave-one-out frequency-domain product.
          P_k = Π_{j ≠ k} F_j        (componentwise)
      computed via prefix/suffix sweeps in O(d_c · q).
  4.  Inverse Walsh-Hadamard. Since H4² = 4 · I,
          r_k = (H4 · P_k) / 4.
  5.  Backward edge permutation, with inverse of step 1:
          μ_{c → v_k}[x] = r_k[h_k · x].
      Clip negatives to 0 (numerical floor) and renormalise to sum = 1.

VARIABLE-NODE UPDATE. For each variable v with incident checks
{c_1, ..., c_{d_v}}, channel likelihood L_v, and incoming check-to-var
messages μ_{c_j → v}:
  • Posterior belief:
        b_v(x)  ∝  L_v(x) · Π_j μ_{c_j → v}(x).
  • Outgoing var-to-check message (leave-one-out):
        μ_{v → c_l}(x)  ∝  L_v(x) · Π_{j ≠ l} μ_{c_j → v}(x).
  Computed via prefix/suffix sweeps in O(d_v · q).
  Optional damping with parameter α:
        μ_new = (1 - α) · μ_proposed + α · μ_previous.

CONVERGENCE. After each check-node update we form the hard decision
x_hat[v] = argmax_x b_v(x) and stop early if the syndrome H · x_hat = 0
over F_4. Otherwise we keep iterating until max_iters.

The 4×4 Walsh-Hadamard matrix in the (b<<1)|a encoding (rows / cols ordered
0, 1, ω, ω²) is, from H4[v, w] = (-1)^{<v,w>_{F_2}} on the bit-pair indices,

        ⎡ +1  +1  +1  +1 ⎤
        ⎢ +1  -1  +1  -1 ⎥
   H4 = ⎢ +1  +1  -1  -1 ⎥
        ⎣ +1  -1  -1  +1 ⎦

which is symmetric (H4 = H4^T) and self-inverse up to scale: H4 · H4 = 4·I.

API contract: callable as `dec(code, received, rng=None) -> codeword`,
matching evaluation.Decoder. The `rng` argument is ignored (BP is
deterministic).

References
----------
[1] D. Declercq and M. Fossorier, "Decoding algorithms for nonbinary LDPC
    codes over GF(q)", IEEE Trans. Comm., vol. 55, no. 4, Apr. 2007.
[2] M. Davey and D. MacKay, "Low density parity check codes over GF(q)",
    IEEE Comm. Letters, vol. 2, no. 6, Jun. 1998.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bb_constructor import BBCode
from gf4_lib import _INV_TABLE, _MUL_TABLE, gf4_matvec


# ---------------------------------------------------------------------------
# Module-level constants: the 4×4 Walsh-Hadamard matrix and edge-permutation
# tables. Computing them once at import time and indexing into them is much
# cheaper than rebuilding per call.
# ---------------------------------------------------------------------------

# H4[v, w] = (-1)^{ (v&1)(w&1) XOR ((v>>1)&1)((w>>1)&1) }.
# The XOR appears because addition in F_2 is XOR; the outer sign is the
# character (-1)^{inner product}. Result is symmetric and orthogonal (up to 4).
def _build_H4() -> np.ndarray:
    idx = np.arange(4)
    a = idx & 1                       # low bit of each integer-encoded F_4 element
    b = (idx >> 1) & 1                # high bit
    # Outer dot products on the two bits, in F_2 (XOR), lifted to ±1.
    dots = np.bitwise_xor.outer(a, a) * 0  # placeholder for shape; will overwrite
    sign = np.zeros((4, 4), dtype=np.int8)
    for v in range(4):
        for w in range(4):
            ip = ((v & 1) & (w & 1)) ^ (((v >> 1) & 1) & ((w >> 1) & 1))
            sign[v, w] = -1 if ip else 1
    return sign.astype(np.float64)


H4: np.ndarray = _build_H4()
assert np.allclose(H4 @ H4, 4.0 * np.eye(4)), "H4 must satisfy H4·H4 = 4·I"


def _build_perm_tables() -> tuple[np.ndarray, np.ndarray]:
    """
    Build edge-permutation index arrays keyed by the F_4 edge weight h.

    PERM_FORWARD[h] is the index array such that new_msg = old_msg[PERM_FORWARD[h]]
    implements  new_msg[y] = old_msg[h^{-1} · y]  (var → check direction,
    DF eq. 4 "multiplication of tensor indices by h_k").

    PERM_BACKWARD[h] is the index array such that new_msg = old_msg[PERM_BACKWARD[h]]
    implements  new_msg[x] = old_msg[h · x]  (check → var direction,
    the inverse permutation, "division of indices by h_k").

    Row h = 0 is unused on real edges (parity-check entries are nonzero) and
    is left as the identity so that any accidental indexing stays in-bounds.
    """
    forward = np.tile(np.arange(4, dtype=np.uint8), (4, 1))
    backward = np.tile(np.arange(4, dtype=np.uint8), (4, 1))
    for h in (1, 2, 3):
        h_inv = int(_INV_TABLE[h])
        for y in range(4):
            forward[h, y] = _MUL_TABLE[h_inv, y]
            backward[h, y] = _MUL_TABLE[h, y]
    return forward, backward


PERM_FORWARD, PERM_BACKWARD = _build_perm_tables()


# ---------------------------------------------------------------------------
# The decoder.
# ---------------------------------------------------------------------------
@dataclass
class BPFFTDecoder:
    """
    Sum-product Belief Propagation over GF(4) with FFT check-node update,
    in the probability domain. Conforms to evaluation.Decoder when called.

    Parameters
    ----------
    p : float
        Channel error rate assumed by the decoder. Builds the per-symbol
        likelihood vectors. Must lie in (0, 1).
    max_iters : int, default 50
        Maximum BP iteration count before declaring no convergence.
    damping : float, default 0.0
        Damping factor in [0, 1) applied to var→check messages:
            μ_new = (1 - damping) · μ_proposed + damping · μ_previous.
        damping = 0 disables damping (standard flooding BP).
    return_belief_argmax : bool, default True
        If True (default), even on non-convergence return the hard decision
        from the final-iteration posterior beliefs. If False, raise.
    name : str, default ""
        Identifier carried into EvalResult.decoder_name. Auto-generated
        from parameters if blank.
    """

    p: float
    max_iters: int = 50
    damping: float = 0.0
    return_belief_argmax: bool = True
    name: str = ""

    # Cached Tanner-graph metadata, keyed by id(code.H) so we don't rebuild
    # when the harness loops over many received words for the same code.
    _graph_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError(f"p must be in (0, 1); got {self.p}")
        if not (0.0 <= self.damping < 1.0):
            raise ValueError(f"damping must be in [0, 1); got {self.damping}")
        if self.max_iters <= 0:
            raise ValueError(f"max_iters must be positive; got {self.max_iters}")
        if not self.name:
            self.name = f"bp-fft_p{self.p:.3f}_it{self.max_iters}_d{self.damping:.2f}"

    # ---- one-off graph metadata extraction ------------------------------
    def _get_graph(self, code: BBCode):
        key = id(code.H)
        cached = self._graph_cache.get(key)
        if cached is not None:
            return cached
        H = code.H
        M, N = H.shape
        # Flat edge list, in row-major (check-major) order. This guarantees
        # that np.where(edge_check == c) returns edges in a stable order.
        rows, cols = np.nonzero(H)
        weights = H[rows, cols].astype(np.uint8)
        E = int(len(rows))
        edge_check = rows.astype(np.int64)
        edge_var = cols.astype(np.int64)

        # Per-check and per-variable lists of edge indices. Using numpy arrays
        # keeps fancy-indexing fast inside the BP loop.
        checks_edges = [np.where(edge_check == c)[0] for c in range(M)]
        vars_edges = [np.where(edge_var == v)[0] for v in range(N)]

        # Pre-extract per-edge permutation index arrays. Each is shape (E, 4).
        perm_fwd = PERM_FORWARD[weights].astype(np.int64)
        perm_bwd = PERM_BACKWARD[weights].astype(np.int64)

        cached = {
            "M": M, "N": N, "E": E,
            "edge_check": edge_check, "edge_var": edge_var,
            "edge_weight": weights,
            "checks_edges": checks_edges, "vars_edges": vars_edges,
            "perm_fwd": perm_fwd, "perm_bwd": perm_bwd,
        }
        self._graph_cache[key] = cached
        return cached

    # ---- public callable ------------------------------------------------
    def __call__(
        self,
        code: BBCode,
        received: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        return self._decode(code, np.asarray(received, dtype=np.uint8))

    # ---- core decode loop -----------------------------------------------
    def _decode(self, code: BBCode, received: np.ndarray) -> np.ndarray:
        g = self._get_graph(code)
        N, E = g["N"], g["E"]
        p = self.p

        # Channel likelihood per variable.
        # L[v, x] = (1-p) if x == received[v] else p/3.
        L = np.full((N, 4), p / 3.0, dtype=np.float64)
        L[np.arange(N), received.astype(np.int64)] = 1.0 - p

        # Initial var→check messages: copy of the channel likelihood at each
        # incident variable. Same vector replicated across the variable's
        # incident edges, since the first iteration has no extrinsic info.
        mu_vc = L[g["edge_var"]].copy()      # shape (E, 4)
        mu_cv = np.zeros((E, 4), dtype=np.float64)

        x_hat = np.zeros(N, dtype=np.uint8)
        H_mat = code.H

        for iteration in range(self.max_iters):
            # ============ CHECK-NODE UPDATE ============
            self._check_node_update(mu_vc, mu_cv, g)

            # ============ DECISION + SYNDROME EARLY-STOP ============
            beliefs = self._compute_beliefs(L, mu_cv, g)
            x_hat = np.argmax(beliefs, axis=1).astype(np.uint8)
            syndrome = gf4_matvec(H_mat, x_hat)
            if not np.any(syndrome):
                return x_hat

            # ============ VARIABLE-NODE UPDATE ============
            self._var_node_update(L, mu_cv, mu_vc, g)

        if self.return_belief_argmax:
            return x_hat
        raise RuntimeError(
            f"BP did not converge in {self.max_iters} iterations "
            f"(syndrome weight = {int(np.count_nonzero(syndrome))})"
        )

    # ---- check-node FFT update ------------------------------------------
    @staticmethod
    def _check_node_update(
        mu_vc: np.ndarray, mu_cv: np.ndarray, g: dict
    ) -> None:
        """
        Overwrite `mu_cv` with the FFT-based check-node update derived from
        `mu_vc`. Steps follow DF Section II-C verbatim.
        """
        # ---- Step 1: forward edge permutation on every incoming msg ----
        # permuted[e, y] = mu_vc[e, h_e^{-1} · y].
        # take_along_axis lets each row of mu_vc be re-indexed by its own perm.
        permuted = np.take_along_axis(mu_vc, g["perm_fwd"], axis=1)

        # ---- Step 2: Walsh-Hadamard transform of each permuted message ----
        # Right-multiply by H4 (which equals H4^T): freq[e] = H4 @ permuted[e].
        # Numpy broadcasting form: (E, 4) @ (4, 4)^T = (E, 4).
        freq = permuted @ H4.T

        # ---- Step 3: per-check, leave-one-out frequency-domain product ----
        out_freq = np.empty_like(freq)
        for c, edges in enumerate(g["checks_edges"]):
            dc = edges.shape[0]
            if dc == 0:
                continue
            F_c = freq[edges]                        # (dc, 4)
            # Two passes for O(dc) leave-one-out. left[k] = ∏_{j<k} F_c[j].
            left = np.empty_like(F_c)
            right = np.empty_like(F_c)
            left[0] = 1.0
            for k in range(1, dc):
                left[k] = left[k - 1] * F_c[k - 1]
            right[dc - 1] = 1.0
            for k in range(dc - 2, -1, -1):
                right[k] = right[k + 1] * F_c[k + 1]
            out_freq[edges] = left * right

        # ---- Step 4: inverse Walsh-Hadamard (factor 1/4) ----
        # Since H4 · H4 = 4·I, the inverse transform is H4 / 4.
        out_prob = (out_freq @ H4.T) * 0.25

        # ---- Step 5: backward edge permutation ----
        # mu_cv[e, x] = out_prob[e, h_e · x].
        new_mu_cv = np.take_along_axis(out_prob, g["perm_bwd"], axis=1)

        # ---- Step 6: clip negative floating-point residue, renormalise ----
        # The convolution of probability vectors is itself a probability
        # vector, but small negatives can appear from finite-precision IFHT.
        # We floor at 0 and rescale to sum = 1 per edge.
        np.maximum(new_mu_cv, 0.0, out=new_mu_cv)
        sums = new_mu_cv.sum(axis=1, keepdims=True)
        # Where the row degenerates to zero (numerical pathology), fall back
        # to a uniform message rather than dividing by zero.
        zero_rows = (sums.ravel() <= 0.0)
        sums[zero_rows] = 1.0
        new_mu_cv[zero_rows] = 0.25
        new_mu_cv /= sums

        np.copyto(mu_cv, new_mu_cv)

    # ---- variable-node update -------------------------------------------
    def _var_node_update(
        self,
        L: np.ndarray, mu_cv: np.ndarray, mu_vc: np.ndarray, g: dict,
    ) -> None:
        """
        Overwrite `mu_vc` with the pointwise-product variable-node update
        applied to `mu_cv` and the channel likelihood `L`, with optional
        damping toward the previous `mu_vc`.
        """
        new_mu_vc = np.empty_like(mu_vc)
        damping = self.damping

        for v, edges in enumerate(g["vars_edges"]):
            dv = edges.shape[0]
            if dv == 0:
                continue
            Mv = mu_cv[edges]                        # (dv, 4)
            # Leave-one-out via prefix/suffix products of the incoming msgs.
            left = np.empty_like(Mv)
            right = np.empty_like(Mv)
            left[0] = 1.0
            for k in range(1, dv):
                left[k] = left[k - 1] * Mv[k - 1]
            right[dv - 1] = 1.0
            for k in range(dv - 2, -1, -1):
                right[k] = right[k + 1] * Mv[k + 1]
            outgoing = L[v] * left * right           # (dv, 4)
            # Normalise each row independently.
            sums = outgoing.sum(axis=1, keepdims=True)
            zero_rows = (sums.ravel() <= 0.0)
            sums[zero_rows] = 1.0
            outgoing[zero_rows] = 0.25
            outgoing /= sums
            new_mu_vc[edges] = outgoing

        if damping > 0.0:
            new_mu_vc = (1.0 - damping) * new_mu_vc + damping * mu_vc
            # Both operands are already normalised pdfs, so a convex
            # combination is also normalised; no further rescale needed.

        np.copyto(mu_vc, new_mu_vc)

    # ---- posterior beliefs ----------------------------------------------
    @staticmethod
    def _compute_beliefs(L: np.ndarray, mu_cv: np.ndarray, g: dict) -> np.ndarray:
        """
        Posterior beliefs b[v, x] ∝ L[v, x] · ∏_{c ∈ N(v)} mu_cv[(v,c), x].
        Normalised per variable.
        """
        N = g["N"]
        beliefs = L.copy()
        for v, edges in enumerate(g["vars_edges"]):
            for e in edges:
                beliefs[v] *= mu_cv[e]
        sums = beliefs.sum(axis=1, keepdims=True)
        sums[sums.ravel() <= 0.0] = 1.0
        beliefs /= sums
        return beliefs


# ===========================================================================
# Unit tests — exercised when running this module directly. Designed to
# isolate the FFT machinery from the BP loop so a failure points at one
# specific subsystem.
# ===========================================================================
def _direct_conv_f4(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reference additive convolution over F_4: (u*v)[c] = sum_{a+b=c} u[a]v[b]."""
    out = np.zeros(4, dtype=np.float64)
    for a in range(4):
        for b in range(4):
            c = a ^ b           # F_4 addition == bitwise XOR on (b<<1)|a
            out[c] += u[a] * v[b]
    return out


def _test_H4_self_inverse() -> None:
    """The 4×4 Walsh-Hadamard satisfies H4 · H4 = 4·I."""
    assert np.allclose(H4 @ H4, 4.0 * np.eye(4))
    # Symmetric matrix.
    assert np.array_equal(H4, H4.T)


def _test_FHT_convolution_theorem() -> None:
    """For random pdfs u, v over F_4, IFHT(FHT(u) ⊙ FHT(v)) == direct conv."""
    rng = np.random.default_rng(20260601)
    for trial in range(10):
        u = rng.random(4); u /= u.sum()
        v = rng.random(4); v /= v.sum()
        direct = _direct_conv_f4(u, v)
        via_fft = (H4 @ ((H4 @ u) * (H4 @ v))) / 4.0
        assert np.allclose(direct, via_fft), (
            f"FHT convolution disagreement on trial {trial}:\n"
            f"  direct = {direct}\n  via_fft = {via_fft}"
        )


def _test_perm_tables_are_inverses() -> None:
    """For every nonzero h, PERM_FORWARD[h] and PERM_BACKWARD[h] are inverses."""
    for h in (1, 2, 3):
        fwd = PERM_FORWARD[h]
        bwd = PERM_BACKWARD[h]
        # Composing them either way gives the identity.
        assert np.array_equal(fwd[bwd], np.arange(4))
        assert np.array_equal(bwd[fwd], np.arange(4))


def _test_perm_zero_fixed() -> None:
    """The zero element of F_4 is fixed under multiplication by any h."""
    for h in (1, 2, 3):
        assert PERM_FORWARD[h, 0] == 0
        assert PERM_BACKWARD[h, 0] == 0


def _test_perm_cycles_on_nonzero() -> None:
    """
    The forward-permutation action on the 3 nonzero F_4 elements matches
    multiplication by h^{-1}. Spot-check h = ω (= 2): h^{-1} = ω² (= 3), so
    new_msg[1] = old_msg[3], new_msg[2] = old_msg[3 * 2] = old_msg[1], etc.
    """
    fwd = PERM_FORWARD[2]   # h = ω
    # h^{-1} · 1 = ω² · 1 = 3; h^{-1} · ω = ω² · ω = ω³ = 1;
    # h^{-1} · ω² = ω² · ω² = ω⁴ = ω = 2.
    assert fwd[1] == 3 and fwd[2] == 1 and fwd[3] == 2


def _test_zero_input_returns_zero() -> None:
    """y = 0 (zero codeword + zero noise) must round-trip to the zero codeword."""
    from bb_constructor import make_tiny
    code = make_tiny()
    dec = BPFFTDecoder(p=0.05, max_iters=20)
    out = dec(code, np.zeros(code.n, dtype=np.uint8))
    assert np.array_equal(out, np.zeros(code.n, dtype=np.uint8)), out


def _test_codeword_input_preserved() -> None:
    """No noise on a non-trivial codeword: decoder must return it unchanged."""
    from bb_constructor import make_tiny
    code = make_tiny()
    basis = code.codeword_basis()
    dec = BPFFTDecoder(p=0.05, max_iters=20)
    for cw in basis[:3]:
        out = dec(code, cw)
        assert np.array_equal(out, cw), (
            f"codeword not preserved: input={cw}, output={out}"
        )


def _test_single_symbol_error_corrected() -> None:
    """
    With one error at low p, BP on the tiny code (d ≤ 6) should recover the
    zero codeword for every single-symbol error pattern.
    """
    from bb_constructor import make_tiny
    code = make_tiny()
    dec = BPFFTDecoder(p=0.02, max_iters=50)
    n = code.n
    zero = np.zeros(n, dtype=np.uint8)
    failures = 0
    for pos in range(n):
        for sym in (1, 2, 3):
            received = zero.copy()
            received[pos] = sym
            out = dec(code, received)
            if not np.array_equal(out, zero):
                failures += 1
    # We allow zero failures on the tiny code at d=6: a single symbol error
    # is well within correction radius and BP should converge fast.
    assert failures == 0, f"BP failed on {failures}/{3*n} single-error patterns"


def _test_low_noise_recovery_rate() -> None:
    """At p = 0.01 BP should recover ≥ 95% of frames on the tiny code."""
    from bb_constructor import make_tiny
    from channel import QSC
    code = make_tiny()
    dec = BPFFTDecoder(p=0.01, max_iters=50)
    qsc = QSC(0.01)
    rng = np.random.default_rng(202606)
    zero = np.zeros(code.n, dtype=np.uint8)
    trials = 300
    successes = sum(
        np.array_equal(dec(code, qsc.sample_noise(code.n, rng)), zero)
        for _ in range(trials)
    )
    rate = successes / trials
    assert rate >= 0.95, f"recovery rate {rate:.2f} too low at p=0.01"


def _test_reproducible() -> None:
    """Same received word ⇒ same output. BP is deterministic; rng is ignored."""
    from bb_constructor import make_tiny
    from channel import QSC
    code = make_tiny()
    dec = BPFFTDecoder(p=0.05, max_iters=30)
    qsc = QSC(0.05)
    noise = qsc.sample_noise(code.n, np.random.default_rng(7))
    out1 = dec(code, noise)
    out2 = dec(code, noise, rng=np.random.default_rng(123))
    assert np.array_equal(out1, out2)


def _test_invalid_params_rejected() -> None:
    for bad_p in (0.0, 1.0, -0.1, 1.5):
        try:
            BPFFTDecoder(p=bad_p)
        except ValueError:
            continue
        raise AssertionError(f"p={bad_p} should have been rejected")
    for bad_d in (-0.1, 1.0, 1.5):
        try:
            BPFFTDecoder(p=0.05, damping=bad_d)
        except ValueError:
            continue
        raise AssertionError(f"damping={bad_d} should have been rejected")
    try:
        BPFFTDecoder(p=0.05, max_iters=0)
    except ValueError:
        pass
    else:
        raise AssertionError("max_iters=0 should have been rejected")


def _test_check_node_matches_direct_convolution() -> None:
    """
    Build a single check node of degree 2 with weights (1, ω) on the edges,
    feed two known input messages, and verify that the FFT-BP output equals
    the textbook direct-convolution result.

    Direct rule (DF eq. (5)) for degree-2 check  h_1·x_1 + h_2·x_2 = 0:
        μ_out_to_edge_2 (x_2) = sum_{x_1 : h_1 x_1 + h_2 x_2 = 0} mu_in_1(x_1)
                              = mu_in_1(h_1^{-1} · h_2 · x_2)
    i.e. a permutation of the other incoming message — easy to hand-verify.
    """
    rng = np.random.default_rng(11)
    m1 = rng.random(4); m1 /= m1.sum()
    m2 = rng.random(4); m2 /= m2.sum()

    h1, h2 = 1, 2            # edges carry weights 1, ω
    # Outgoing on edge 2 = permute m1 by h1^{-1} · h2 = 1 · ω = ω: out2[x] = m1[ω·x].
    expected_out2 = m1[PERM_BACKWARD[2]]       # backward perm corresponds to ·h
    # We can't apply PERM_FORWARD/BACKWARD directly here because they encode
    # the "var → check, then check → var" round trip. Easiest: assemble a
    # tiny 1-check 2-variable graph and run _check_node_update on it.
    edges_perm_fwd = PERM_FORWARD[np.array([h1, h2])].astype(np.int64)
    edges_perm_bwd = PERM_BACKWARD[np.array([h1, h2])].astype(np.int64)
    g = {
        "M": 1, "N": 2, "E": 2,
        "edge_check": np.array([0, 0]),
        "edge_var": np.array([0, 1]),
        "edge_weight": np.array([h1, h2], dtype=np.uint8),
        "checks_edges": [np.array([0, 1])],
        "vars_edges": [np.array([0]), np.array([1])],
        "perm_fwd": edges_perm_fwd,
        "perm_bwd": edges_perm_bwd,
    }
    mu_vc = np.stack([m1, m2])
    mu_cv = np.zeros_like(mu_vc)
    BPFFTDecoder._check_node_update(mu_vc, mu_cv, g)

    # Expected: on edge 1, outgoing depends only on m2 via the degree-2 rule;
    # likewise on edge 2 it depends only on m1. Verify edge 2.
    # Computed from direct argument: out_2[x_2] ∝ m1(h1^{-1} h2 x_2).
    raw_out2 = m1[(_MUL_TABLE[_INV_TABLE[h1], _MUL_TABLE[h2, np.arange(4)]])]
    raw_out2 = raw_out2 / raw_out2.sum()
    assert np.allclose(mu_cv[1], raw_out2, atol=1e-12), (mu_cv[1], raw_out2)


def _run_all_tests() -> None:
    tests = [
        _test_H4_self_inverse,
        _test_FHT_convolution_theorem,
        _test_perm_tables_are_inverses,
        _test_perm_zero_fixed,
        _test_perm_cycles_on_nonzero,
        _test_check_node_matches_direct_convolution,
        _test_zero_input_returns_zero,
        _test_codeword_input_preserved,
        _test_single_symbol_error_corrected,
        _test_low_noise_recovery_rate,
        _test_reproducible,
        _test_invalid_params_rejected,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} BP-FFT tests passed.\n")


if __name__ == "__main__":
    _run_all_tests()
