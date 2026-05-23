"""
channel.py — Quaternary Symmetric Channel (QSC) over F_4.

Model
-----
For symbol-error rate p ∈ [0, 1], the QSC acts on each symbol independently:

    Pr[y = x]        = 1 − p
    Pr[y = x + e_k]  = p / 3   for each nonzero e_k ∈ F_4* = {1, ω, ω²}

In additive form  y = x + e  where the noise e is drawn from
    Pr[e = 0]      = 1 − p
    Pr[e = nonzero ∈ F_4*] = p / 3 each.

The channel is symmetric over F_4 (translation-invariant under addition),
which justifies the standard all-zero codeword Monte-Carlo convention used
in `evaluation.py`. Capacity (per symbol) is
    C(p) = log_2 4 − H_2(p) − p · log_2 3,
useful as a downstream sanity check on threshold predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from gf4_lib import gf4_add


@dataclass(frozen=True)
class QSC:
    """Quaternary symmetric channel with per-symbol error probability p."""

    p: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.p <= 1.0):
            raise ValueError(f"p must lie in [0, 1]; got {self.p}")

    # --- Sampling ----------------------------------------------------------
    def sample_noise(
        self, n: Union[int, tuple], rng: np.random.Generator
    ) -> np.ndarray:
        """
        Sample additive noise e ∈ F_4^n (or any shape).

        Vectorised implementation: draw a Uniform[0,1) per coordinate to decide
        which symbols flip, then draw a uniform F_4* value for those positions.
        """
        size = (n,) if isinstance(n, int) else tuple(n)
        flip_mask = rng.random(size=size) < self.p
        # F_4* = {1, 2, 3} uniformly.
        nonzero_values = rng.integers(low=1, high=4, size=size, dtype=np.uint8)
        noise = np.where(flip_mask, nonzero_values, np.uint8(0)).astype(np.uint8)
        return noise

    def transmit(
        self, codeword: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Send `codeword` through the channel. Returns (received, noise).
        Caller can recover noise = received XOR codeword if needed; we return
        it explicitly because reporting BER conditional on the error pattern
        is cleaner.
        """
        cw = np.asarray(codeword, dtype=np.uint8)
        noise = self.sample_noise(cw.shape, rng)
        received = gf4_add(cw, noise)
        return received, noise

    # --- Information-theoretic sanity --------------------------------------
    @property
    def capacity_per_symbol(self) -> float:
        """Channel capacity in bits per F_4 symbol."""
        if self.p == 0.0:
            return 2.0
        if self.p == 1.0:
            return 2.0 - np.log2(3)  # remaining 3 possible outputs uniformly
        h2 = -self.p * np.log2(self.p) - (1 - self.p) * np.log2(1 - self.p)
        return 2.0 - h2 - self.p * np.log2(3)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------
def _test_noise_marginals() -> None:
    """Empirical marginals of e match (1-p, p/3, p/3, p/3)."""
    rng = np.random.default_rng(20260524)
    n = 200_000
    for p in (0.0, 0.01, 0.1, 0.5, 0.9, 1.0):
        ch = QSC(p)
        e = ch.sample_noise(n, rng)
        counts = np.bincount(e, minlength=4) / n
        expected = np.array([1 - p, p / 3, p / 3, p / 3])
        diff = np.abs(counts - expected).max()
        # Tolerance ≈ 4 sigma for a Bernoulli(p) over n trials.
        tol = max(0.005, 4 * np.sqrt(max(p, 1e-6) * (1 - p + 1e-6) / n))
        assert diff < tol, f"p={p}: counts={counts} vs expected={expected}, diff={diff}, tol={tol}"


def _test_transmit_is_additive() -> None:
    """`received - codeword = noise` exactly, for any codeword."""
    rng = np.random.default_rng(20260524)
    n = 100
    ch = QSC(0.1)
    cw = rng.integers(0, 4, size=n, dtype=np.uint8)
    rx, noise = ch.transmit(cw, rng)
    assert np.array_equal(np.bitwise_xor(rx, cw).astype(np.uint8), noise)


def _test_capacity_endpoints() -> None:
    assert abs(QSC(0.0).capacity_per_symbol - 2.0) < 1e-12
    # At p = 3/4, every output is uniform on F_4 ⇒ C = 0.
    assert abs(QSC(0.75).capacity_per_symbol - 0.0) < 1e-12


def _test_independent_rngs_decorrelate() -> None:
    """Two QSC objects sharing p but different rng streams produce uncorrelated noise."""
    n = 5000
    ch = QSC(0.2)
    e1 = ch.sample_noise(n, np.random.default_rng(1))
    e2 = ch.sample_noise(n, np.random.default_rng(2))
    # Hamming agreement rate should approach (1-p)^2 + 3 (p/3)^2 = (1-p)^2 + p^2/3.
    agree = float((e1 == e2).mean())
    expected = (1 - 0.2) ** 2 + 3 * (0.2 / 3) ** 2
    assert abs(agree - expected) < 0.02, (agree, expected)


def _run_all_tests() -> None:
    tests = [
        _test_noise_marginals,
        _test_transmit_is_additive,
        _test_capacity_endpoints,
        _test_independent_rngs_decorrelate,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} channel tests passed.\n")


if __name__ == "__main__":
    _run_all_tests()
    # Print capacity curve as a quick sanity check.
    print("Capacity (bits/symbol) at sample p values:")
    for p in (0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 0.9):
        print(f"  p = {p:.2f}:  C(p) = {QSC(p).capacity_per_symbol:.4f}")
