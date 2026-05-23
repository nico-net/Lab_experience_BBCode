"""
evaluation.py — FER/BER measurement harness for the GF(4) BB-code paper.

The harness is decoder-agnostic. A decoder is any callable with signature

    decoder(code, received_word, rng=None) -> estimated_codeword

where `received_word`, `estimated_codeword` are np.ndarray[uint8] of length
code.n holding F_4 symbols. `rng` is an optional np.random.Generator passed
through to stochastic decoders (Metropolis); deterministic decoders may
ignore it.

Monte-Carlo convention: we transmit the all-zero codeword every trial. For a
F_4-linear code over a symmetric channel the decoder's error probability is
codeword-independent, so this loses no generality. A validation test
(`_test_random_codeword_matches_zero`) verifies this empirically by also
running with random codewords drawn from the code's null-space basis.

Sub-seeding: a single user-provided `seed` deterministically derives
independent streams for (a) the channel and (b) the decoder, via
`np.random.SeedSequence.spawn`. Re-running with the same seed therefore
exactly reproduces both the noise realisations and any stochastic decoder
state.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from bb_constructor import BBCode
from channel import QSC
from gf4_lib import gf4_add, gf4_matmul

# A decoder takes (code, received, optional rng) and returns an n-vector over F_4.
Decoder = Callable[[BBCode, np.ndarray, Optional[np.random.Generator]], np.ndarray]


# ---------------------------------------------------------------------------
# Statistics: Wilson score interval (95% CI on a binomial proportion)
# ---------------------------------------------------------------------------
def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """
    Two-sided z-confidence interval for the binomial proportion successes/trials.
    Default z = 1.96 → 95% CI. Handles successes = 0 and successes = trials
    cleanly, unlike the normal approximation.
    """
    if trials == 0:
        return (0.0, 1.0)
    n = trials
    p_hat = successes / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if successes == 0 else max(0.0, center - half)
    hi = 1.0 if successes == trials else min(1.0, center + half)
    return (lo, hi)


# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    """One (code × decoder × p × num_trials) Monte-Carlo data point."""
    code_name: str
    decoder_name: str
    n: int
    k: int
    p: float
    num_trials: int
    num_frame_errors: int
    fer: float
    fer_ci_low: float
    fer_ci_high: float
    num_symbol_errors: int
    total_symbols: int
    symbol_error_rate: float
    symbol_ci_low: float
    symbol_ci_high: float
    num_bit_errors: int
    total_bits: int
    bit_error_rate: float
    bit_ci_low: float
    bit_ci_high: float
    seed: int
    elapsed_sec: float
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reproducible sub-seeding
# ---------------------------------------------------------------------------
def derive_rngs(
    seed: int, *labels: str
) -> tuple[np.random.Generator, ...]:
    """
    Deterministically derive one independent np.random.Generator per label,
    seeded from a parent SeedSequence so that streams are statistically
    independent yet fully reproducible from `seed`. The `labels` are mixed
    into the spawn key so that adding/removing decoders doesn't perturb
    earlier streams.
    """
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(labels))
    return tuple(np.random.default_rng(child) for child in children)


# ---------------------------------------------------------------------------
# Bit-level error counting
# ---------------------------------------------------------------------------
def _count_bit_errors(c_true: np.ndarray, c_hat: np.ndarray) -> int:
    """
    Number of differing bits when each F_4 symbol is expanded to its 2-bit
    representation (low + high·ω). Because XOR over F_4 == bitwise XOR of
    representations, the number of differing bits is exactly popcount(c_true ⊕ c_hat).
    """
    xor = np.bitwise_xor(c_true, c_hat).astype(np.uint8)
    return int(((xor & 1) + ((xor >> 1) & 1)).sum())


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------
def evaluate(
    code: BBCode,
    decoder: Decoder,
    p: float,
    num_trials: int,
    seed: int,
    decoder_name: str = "anonymous",
    transmit_zero: bool = True,
) -> EvalResult:
    """
    Estimate FER, symbol-error rate, and bit-error rate via Monte Carlo.

    Parameters
    ----------
    transmit_zero
        If True, every trial transmits c = 0 (standard symmetric-channel
        convention). If False, every trial draws a random codeword by
        encoding a uniform F_4-message against the codeword basis. Slower
        but useful for sanity-checking decoder symmetry.
    """
    channel_rng, decoder_rng, codeword_rng = derive_rngs(
        seed, f"channel:{code.name}:{decoder_name}:{p}",
        f"decoder:{code.name}:{decoder_name}:{p}",
        f"codeword:{code.name}:{decoder_name}:{p}",
    )
    qsc = QSC(p)
    n = code.n

    # Precompute the codeword basis only if we need random codewords; this
    # also lets us assert is_codeword on each generated input.
    basis: Optional[np.ndarray] = None
    if not transmit_zero:
        basis = code.codeword_basis()

    num_frame_errors = 0
    num_symbol_errors = 0
    num_bit_errors = 0
    total_symbols = num_trials * n
    total_bits = 2 * total_symbols

    t0 = time.perf_counter()
    for _ in range(num_trials):
        if transmit_zero:
            c_true = np.zeros(n, dtype=np.uint8)
        else:
            msg = codeword_rng.integers(0, 4, size=basis.shape[0], dtype=np.uint8)
            c_true = gf4_matmul(msg.reshape(1, -1), basis)[0]

        received, _ = qsc.transmit(c_true, channel_rng)
        c_hat = decoder(code, received, decoder_rng)

        if c_hat.shape != (n,):
            raise ValueError(
                f"decoder returned shape {c_hat.shape}, expected ({n},)"
            )
        diffs = c_hat != c_true
        sym_err = int(diffs.sum())
        num_symbol_errors += sym_err
        num_bit_errors += _count_bit_errors(c_true, c_hat)
        if sym_err > 0:
            num_frame_errors += 1
    elapsed = time.perf_counter() - t0

    fer = num_frame_errors / num_trials
    fer_lo, fer_hi = wilson_interval(num_frame_errors, num_trials)
    ser = num_symbol_errors / total_symbols
    ser_lo, ser_hi = wilson_interval(num_symbol_errors, total_symbols)
    ber = num_bit_errors / total_bits
    ber_lo, ber_hi = wilson_interval(num_bit_errors, total_bits)

    return EvalResult(
        code_name=code.name,
        decoder_name=decoder_name,
        n=n,
        k=code.k,
        p=p,
        num_trials=num_trials,
        num_frame_errors=num_frame_errors,
        fer=fer,
        fer_ci_low=fer_lo,
        fer_ci_high=fer_hi,
        num_symbol_errors=num_symbol_errors,
        total_symbols=total_symbols,
        symbol_error_rate=ser,
        symbol_ci_low=ser_lo,
        symbol_ci_high=ser_hi,
        num_bit_errors=num_bit_errors,
        total_bits=total_bits,
        bit_error_rate=ber,
        bit_ci_low=ber_lo,
        bit_ci_high=ber_hi,
        seed=seed,
        elapsed_sec=elapsed,
        extra={"transmit_zero": transmit_zero},
    )


def sweep_p(
    code: BBCode,
    decoder: Decoder,
    p_values: Sequence[float],
    num_trials: int,
    seed: int,
    decoder_name: str = "anonymous",
    transmit_zero: bool = True,
    verbose: bool = False,
) -> list[EvalResult]:
    """Run `evaluate` at each p in p_values, returning a list of results."""
    results = []
    for p in p_values:
        r = evaluate(
            code, decoder, p, num_trials, seed,
            decoder_name=decoder_name, transmit_zero=transmit_zero,
        )
        results.append(r)
        if verbose:
            print(
                f"  {code.name:>6} | {decoder_name:>12} | p={p:.4f} | "
                f"FER={r.fer:.4f} [{r.fer_ci_low:.4f}, {r.fer_ci_high:.4f}] | "
                f"trials={r.num_trials} | {r.elapsed_sec:.2f}s"
            )
    return results


# ---------------------------------------------------------------------------
# Trivial decoders for validation
# ---------------------------------------------------------------------------
def null_decoder(
    code: BBCode, received: np.ndarray, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Pass-through decoder: returns received unchanged. Useful as a baseline
    because its expected FER on the QSC has a closed form when c_true = 0:
        FER = 1 − (1 − p)^n.
    """
    return np.asarray(received, dtype=np.uint8).copy()


def zero_decoder(
    code: BBCode, received: np.ndarray, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Constant-zero decoder. With the all-zero-codeword convention this is the
    optimal trivial baseline (always correct), giving FER = 0 exactly. Used
    to verify the evaluator's counters when there are no errors.
    """
    return np.zeros(code.n, dtype=np.uint8)


def to_json(results: Sequence[EvalResult]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _test_wilson_endpoints() -> None:
    # 0 of 100 → upper bound strictly between 0 and Hanley-Lippman ≈ 0.0362.
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0 and 0.02 < hi < 0.05, (lo, hi)
    # 100 of 100 → mirror of above.
    lo, hi = wilson_interval(100, 100)
    assert hi == 1.0 and 0.95 < lo < 0.98, (lo, hi)
    # 50 of 100 → centered near 0.5 with reasonable half-width.
    lo, hi = wilson_interval(50, 100)
    assert 0.35 < lo < 0.45 and 0.55 < hi < 0.65, (lo, hi)


def _test_zero_decoder_perfect() -> None:
    """zero_decoder with transmit_zero must have FER = 0 exactly."""
    from bb_constructor import make_tiny
    code = make_tiny()
    r = evaluate(code, zero_decoder, p=0.1, num_trials=500, seed=42,
                 decoder_name="zero")
    assert r.fer == 0.0
    assert r.num_frame_errors == 0
    assert r.num_symbol_errors == 0
    assert r.num_bit_errors == 0


def _test_null_decoder_matches_analytical_fer() -> None:
    """
    For the all-zero codeword, null_decoder reports an error iff any symbol
    flipped, so FER = 1 - (1-p)^n exactly. Verify the empirical estimate is
    within 4 sigma of analytical — false-positive rate < 1 in 15000. We
    intentionally do NOT test "expected ∈ 95% CI" because that has a 5%
    failure rate by definition and would make the test suite flaky.
    """
    from bb_constructor import make_tiny
    code = make_tiny()
    n = code.n
    for p in (0.01, 0.05, 0.1):
        r = evaluate(code, null_decoder, p=p, num_trials=10_000, seed=20260524,
                     decoder_name="null")
        expected_fer = 1 - (1 - p) ** n
        sigma_fer = math.sqrt(expected_fer * (1 - expected_fer) / r.num_trials)
        assert abs(r.fer - expected_fer) < 4 * sigma_fer, (
            f"p={p}: empirical FER {r.fer:.4f} too far from analytical "
            f"{expected_fer:.4f} (sigma={sigma_fer:.4f})"
        )
        # Empirical symbol error rate must also be within 4 sigma of p.
        sigma_ser = math.sqrt(p * (1 - p) / r.total_symbols)
        assert abs(r.symbol_error_rate - p) < 4 * sigma_ser, (
            f"p={p}: SER {r.symbol_error_rate:.4f} too far from {p}"
        )


def _test_reproducibility() -> None:
    """Same seed → identical numerical result."""
    from bb_constructor import make_tiny
    code = make_tiny()
    r1 = evaluate(code, null_decoder, p=0.05, num_trials=500, seed=7,
                  decoder_name="null")
    r2 = evaluate(code, null_decoder, p=0.05, num_trials=500, seed=7,
                  decoder_name="null")
    assert r1.num_frame_errors == r2.num_frame_errors
    assert r1.num_symbol_errors == r2.num_symbol_errors
    assert r1.num_bit_errors == r2.num_bit_errors


def _test_seed_independence() -> None:
    """Different seeds → different draws (no accidental coupling)."""
    from bb_constructor import make_tiny
    code = make_tiny()
    r1 = evaluate(code, null_decoder, p=0.05, num_trials=500, seed=7,
                  decoder_name="null")
    r2 = evaluate(code, null_decoder, p=0.05, num_trials=500, seed=8,
                  decoder_name="null")
    # With 500 trials at p=0.05 we expect different exact counts.
    assert (r1.num_frame_errors != r2.num_frame_errors
            or r1.num_symbol_errors != r2.num_symbol_errors)


def _test_random_codeword_matches_zero() -> None:
    """
    Sanity check on the all-zero convention: null_decoder against random
    codewords must give the same FER as against the all-zero codeword,
    because the channel is symmetric.
    """
    from bb_constructor import make_tiny
    code = make_tiny()
    r_zero = evaluate(code, null_decoder, p=0.1, num_trials=5_000, seed=11,
                      decoder_name="null", transmit_zero=True)
    r_rand = evaluate(code, null_decoder, p=0.1, num_trials=5_000, seed=11,
                      decoder_name="null", transmit_zero=False)
    # CIs must overlap.
    assert max(r_zero.fer_ci_low, r_rand.fer_ci_low) <= min(r_zero.fer_ci_high, r_rand.fer_ci_high), (
        f"all-zero CI [{r_zero.fer_ci_low:.4f}, {r_zero.fer_ci_high:.4f}] "
        f"disjoint from random-codeword CI "
        f"[{r_rand.fer_ci_low:.4f}, {r_rand.fer_ci_high:.4f}]"
    )


def _test_sweep_p_orders_results() -> None:
    """`sweep_p` returns results in the same order as input p_values."""
    from bb_constructor import make_tiny
    code = make_tiny()
    p_values = [0.01, 0.05, 0.1]
    results = sweep_p(code, null_decoder, p_values, num_trials=200, seed=3,
                      decoder_name="null")
    assert [r.p for r in results] == p_values


def _run_all_tests() -> None:
    tests = [
        _test_wilson_endpoints,
        _test_zero_decoder_perfect,
        _test_null_decoder_matches_analytical_fer,
        _test_reproducibility,
        _test_seed_independence,
        _test_random_codeword_matches_zero,
        _test_sweep_p_orders_results,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} evaluation tests passed.\n")


if __name__ == "__main__":
    _run_all_tests()
