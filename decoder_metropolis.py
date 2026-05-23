"""
decoder_metropolis.py — Sourlas-Nishimori Metropolis decoder.

Energy (canonical):
    E(x) = sum_i V_i(x_i)  +  gamma * | {c : (H x)_c != 0} |

where V_i(x_i) = 0 if x_i == y_i, else mu := log((1-p) / (p/3)).
The per-symbol cost mu is the log-likelihood ratio for a single QSC symbol;
the syndrome term is the textbook indicator penalty. With p known and T set
to the Nishimori value T = 1, the Boltzmann distribution at this energy is
the Bayes posterior over the codeword given the received word — i.e. the
chain (in equilibrium) samples the ML decoder's belief. See Sourlas (1989)
and Nishimori (2001) for derivations.

API contract: callable as `dec(code, received, rng) -> estimated_codeword`,
conforming to evaluation.Decoder.

Notes vs the earlier syndrome-only baseline:
- We do NOT early-stop on zero syndrome. Once a codeword is reached, the
  chain can still improve by migrating between codewords (crossing
  syndrome barriers of height gamma per check), and the V term gives the
  right gradient toward the ML codeword.
- We return the lowest-E state visited, not the final state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bb_constructor import BBCode
from gf4_lib import _MUL_TABLE


@dataclass
class NishimoriMetropolisDecoder:
    """
    Parameters
    ----------
    p : float
        Channel error rate assumed by the decoder (used to set mu and T).
    gamma : float, optional
        Syndrome penalty per failed check. Default = max(1.0, mu).
    T : float, optional
        Temperature. Default = 1.0 (the Nishimori line for our V scaling).
    num_sweeps : int
        Maximum sweep budget; one sweep = n single-symbol updates.
    return_best : bool
        If True (default), return the lowest-E state visited.
    name : str
        Identifier carried into EvalResult.decoder_name.
    """

    p: float
    gamma: Optional[float] = None
    T: Optional[float] = None
    num_sweeps: int = 200
    return_best: bool = True
    name: str = ""

    mu: float = field(init=False)
    _col_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (0 < self.p < 1):
            raise ValueError(f"p must be in (0, 1); got {self.p}")
        self.mu = math.log((1 - self.p) / (self.p / 3))
        if self.gamma is None:
            self.gamma = max(1.0, self.mu)
        if self.T is None:
            self.T = 1.0
        if self.T <= 0:
            raise ValueError(f"T must be positive; got {self.T}")
        if self.num_sweeps <= 0:
            raise ValueError(f"num_sweeps must be positive; got {self.num_sweeps}")
        if not self.name:
            self.name = f"nishimori_p{self.p:.3f}_T{self.T:.2f}_g{self.gamma:.2f}"

    def _get_columns(self, H: np.ndarray):
        key = id(H)
        cached = self._col_cache.get(key)
        if cached is not None:
            return cached
        n_vars = H.shape[1]
        col_rows = [np.flatnonzero(H[:, i]) for i in range(n_vars)]
        col_coefs = [H[col_rows[i], i].copy() for i in range(n_vars)]
        self._col_cache[key] = (col_rows, col_coefs)
        return self._col_cache[key]

    def __call__(
        self,
        code: BBCode,
        received: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        H = code.H
        n_vars = H.shape[1]
        col_rows, col_coefs = self._get_columns(H)

        y = np.asarray(received, dtype=np.uint8)
        x = y.copy()
        s = np.bitwise_xor.reduce(_MUL_TABLE[H, x[None, :]], axis=1).astype(np.uint8)
        violations = int(np.count_nonzero(s))
        E_total = float(self.gamma * violations)  # V(x)=0 since x==y initially

        best_x = x.copy()
        best_E = E_total

        inv_T = 1.0 / self.T
        gamma = self.gamma
        mu = self.mu

        for _ in range(self.num_sweeps):
            positions = rng.permutation(n_vars)
            flips = rng.integers(1, 4, size=n_vars, dtype=np.uint8)
            uniforms = rng.random(size=n_vars)
            for step in range(n_vars):
                i = int(positions[step])
                f = int(flips[step])
                x_old = int(x[i])
                x_new = x_old ^ f
                y_i = int(y[i])
                was_dis = int(x_old != y_i)
                will_dis = int(x_new != y_i)
                dV = mu * (will_dis - was_dis)
                rows = col_rows[i]
                if rows.size:
                    coefs = col_coefs[i]
                    delta = _MUL_TABLE[f, coefs]
                    old_sc = s[rows]
                    new_sc = np.bitwise_xor(old_sc, delta)
                    d_violations = (
                        int(np.count_nonzero(new_sc)) - int(np.count_nonzero(old_sc))
                    )
                else:
                    new_sc = None
                    d_violations = 0
                dE = dV + gamma * d_violations
                if dE <= 0.0 or uniforms[step] < math.exp(-dE * inv_T):
                    if rows.size:
                        s[rows] = new_sc
                    x[i] = x_new
                    violations += d_violations
                    E_total += dE
                    if self.return_best and E_total < best_E:
                        best_E = E_total
                        best_x = x.copy()
        return best_x if self.return_best else x


# Aliases for clarity in the rest of the project.
MetropolisDecoder = NishimoriMetropolisDecoder  # default Metropolis = Nishimori


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _test_zero_input_returns_zero() -> None:
    from bb_constructor import make_tiny
    code = make_tiny()
    dec = NishimoriMetropolisDecoder(p=0.05, num_sweeps=50)
    out = dec(code, np.zeros(code.n, dtype=np.uint8), np.random.default_rng(0))
    assert np.array_equal(out, np.zeros(code.n, dtype=np.uint8))


def _test_codeword_input_preserved() -> None:
    from bb_constructor import make_tiny
    code = make_tiny()
    cw = code.codeword_basis()[0]
    dec = NishimoriMetropolisDecoder(p=0.05, num_sweeps=50)
    out = dec(code, cw, np.random.default_rng(1))
    # E_initial = 0 (codeword, x==y), so best-E stays 0 and best-x stays y.
    assert np.array_equal(out, cw)


def _test_low_noise_high_recovery() -> None:
    from bb_constructor import make_tiny
    from channel import QSC
    code = make_tiny()
    dec = NishimoriMetropolisDecoder(p=0.01, num_sweeps=200)
    channel_rng = np.random.default_rng(42)
    decoder_rng = np.random.default_rng(43)
    qsc = QSC(0.01)
    zero = np.zeros(code.n, dtype=np.uint8)
    successes = sum(
        np.array_equal(dec(code, qsc.sample_noise(code.n, channel_rng), decoder_rng), zero)
        for _ in range(200)
    )
    assert successes >= 180, f"Recovery rate {successes/200:.2f} too low at p=0.01"


def _test_reproducible() -> None:
    from bb_constructor import make_tiny
    from channel import QSC
    code = make_tiny()
    dec = NishimoriMetropolisDecoder(p=0.05, num_sweeps=100)
    qsc = QSC(0.05)
    noise = qsc.sample_noise(code.n, np.random.default_rng(99))
    out1 = dec(code, noise, np.random.default_rng(100))
    out2 = dec(code, noise, np.random.default_rng(100))
    assert np.array_equal(out1, out2)


def _test_p_validation() -> None:
    for bad_p in (0.0, 1.0, -0.1, 1.5):
        try:
            NishimoriMetropolisDecoder(p=bad_p)
        except ValueError:
            continue
        raise AssertionError(f"Should reject p={bad_p}")


def _run_all_tests() -> None:
    tests = [
        _test_zero_input_returns_zero,
        _test_codeword_input_preserved,
        _test_low_noise_high_recovery,
        _test_reproducible,
        _test_p_validation,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} Nishimori tests passed.\n")


if __name__ == "__main__":
    _run_all_tests()
