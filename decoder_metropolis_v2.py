"""
decoder_metropolis_v2.py — Min-cost-repair Metropolis decoder (experimental).

Energy:
    E(x) = sum_i V_i(x_i)
         + beta * sum_{c : (Hx)_c != 0}  min_{j in N(c)} (V_j(a_j^{(c)}) - V_j(x_j))

where V_i(x_i) = 0 if x_i == y_i else mu := log((1-p)/(p/3)), and
a_j^{(c)} := x_j  XOR  (s_c * h_{c,j}^{-1})  is the unique F_4 symbol that,
if placed at position j with all other neighbors of c held fixed, satisfies
check c. The penalty per failed check is the cost of its cheapest
single-symbol repair via any one neighbor.

This is *not* the textbook Sourlas-Nishimori energy. It is a reweighted-
min-sum-flavoured variant: the second term gives small (possibly negative)
penalty to checks whose neighbors include a corrupted symbol that "wants"
to snap back, and large penalty to checks whose only repair would require
corrupting a confident symbol. Provided for benchmarking against
NishimoriMetropolisDecoder; see decoder_metropolis.py for the canonical
energy used in the paper.

KNOWN PATHOLOGY (implemented faithfully, NOT patched):
    The min-over-neighbors can be negative when the cheapest repair would
    flip a currently-corrupted symbol back to its received value. In that
    case the penalty per failed check is -mu, which is *lower* than zero
    -- the energy of a satisfied check. Concretely: starting from x = y = 0
    (a codeword, E = 0) and flipping a single position x_i to omega makes
    +mu in V_i and ~3 * (-mu) in syndrome penalty, net dE = -2*mu < 0.
    The chain accepts, best-state tracking records E < 0, and the decoder
    returns a non-codeword. Fix would be a max(0, .) clamp on the per-
    check penalty; we have NOT applied it because the user's formula does
    not contain one. Empirical effect: see day5_benchmark.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bb_constructor import BBCode
from gf4_lib import _INV_TABLE, _MUL_TABLE


@dataclass
class MinCostRepairMetropolisDecoder:
    """
    Parameters
    ----------
    p : float
        Channel error rate assumed by the decoder (used to set mu).
    beta : float
        Scaling on the min-cost-repair term. Default 1.0 (since both terms
        are already in units of mu, beta is dimensionless).
    T : float
        Temperature. Default 1.0.
    num_sweeps : int
        Sweep budget; one sweep = n single-symbol updates.
    return_best : bool
        If True (default), return the lowest-E state visited.
    name : str
        Identifier carried into EvalResult.decoder_name.
    """

    p: float
    beta: float = 1.0
    T: float = 1.0
    num_sweeps: int = 200
    return_best: bool = True
    name: str = ""

    mu: float = field(init=False)
    _idx_cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (0 < self.p < 1):
            raise ValueError(f"p must be in (0, 1); got {self.p}")
        self.mu = math.log((1 - self.p) / (self.p / 3))
        if self.T <= 0:
            raise ValueError(f"T must be positive; got {self.T}")
        if self.num_sweeps <= 0:
            raise ValueError(f"num_sweeps must be positive; got {self.num_sweeps}")
        if not self.name:
            self.name = f"mincost_p{self.p:.3f}_T{self.T:.2f}_b{self.beta:.2f}"

    # ----------------------------------------------------------------------
    # One-time precomputation per code: column→checks, check→columns,
    # check→coef inverses, and per-(column, touched-check) local indices
    # so that override on the moving symbol costs O(1) inside the hot loop.
    # ----------------------------------------------------------------------
    def _get_indices(self, H: np.ndarray):
        key = id(H)
        cached = self._idx_cache.get(key)
        if cached is not None:
            return cached
        n_checks, n_vars = H.shape
        col_rows = [np.flatnonzero(H[:, i]) for i in range(n_vars)]
        col_coefs = [H[col_rows[i], i].copy() for i in range(n_vars)]
        row_cols = [np.flatnonzero(H[r, :]) for r in range(n_checks)]
        row_coefs = [H[r, row_cols[r]].copy() for r in range(n_checks)]
        row_coef_invs = [_INV_TABLE[row_coefs[r]] for r in range(n_checks)]
        # For each column i and each check c in col_rows[i], the local index
        # of i within row_cols[c]. Stored as a parallel list.
        col_local_in_row = []
        for i in range(n_vars):
            locals_for_i = []
            for c in col_rows[i]:
                where = np.flatnonzero(row_cols[c] == i)
                assert where.size == 1, (i, c, row_cols[c])
                locals_for_i.append(int(where[0]))
            col_local_in_row.append(locals_for_i)
        cache = (col_rows, col_coefs, row_cols, row_coef_invs, col_local_in_row)
        self._idx_cache[key] = cache
        return cache

    # ----------------------------------------------------------------------
    def _check_penalty(
        self,
        x_at_cols: np.ndarray,
        y_at_cols: np.ndarray,
        s_c: int,
        coef_invs: np.ndarray,
    ) -> float:
        """
        Min-cost repair penalty for one check given pre-extracted column values.
        Returns 0 if check is satisfied (s_c == 0).
        """
        if s_c == 0:
            return 0.0
        # a_j = x_j XOR (s_c * h_{c,j}^{-1})  for each j in this check's support.
        a_j = np.bitwise_xor(x_at_cols, _MUL_TABLE[s_c, coef_invs])
        dis_new = (a_j != y_at_cols).astype(np.int8)
        dis_old = (x_at_cols != y_at_cols).astype(np.int8)
        return max(0.0, self.mu * float((dis_new - dis_old).min()))
    # ----------------------------------------------------------------------
    def __call__(
        self,
        code: BBCode,
        received: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        H = code.H
        n_checks, n_vars = H.shape
        col_rows, col_coefs, row_cols, row_coef_invs, col_local_in_row = \
            self._get_indices(H)

        y = np.asarray(received, dtype=np.uint8)
        x = y.copy()

        # Initial syndrome.
        s = np.bitwise_xor.reduce(_MUL_TABLE[H, x[None, :]], axis=1).astype(np.uint8)

        # Initial per-check penalties (all zero, since x = y => zero syndrome
        # only if y is a codeword; if not, some s_c != 0 and we compute).
        check_pen = np.zeros(n_checks, dtype=np.float64)
        for c in range(n_checks):
            if s[c]:
                check_pen[c] = self._check_penalty(
                    x[row_cols[c]], y[row_cols[c]], int(s[c]), row_coef_invs[c]
                )
        # Initial V term: x == y => 0.
        E_total = float(self.beta * check_pen.sum())

        best_x = x.copy()
        best_E = E_total

        inv_T = 1.0 / self.T
        beta = self.beta
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
                if rows.size == 0:
                    dE = dV
                    new_s_at_rows = None
                    new_pens = None
                else:
                    coefs = col_coefs[i]
                    locals_in_rows = col_local_in_row[i]
                    delta = _MUL_TABLE[f, coefs]  # change to each s_c
                    new_s_at_rows = np.bitwise_xor(s[rows], delta)

                    # Recompute penalty for each affected check, both old and new.
                    pen_delta = 0.0
                    new_pens = np.empty(rows.size, dtype=np.float64)
                    for k in range(rows.size):
                        c = int(rows[k])
                        cols_c = row_cols[c]
                        coef_invs_c = row_coef_invs[c]
                        # Old at cols_c uses current x.
                        x_at_cols_old = x[cols_c]
                        y_at_cols = y[cols_c]
                        pen_old = check_pen[c]  # cached
                        # New at cols_c: same as old except x[i] -> x_new.
                        x_at_cols_new = x_at_cols_old.copy()
                        x_at_cols_new[locals_in_rows[k]] = x_new
                        pen_new = self._check_penalty(
                            x_at_cols_new, y_at_cols,
                            int(new_s_at_rows[k]), coef_invs_c,
                        )
                        new_pens[k] = pen_new
                        pen_delta += (pen_new - pen_old)
                    dE = dV + beta * pen_delta

                # Accept / reject.
                if dE <= 0.0 or uniforms[step] < math.exp(-dE * inv_T):
                    if rows.size:
                        s[rows] = new_s_at_rows
                        check_pen[rows] = new_pens
                    x[i] = x_new
                    E_total += dE
                    if self.return_best and E_total < best_E:
                        best_E = E_total
                        best_x = x.copy()
        return best_x if self.return_best else x


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _test_index_consistency() -> None:
    """col_local_in_row[i][k] should index back to i within row_cols[col_rows[i][k]]."""
    from bb_constructor import make_tiny
    code = make_tiny()
    dec = MinCostRepairMetropolisDecoder(p=0.05, num_sweeps=1)
    col_rows, col_coefs, row_cols, row_coef_invs, col_local_in_row = \
        dec._get_indices(code.H)
    for i in range(code.n):
        for k, c in enumerate(col_rows[i]):
            assert row_cols[c][col_local_in_row[i][k]] == i


def _test_documents_pathology_on_codeword_input() -> None:
    """
    Document the known energy pathology: even with y = 0 (a codeword), the
    chain finds E < 0 states and `best_x` migrates off zero. This test
    ASSERTS the broken behavior so we are alerted if the energy ever gets
    silently patched.
    """
    from bb_constructor import make_tiny
    code = make_tiny()
    dec = MinCostRepairMetropolisDecoder(p=0.05, num_sweeps=50)
    out = dec(code, np.zeros(code.n, dtype=np.uint8), np.random.default_rng(0))
    # The faithful implementation moves AWAY from y=0.
    assert not np.array_equal(out, np.zeros(code.n, dtype=np.uint8)), (
        "v2 unexpectedly preserved y=0; energy may have been patched"
    )


def _test_reproducible() -> None:
    from bb_constructor import make_tiny
    from channel import QSC
    code = make_tiny()
    dec = MinCostRepairMetropolisDecoder(p=0.05, num_sweeps=100)
    qsc = QSC(0.05)
    noise = qsc.sample_noise(code.n, np.random.default_rng(99))
    out1 = dec(code, noise, np.random.default_rng(100))
    out2 = dec(code, noise, np.random.default_rng(100))
    assert np.array_equal(out1, out2)


def _test_output_well_formed() -> None:
    """Regardless of pathology, output must be a length-n F_4 vector."""
    from bb_constructor import make_tiny
    from channel import QSC
    code = make_tiny()
    dec = MinCostRepairMetropolisDecoder(p=0.05, num_sweeps=50)
    rng = np.random.default_rng(7)
    qsc = QSC(0.05)
    noise = qsc.sample_noise(code.n, rng)
    out = dec(code, noise, rng)
    assert out.shape == (code.n,) and out.dtype == np.uint8 and (out < 4).all()


def _run_all_tests() -> None:
    tests = [
        _test_index_consistency,
        _test_documents_pathology_on_codeword_input,
        _test_reproducible,
        _test_output_well_formed,
    ]
    for t in tests:
        t()
        print(f"  ✓ {t.__name__}")
    print(f"\nAll {len(tests)} MinCostRepair tests passed.\n")


if __name__ == "__main__":
    _run_all_tests()
