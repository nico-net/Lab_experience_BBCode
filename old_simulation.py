#!/usr/bin/env python3
"""
Biological Quantum Error Correction -- GF(4) Topological Lattice
Monte Carlo Simulation Suite

Models DNA as an effective GF(4) topological code on a periodic
L x L lattice with overlapping 2x2 plaquette stabilizers.
Decoder: Thermodynamic relaxation via Metropolis-Hastings MCMC.

Experiments:
  1. Threshold & Scaling  -- p_c identification via curve crossing
  2. Metastability        -- Rugged energy landscape time-series
  3. Localization         -- Defect confinement verification
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import deque
import time

# ── Publication-quality plot defaults ─────────────────────────────────
rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.linewidth': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'figure.dpi': 150,
})

# ======================================================================
#  GF(4) ARITHMETIC
# ======================================================================
# GF(4) = GF(2)[x]/(x^2+x+1) = {0, 1, alpha, alpha+1} -> {0, 1, 2, 3}
# Addition is bitwise XOR (characteristic 2).
# Pauli correspondence: 0->I, 1->X, 2->Z, 3->Y (up to phases).


# ======================================================================
#  LATTICE & HAMILTONIAN
# ======================================================================
class GF4TopoLattice:
    """
    L x L periodic lattice with GF(4) node values and 2x2 plaquette
    stabilizer Hamiltonian.

    Plaquette (i,j) covers nodes: (i,j), (i+1,j), (i,j+1), (i+1,j+1)
    with periodic wrapping.  There are L^2 plaquettes total.

    Syndrome:  s_p = a XOR b XOR c XOR d   (GF(4) sum)
    Energy:    g_p = -1 if s_p == 0 (calm), +1 otherwise (frustrated)
    Total:     E   = sum(g_p)  in  [-L^2, +L^2]
    """

    def __init__(self, L):
        self.L = L
        self.lattice = np.zeros((L, L), dtype=np.int8)
        # Pre-compute shifted index arrays for fast plaquette lookup
        self._ip1 = np.arange(L)              # (i+1) % L for each i
        self._ip1 = np.roll(np.arange(L), -1)
        self._im1 = np.roll(np.arange(L), 1)

    # -- Vectorised energy -------------------------------------------
    def _syndrome_array(self):
        """Return L x L array of plaquette syndromes (vectorised)."""
        g = self.lattice
        return (g
                ^ np.roll(g, -1, axis=0)
                ^ np.roll(g, -1, axis=1)
                ^ np.roll(np.roll(g, -1, axis=0), -1, axis=1))

    def total_energy(self):
        """E = 2 * (# frustrated) - L^2.  Ground state E = -L^2."""
        n_frust = int(np.count_nonzero(self._syndrome_array()))
        return 2 * n_frust - self.L * self.L

    def ground_state_energy(self):
        return -(self.L * self.L)

    # -- Local energy (4 plaquettes touching one node) ----------------
    def _node_energy(self, r, c):
        """Energy of the 4 plaquettes that contain node (r, c)."""
        L = self.L
        g = self.lattice
        E = 0
        for di in (0, -1):
            for dj in (0, -1):
                pi = (r + di) % L
                pj = (c + dj) % L
                s = (g[pi, pj]
                     ^ g[(pi + 1) % L, pj]
                     ^ g[pi, (pj + 1) % L]
                     ^ g[(pi + 1) % L, (pj + 1) % L])
                E += -1 if s == 0 else 1
        return E

    # -- Noise models -------------------------------------------------
    def apply_iid_noise(self, p):
        """Depolarising channel: each node gets a random GF(4)\\{0}
        error with probability p."""
        L = self.L
        mask = np.random.random((L, L)) < p
        errors = np.random.randint(1, 4, size=(L, L)).astype(np.int8)
        self.lattice[mask] ^= errors[mask]

    def apply_localized_string(self, length):
        """Inject a short horizontal error string."""
        L = self.L
        r = np.random.randint(L)
        c0 = np.random.randint(L)
        for k in range(length):
            self.lattice[r, (c0 + k) % L] ^= np.random.randint(1, 4)

    # -- Metropolis-Hastings single step -------------------------------
    def _metropolis_step(self, T):
        """Single-site Metropolis update.  Returns accepted dE."""
        L = self.L
        r = np.random.randint(L)
        c = np.random.randint(L)

        E_before = self._node_energy(r, c)

        old_val = self.lattice[r, c]
        flip = np.random.randint(1, 4)
        self.lattice[r, c] = old_val ^ flip

        E_after = self._node_energy(r, c)
        dE = E_after - E_before

        if dE <= 0:
            return dE
        elif T > 0 and np.random.random() < np.exp(-dE / T):
            return dE
        else:
            self.lattice[r, c] = old_val
            return 0

    # -- Optimised batch Metropolis sweep ------------------------------
    def _metropolis_sweep_batch(self, T, steps_per_sweep=None):
        """Run one sweep of Metropolis updates with pre-generated
        random numbers for reduced Python overhead."""
        L = self.L
        n = steps_per_sweep or (L * L)

        # Pre-generate all random numbers for this sweep
        rows = np.random.randint(0, L, size=n)
        cols = np.random.randint(0, L, size=n)
        flips = np.random.randint(1, 4, size=n)
        rands = np.random.random(n)

        for k in range(n):
            r, c = int(rows[k]), int(cols[k])
            E_before = self._node_energy(r, c)

            old_val = self.lattice[r, c]
            self.lattice[r, c] = old_val ^ int(flips[k])

            E_after = self._node_energy(r, c)
            dE = E_after - E_before

            if dE > 0:
                if T <= 0 or rands[k] >= np.exp(-dE / T):
                    self.lattice[r, c] = old_val  # reject

    # -- MCMC decoder (full run) --------------------------------------
    def run_decoder(self, T, max_sweeps, record_energy=False):
        """
        Run Metropolis MCMC decoder.
        One 'sweep' = L^2 single-site updates.
        """
        E_gs = self.ground_state_energy()
        energies = []

        for sweep in range(max_sweeps):
            self._metropolis_sweep_batch(T)

            if record_energy:
                E = self.total_energy()
                energies.append(E)
                if E == E_gs:
                    break
            else:
                if (sweep + 1) % 5 == 0:
                    if self.total_energy() == E_gs:
                        break

        return energies if record_energy else None

    # -- Logical failure check ----------------------------------------
    def is_in_codespace(self):
        """True if ALL syndromes vanish."""
        return self.total_energy() == self.ground_state_energy()

    def has_logical_error(self):
        """Logical error = decoder didn't converge OR converged to
        wrong coset (lattice not all-zero)."""
        if not self.is_in_codespace():
            return True
        return bool(np.any(self.lattice != 0))

    # -- Defect cluster analysis --------------------------------------
    def max_defect_cluster(self):
        """Largest connected component of frustrated plaquettes
        (4-connected on the periodic grid)."""
        L = self.L
        syn = self._syndrome_array()
        frustrated = syn != 0

        if not np.any(frustrated):
            return 0

        visited = np.zeros((L, L), dtype=bool)
        max_sz = 0

        for i in range(L):
            for j in range(L):
                if frustrated[i, j] and not visited[i, j]:
                    queue = deque([(i, j)])
                    visited[i, j] = True
                    sz = 0
                    while queue:
                        ci, cj = queue.popleft()
                        sz += 1
                        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ni, nj = (ci + di) % L, (cj + dj) % L
                            if frustrated[ni, nj] and not visited[ni, nj]:
                                visited[ni, nj] = True
                                queue.append((ni, nj))
                    if sz > max_sz:
                        max_sz = sz

        return max_sz


# ======================================================================
#  EXPERIMENT 1 -- THRESHOLD & SCALING
# ======================================================================
def experiment_threshold(
    L_values=(6, 12, 24),
    p_values=np.linspace(0.01, 0.30, 16),
    N_trials=50,
    T=0.8,
    max_sweeps=200,
    save_path='plot1_threshold_scaling.png',
):
    """
    For each (L, p) pair, run N_trials independent noise -> decode cycles.
    P_rec = fraction that recover without logical error.
    The crossing point of curves identifies p_c.
    """
    print("=" * 64)
    print("  EXPERIMENT 1: THRESHOLD & SCALING")
    print("=" * 64)

    results = {}

    for L in L_values:
        P_rec = np.zeros(len(p_values))
        print(f"\n  L = {L}  (max_sweeps = {max_sweeps})")

        for ip, p in enumerate(p_values):
            successes = 0
            for trial in range(N_trials):
                lat = GF4TopoLattice(L)
                lat.apply_iid_noise(p)
                lat.run_decoder(T, max_sweeps)
                if not lat.has_logical_error():
                    successes += 1
            P_rec[ip] = successes / N_trials
            print(f"    p = {p:.3f}  ->  P_rec = {P_rec[ip]:.3f}"
                  f"  ({successes}/{N_trials})")

        results[L] = P_rec

    # -- Plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    markers = ['o', 's', '^', 'D', 'v']
    colors = ['#1565C0', '#C62828', '#2E7D32', '#F57F17', '#6A1B9A']

    for idx, L in enumerate(L_values):
        ax.plot(p_values, results[L],
                marker=markers[idx % len(markers)],
                color=colors[idx % len(colors)],
                linewidth=2.0, markersize=5,
                label=f'$L = {L}$')

    ax.set_xlabel(r'Mutation / Error Rate  $p$', fontsize=13)
    ax.set_ylabel(r'Recovery Probability  $P_{\mathrm{rec}}$', fontsize=13)
    ax.set_title('GF(4) Topological Code -- Threshold Behaviour',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_xlim(p_values[0], p_values[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  [OK]  Saved -> {save_path}")

    return results


# ======================================================================
#  EXPERIMENT 2 -- METASTABILITY
# ======================================================================
def experiment_metastability(
    L=12,
    p=0.10,
    T=0.8,
    total_sweeps=600,
    save_path='plot2_metastability.png',
):
    """
    Single run: inject noise then record E after every MCMC sweep.
    Near threshold the energy landscape is rugged with metastable plateaus.
    """
    print("\n" + "=" * 64)
    print("  EXPERIMENT 2: METASTABILITY  (Energy Landscape)")
    print("=" * 64)

    lat = GF4TopoLattice(L)
    lat.apply_iid_noise(p)
    E_init = lat.total_energy()
    E_gs = lat.ground_state_energy()

    print(f"  L = {L},  p = {p:.3f},  T = {T:.2f}")
    print(f"  Initial energy  E0 = {E_init}")
    print(f"  Ground state    Egs = {E_gs}")

    energies = lat.run_decoder(T, total_sweeps, record_energy=True)

    # -- Plot ---------------------------------------------------------
    sweeps = np.arange(1, len(energies) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweeps, energies,
            color='#AD1457', linewidth=1.0, alpha=0.85,
            label='Energy trajectory')
    ax.axhline(y=E_gs, color='#2E7D32', ls='--', lw=1.8,
               label=f'Ground state $E = {E_gs}$')

    ax.set_xlabel('MCMC Sweeps  (1 sweep = $L^2$ single-site updates)',
                  fontsize=12)
    ax.set_ylabel('Total System Energy  $E$', fontsize=13)
    ax.set_title(
        f'Metastable Plateaus in Rugged Energy Landscape\n'
        f'($L={L}$,  $p={p}$,  $T={T}$)',
        fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close(fig)

    print(f"  Final energy = {energies[-1]}")
    print(f"  [OK]  Saved -> {save_path}")
    return energies


# ======================================================================
#  EXPERIMENT 3 -- LOCALIZATION  (Defect Confinement)
# ======================================================================
def experiment_localization(
    L=12,
    string_length=3,
    T=0.8,
    total_sweeps=300,
    N_trials=20,
    save_path='plot3_localization.png',
):
    """
    Below threshold: inject short localised error strings and track
    the maximum connected cluster of frustrated plaquettes over
    MCMC time.  If defects are confined (xi < inf), the cluster
    size remains bounded and decays to zero.
    """
    print("\n" + "=" * 64)
    print("  EXPERIMENT 3: DEFECT LOCALIZATION")
    print("=" * 64)
    print(f"  L = {L},  string_length = {string_length},  T = {T}")
    print(f"  N_trials = {N_trials},  total_sweeps = {total_sweeps}")

    all_curves = np.zeros((N_trials, total_sweeps + 1))

    for trial in range(N_trials):
        lat = GF4TopoLattice(L)
        lat.apply_localized_string(string_length)

        all_curves[trial, 0] = lat.max_defect_cluster()

        for sweep in range(1, total_sweeps + 1):
            lat._metropolis_sweep_batch(T)
            all_curves[trial, sweep] = lat.max_defect_cluster()

        if (trial + 1) % 5 == 0:
            print(f"    trial {trial + 1}/{N_trials} done")

    avg_cluster = all_curves.mean(axis=0)
    std_cluster = all_curves.std(axis=0)
    sweeps = np.arange(total_sweeps + 1)

    # -- Plot ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.fill_between(sweeps,
                    np.maximum(avg_cluster - std_cluster, 0),
                    avg_cluster + std_cluster,
                    color='#7B1FA2', alpha=0.15,
                    label=r'$\pm 1\sigma$')
    ax.plot(sweeps, avg_cluster,
            color='#7B1FA2', linewidth=2.0,
            label='Mean max cluster size')
    ax.axhline(y=L, color='#B71C1C', ls=':', lw=1.5,
               label=f'Lattice extent $L = {L}$  (percolation)')

    ax.set_xlabel('MCMC Sweeps', fontsize=13)
    ax.set_ylabel(r'Avg. Max Defect Cluster Size  $\xi$', fontsize=13)
    ax.set_title(
        f'Defect Confinement Below Threshold\n'
        f'($L={L}$, string length = {string_length})',
        fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.close(fig)

    print(f"  Final avg max cluster = {avg_cluster[-1]:.2f}")
    print(f"  [OK]  Saved -> {save_path}")
    return avg_cluster


# ======================================================================
#  MAIN -- Run all three experiments
# ======================================================================
if __name__ == '__main__':

    np.random.seed(42)
    t_start = time.time()

    # -- Experiment 1: Threshold & Scaling ----------------------------
    res1 = experiment_threshold(
        L_values=[6, 12, 24],
        p_values=np.linspace(0.01, 0.30, 16),
        N_trials=50,
        T=0.8,
        max_sweeps=200,
    )

    # -- Experiment 2: Metastability ----------------------------------
    res2 = experiment_metastability(
        L=12, p=0.10, T=0.8, total_sweeps=600,
    )

    # -- Experiment 3: Localization -----------------------------------
    res3 = experiment_localization(
        L=12, string_length=3, T=0.8,
        total_sweeps=300, N_trials=20,
    )

    elapsed = time.time() - t_start
    print(f"\n{'=' * 64}")
    print(f"  Total wall-clock time: {elapsed:.1f} s  "
          f"({elapsed / 60:.1f} min)")
    print(f"{'=' * 64}")
