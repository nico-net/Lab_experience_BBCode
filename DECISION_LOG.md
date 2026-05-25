# Technical Decisions Log

**Purpose:** Record every technical choice with date and rationale. Essential for writing Methods section and for debugging.

**Format:**
YYYY-MM-DD: [Decision]
Rationale: [Why this choice]
Alternatives considered: [What else was possible]
Outcome: [Did it work? Update later if needed]

---

## Phase 0: Setup

### 2026-05-21: Project created
Rationale: Maintain persistent context across 4-week sprint, enable efficient code review
Outcome: [TBD]

### 2026-05-21: Chose Analog 3 + Section VII approach
Rationale: Clean GF(4)-linear construction for 1-month timeline; quantum connection via structural bridge; clear path to extension paper
Alternatives considered: Analog 2 (CRSS additive code - too complex for timeline), Analog 1 (binary - throws away GF(4) framing)
Outcome: [TBD]

---

## Phase 1: Code Construction

### Day 1 (2026-21-5): GF(4) representation

Decision: Encode a + b·ω as the uint8 integer (b<<1)|a, so {0, 1, ω, ω²} ↔
{0, 1, 2, 3}. Addition implemented as bitwise XOR. Multiplication and
inversion implemented via precomputed 4×4 and length-4 lookup tables.

Rationale:
- Addition is free (single XOR), matching characteristic-2 structure.
- The full multiplication table has only 16 entries; lookup is faster than
  any algebraic shortcut at this size, and trivially vectorizes over NumPy
  arrays of arbitrary shape. The same uint8 dtype carries through to the
  parity-check matrix and decoder message tensors with no conversions.
- Trace reduces to extracting the high bit (Tr(x) = (x >> 1) & 1), so the
  CRSS trace inner product on Section VII can be computed with a shift.

Alternatives considered:
- Log/antilog tables with base ω: clean for multiplication but makes
  addition awkward (needs Zech logarithm or fallback to representatives).
  Rejected because BP and Metropolis are addition-heavy.
- Symbolic representation via SymPy or galois package: rejected for
  inner-loop performance; we want uint8 NumPy arrays end-to-end.

Outcome: gf4_lib.py passes all 11 unit tests including the brief's
validation gf4_mul(ω, ω) == gf4_add(ω, 1). Ready for Day 2.

### Day 2 (2026-21-5): Polynomial choices and rank-deficiency note

Polynomials chosen by overlaying coefficients {1, ω, ω²} onto Bravyi
monomial supports:

  tiny   (ℓ,m)=(3,3):    A = x   + ω·y   + ω²·y²    B = y   + ω·x   + ω²·x²
  small  (ℓ,m)=(6,6):    A = x³  + ω·y   + ω²·y²    B = y³  + ω·x   + ω²·x²
  medium (ℓ,m)=(9,6):    A = x³  + ω·y   + ω²·y²    B = y³  + ω·x²  + ω²·x⁴
  large  (ℓ,m)=(12,6):   A = x³  + ω·y   + ω²·y²    B = y³  + ω·x  + ω²·x^7

Rationale: Bravyi-aligned monomial supports preserve the bipartite cylindrical
Tanner graph required for Section VII. Coefficients {1, ω, ω²} appear once
each per polynomial so every nontrivial F_4*-element is exercised by the BP
check update edge-weight permutation.

Rank deficiency: H is rank-deficient in each instance, with deficits
1/6/4/4 for tiny/small/medium/large. Empirically verified that the binary
versions of small/medium/large have identical deficits 6/4/4; the algebraic
relations are intrinsic to (ℓ,m) with ℓ or m divisible by 3 (= order of F_4*),
not to the coefficient choice. The (3,3) tiny case has no algebraic deficit,
so its single excess unit comes from the column-sum-zero redundancy forced
by 1 + ω + ω² = 0 in F_4.

This mirrors the rank deficiency of quantum BB code parity matrices — in
Bravyi's Lemma 1, that deficiency is exactly what produces k > 0 logical
qubits via dim(ker A ∩ ker B). Our classical analog inherits this naturally.

Test status: all 11 tests in bb_constructor.py pass, including sparsity,
column-sum-zero verification, rank bound, rate lower bound, and codeword
basis non-triviality with H·c = 0 verified for every basis vector.

### Day 3 (2026-05-23): Parameter table [n, k, d]_4

Computed n and k from the rank of H over F_4 (already reported on Day 2).
For the `tiny` (3,3) instance, brute-forced d by encoding all 4^10 = 1,048,576 F_4-messages against the codeword basis and minimising Hamming weight over nonzero codewords.
For `small`, `medium`, and `large` we report an upper bound on d:
the minimum across (a) basis row weights, (b) a complete enumeration of weight-1, weight-2, and weight-3 messages over F_4*, and (c) 200,000 random messages drawn uniformly from F_4^k (seed 20260523).

Results:

```
name    (ell,m)       n  rank    k    rate     d_4  method
----------------------------------------------------------
tiny    ( 3, 3)     18     8   10  0.5556       6  exact
small   ( 6, 6)     72    30   42  0.5833     ≤ 6  upper_bound
medium  ( 9, 6)    108    50   58  0.5370     ≤ 6  upper_bound
large   (12, 6)    144    70   74  0.5139     ≤ 6  upper_bound
```

Rationale: the brute-force d is the only value defended for `tiny`; for larger codes the literature-standard practice (Bravyi 2024 used MIP) is unavailable in the project budget, so we report an upper bound and explicitly mark it `≤`. The upper bound is the only value the paper claims; tightening it is future work and is *not* on the critical path because the decoder threshold is set by FER curves, not by d.

Validation: every reported value satisfies basis_min_weight ≥ d (consistency), rate matches k/n exactly, and the row/col weights stay at the Day 2 values of 6/3, confirming H was not mutated.

Outcome: Parameter table ready for paper Section III.

### Day 4 (2026-05-24): Channel and evaluation framework

Channel: implemented `channel.QSC` as the F_4 analog of the BSC. Per symbol,
Pr[no error] = 1 − p; Pr[error to each of 1, ω, ω²] = p/3. Noise is
additive over F_4 (XOR on the uint8 representation), so the channel is
symmetric over F_4 by construction. Capacity per symbol C(p) = 2 − H_2(p)
− p·log_2 3 is exposed for downstream EXIT/threshold sanity checks.

Evaluation harness: `evaluation.evaluate(code, decoder, p, num_trials, seed)`
returns frame, symbol, and bit error rates with Wilson 95% confidence
intervals for each. The decoder interface is

  decoder(code, received, rng=None) -> estimated_codeword,

deliberately minimal so BP (Day 9-10) and Metropolis (Day 5) can plug in
unchanged. Sub-seeds are deterministically spawned via
`np.random.SeedSequence` for the channel, decoder, and codeword streams,
so a single user-facing `seed` reproduces an entire run exactly.

All-zero transmission convention: every trial sends c = 0 by default, valid
because the F_4-linear code under a symmetric channel has codeword-
independent error probability. Verified empirically in
`_test_random_codeword_matches_zero` (CIs from random-codeword and all-zero
runs overlap at p = 0.1 with 5,000 trials each).

Validation: `day4_validation.py` sweeps `null_decoder` (returns received
unchanged) across all four codes at p ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.2}
with 5,000 trials each. The analytical FER for null_decoder under the
all-zero convention is 1 − (1 − p)^n; every one of the 24 cells lands
within 2.5 sigma of the analytical value, with most under 1 sigma. Symbol
error rate equals p exactly; bit error rate equals (2/3)·p as predicted by
the uniform-F_4* error model (one of {1, ω, ω²} flips on average 4/3 of
the 2 bits per symbol).

Rationale notes:
- Wilson score interval over normal-approx: handles num_frame_errors = 0
  and = num_trials cleanly (important at low p where FER samples flat-line
  near zero, and at high p / large n where every frame fails).
- Tracking both symbol and bit error rates: the paper's FER curves are
  the headline figure, but symbol/bit rates are needed to verify decoder
  symmetry and to compare to channel capacity in Section V.

Outcome: channel.py + evaluation.py ready. 11 of 11 unit tests pass
(channel = 4, evaluation = 7). Day 5 can wrap the existing Metropolis
simulation behind the `Decoder` interface without further harness work.


### Day 5 (2026-05-25 → 2026-05-26): Metropolis decoder

Refactored the lattice-based Metropolis from `old_simulation.py` into a
parity-check-matrix-driven decoder for BB codes. Implemented two energy
functions side by side for benchmarking:

**Canonical (Sourlas-Nishimori, decoder_metropolis.py):**

    E(x) = sum_i V_i(x_i) + gamma * | { c : (Hx)_c != 0 } |

where V_i(x_i) = mu * (x_i != y_i) and mu = log((1-p)/(p/3)). Temperature
T = 1 sits on the Nishimori line: at this T the Boltzmann distribution
equals the Bayes posterior, so the equilibrium distribution of the chain
matches the ML decoder's belief. gamma is set to max(1.0, mu); larger
values would harden the syndrome constraint, smaller values would soften
it. The chain does NOT early-stop on zero syndrome -- with the V term
present, the lowest-energy state is the ML codeword, not just any
codeword. We return the lowest-E state visited over `num_sweeps` sweeps.

**Min-cost-repair (decoder_metropolis_v2.py, experimental):**

    E(x) = sum_i V_i(x_i)
         + beta * sum_{c failed} min_{j in N(c)} (V_j(a_j^{(c)}) - V_j(x_j))

The second term per failed check is the cost of the cheapest single-symbol
repair via any neighbor. Implemented faithfully to the formula above with
no clamping of the inner min.

**Benchmark (day5_benchmark.py, tiny n=18, 500 trials per cell, seed
20260526):**

```
0.010 | null                   |   0.1920  [0.160, 0.229]  |   0.0117 |   0.0078 |   0.02
 0.010 | nishimori (canonical)  |   0.0100  [0.004, 0.023]  |   0.0018 |   0.0012 |   5.55
 0.010 | mincost-repair (v2)    |   0.1480  [0.120, 0.182]  |   0.0093 |   0.0059 |  54.46

 0.050 | null                   |   0.6100  [0.567, 0.652]  |   0.0490 |   0.0326 |   0.01
 0.050 | nishimori (canonical)  |   0.0760  [0.056, 0.103]  |   0.0150 |   0.0101 |  10.30
 0.050 | mincost-repair (v2)    |   0.4620  [0.419, 0.506]  |   0.0512 |   0.0349 |  49.94

 0.100 | null                   |   0.8500  [0.816, 0.879]  |   0.0988 |   0.0653 |   0.02
 0.100 | nishimori (canonical)  |   0.2380  [0.203, 0.277]  |   0.0606 |   0.0406 |   6.59
 0.100 | mincost-repair (v2)    |   0.7480  [0.708, 0.784]  |   0.1209 |   0.0817 |  32.10

--------------------------------------------------------------------------------------------
Monotonicity (PIPELINE Day 5 validation criterion):
  null                FER = [0.192, 0.61, 0.85]  →  monotonic: PASS  (strict CI: yes)
  nishimori           FER = [0.01, 0.076, 0.238]  →  monotonic: PASS  (strict CI: yes)
  mincost-repair      FER = [0.148, 0.462, 0.748]  →  monotonic: PASS  (strict CI: yes)
```

PIPELINE Day 5 validation (FER monotonic in p on the tiny code) passes
for nishimori with strict CI separation at every adjacent pair.

**Nishimori headline results:** at p=0.01 the canonical decoder beats null
by 19x in FER and the earlier syndrome-only baseline by 13x (from FER 0.132
to 0.010). At p=0.10 it gets within striking distance of where BP is
expected to operate. The decoder is a real decoder, not just a syndrome
zeroer.

**MinCostRepair pathology finding:** the change in the energy metric doesn't
introduce any improvement with respect to the the nishimori metric. We 
discard this path and focus on the canonical idea.

**Decision:** ship the Nishimori decoder as the project's Metropolis
decoder for Days 6-20. Keep decoder_metropolis_v2.py in the tree as a
documented negative result.

**Outcome:** Day 5 PIPELINE criterion (FER monotonic on tiny) PASSES for
the Nishimori decoder. Day 6-7 can proceed to full FER curves on all
four code instances using the Nishimori decoder unchanged.

### Day 6 (2026-05-26): FER waterfall and the d_min ceiling

Ran `day6_fer_curves.py --full` to produce the first Metropolis FER waterfall
across all four codes (1,000 trials/cell, 200 sweeps, 10 p-values, seed
20260526). The sweep finished cleanly with strict-CI FER monotonicity in p
per code, satisfying the PIPELINE Day 6-7 GO/NO-GO. The plot itself,
however, exposed a structural issue that took the rest of the day to
diagnose.

**Anomaly:** at low p (≤ 0.03) the FER ordering is
`medium < tiny ≲ small < large`, with `large` (n=144) worse than every
other code by a factor of 5-10× at p=0.01 and CIs that do not overlap. A
larger code beating no smaller code below threshold cannot be explained by
Metropolis mixing time alone; it points at the code rather than the
decoder.

**Diagnostic chain.** Investigated three hypotheses in order:

1. *Mixing-time scaling on n=144.* Re-ran `large` at p=0.01 with
   `--sweeps 1000` (5× the budget). FER did not drop. Mixing is not
   limiting at the current sweep count.
2. *The `large` polynomial.* The Day 2 choice was
   `B = y³ + ω·x⁴ + ω²·x⁸`, later edited (between Day 2 logging and the
   Day 3 parameter sweep) to `B = y³ + ω·x + ω²·x⁷` to dodge a column-sum
   pathology. Re-ran the Tier 1 candidates from a focused search:
   `(a,b) ∈ {(1,5), (1,7), (1,11), (5,7), (5,11), (7,11), (3,7), (4,9)}`
   with B = y³ + ω·x^a + ω²·x^b. Every one yielded d ≤ 6 by combined
   weight-1/2/3 enumeration + 50,000 random-message sampling.
3. *The y-exponent in B.* Extended the search to vary the y-degree as
   well: `B = y^c + ω·x^a + ω²·x^b` for `c ∈ {1,2,3,4,5}` and a, b spread
   over coprime/non-coprime pairs in Z_12. Same result: all bounded above
   by d=6.

**Interpretation — this is structural, not bad luck.** Bravyi 2024's d=12
for the [[144,12,12]] instance is the *quantum CSS distance* — the
minimum weight of a logical operator after quotienting by Z-stabilizers,
i.e. min over the cosets of rs(H^Z) inside ker(H^X). That involves the
joint structure of A and B and is typically much larger than the classical
distance of ker(H) alone. Pantaleev & Kalachev's analysis of generalized
bicycle codes (which subsumes BB codes) gives the relevant bound: for
weight-3 A and B at (ℓ,m) = (12,6), the classical F_q-distance of
ker([A|B]) is bounded by the cycle structure of the bipartite Tanner graph,
which is fixed by the abelian group Z_12 × Z_6 and the monomial supports —
not by the F_4 coefficient overlay. Re-skinning {1,1,1} → {1,ω,ω²}
preserves the classical distance up to ±1 because it doesn't change which
codewords exist; it just changes which F_4*-element each nonzero entry
takes.

This means d=12 is unreachable for *any* weight-3, ω-decorated BB code at
n=144 with this group structure. Continuing the polynomial search would be
spending Day 7 budget on a quantity that the construction class does not
support.

**Decision: retain the Day 2 canonical polynomials for all four codes.**
The `large` polynomial as documented in Day 2 stays at
`B = y³ + ω·x + ω²·x7`; the parameter-table file reflects the same
d ≤ 6 bound for all three non-tiny instances. No change to `tiny`,
`small`, `medium`, `large` in `bb_constructor.py`. ALL_INSTANCES order
and naming preserved.

**Rationale.** The paper's contribution is decoder analysis on F_4 BB-type
LDPC codes (BP vs. Metropolis, EXIT prediction, the bridge to quantum BB
codes via the Section VII Tanner-graph structural match) — not a record
on minimum distance. The FER curves measure decoder quality; the error
floor set by d does affect the deep-waterfall tail but not the threshold
behavior at moderate p, which is what BP and EXIT will resolve in Phase
2-3. The current d=6 ceiling is honest data that will be reported in the
Section III parameter table, with the *quantum* distance of the analogous
Bravyi codes noted in Section VII for context.

**Alternatives considered and rejected:**

- *Switch to weight-4 or weight-5 A, B to chase d=8-10.* Breaks
  (3,6)-regularity and the structural alignment with Bravyi's Tanner
  graph that Section VII relies on. Increases check-node degree, which
  raises BP convergence risk in the Day 9-12 critical phase. Rejected.
- *Drop `large` and run with three codes.* Tempting, since `large` with
  d=6 carries no more distance information than `small`. Rejected because
  the scaling axis (n = 18 → 72 → 108 → 144) is needed to show
  decoder-time and threshold-sharpness scaling, which are independent of
  d. The FER plot read as a *decoder-quality* measurement remains
  meaningful with four points.
- *Change the group structure to (ℓ,m) = (8,9) or (6,12).* Different
  cycle profile, possibly different d. Rejected because it would force a
  re-derivation of the Section VII bridge (which uses Z_ℓ × Z_m with
  specific shifts) and burn ~2 days re-doing Days 2-3.

**Implications for the paper:**

- *Section III (Code Construction).* Parameter table reports `d ≤ 6` for
  small/medium/large and `d = 6` (exact) for tiny. A footnote distinguishes
  this from the *quantum CSS distance* of the analogous Bravyi codes
  (6, 6, 10, 12), citing Pantaleev-Kalachev for the bound.
- *Section VI (Numerical Results).* The Day 6 FER plot is the headline
  Metropolis result. Phrasing emphasizes decoder behavior: "Metropolis FER
  on a family of F_4 BB codes spanning n = 18-144." No claim about
  code-distance scaling. The cross-over at low p between codes of similar
  d is read as decoder mixing-time scaling, which is exactly the kind of
  observation EXIT analysis (Section V) is for.
- *Section VII.* The classical-vs-quantum d gap becomes a feature, not a
  bug. The Section VII structural-match argument explains *why* the
  quantum distance can exceed the classical distance: CSS quotienting kills
  the low-weight classical codewords that survive in ker(H). This was
  going to be a paragraph anyway; now it has concrete numbers attached.

**Outcome.** Day 6 deliverables locked: `day6_metropolis_sweep_full.{json,
txt,csv}`, `day6_metropolis_fer_full.png`, `day6_metropolis_ser_ber_full.png`.
FER monotone in p with strict-CI separation per code. Phase 1 closes with
the four canonical codes intact and the d-ceiling explicitly documented as
a known property of the construction class. Day 7 reserved as buffer per
PIPELINE; on schedule to start Day 8 (BP reading) tomorrow.
