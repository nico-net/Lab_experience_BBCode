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
