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

