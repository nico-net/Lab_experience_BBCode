# Code Parameters [n, k, d]_4

Day 3 deliverable for the GF(4) BB-code paper. All values are over F_4.
Distance for the `tiny` code is exact via brute-force codeword enumeration; larger codes report an upper bound from a combined low-weight information-set search (weights 1–3) and random message sampling (200,000 messages, seed 20260523).

| Instance | (ℓ, m) | n | rank(H) | k | rate | row wt | col wt | d (F_4) | method |
|---|---|---|---|---|---|---|---|---|---|
| tiny | (3, 3) | 18 | 8 | 10 | 0.5556 | 6 | 3 | 6 | exact |
| small | (6, 6) | 72 | 30 | 42 | 0.5833 | 6 | 3 | ≤ 6 | upper_bound |
| medium | (9, 6) | 108 | 50 | 58 | 0.5370 | 6 | 3 | ≤ 6 | upper_bound |
| large | (12, 6) | 144 | 70 | 74 | 0.5139 | 6 | 3 | ≤ 6 | upper_bound |

## Diagnostics

| Instance | basis min wt | weight-1/2/3 search | random search | trials | elapsed (s) |
|---|---|---|---|---|---|
| tiny | 6 | — | — | — | 1.20 |
| small | 6 | 6 | 37 | 200,000 | 5.32 |
| medium | 6 | 6 | 59 | 200,000 | 20.04 |
| large | 6 | 6 | 85 | 200,000 | 122.43 |

