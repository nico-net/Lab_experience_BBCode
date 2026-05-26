# 4-Week Execution Pipeline

**Timeline:** 30 days from start to submission (firm)  
**Current Phase:** [UPDATE DAILY]  
**Last Updated:** [DATE]

---

## Phase 0: Lock Decisions (Day 0, ~2 hours)

**Status:** [ ] Complete

**Tasks:**
- [x] Create Claude Project
- [x] Upload PROJECT_BRIEF.md, PIPELINE.md, DECISIONS_LOG.md
- [x] Upload reference papers (BB_code_DNA__4_.pdf, TR2001-16.pdf)
- [x] Send email to professors (framing, venue, timeline)
- [ ] Receive professor approval
- [ ] Lock framing: Analog 3 + Section VII
- [ ] Lock venue: IEEE COMML + arXiv
- [x] Set git repository

**Deliverable:** Written approval from professors, git repo initialized

**Blockers:** Waiting for professor email response (3-5 business days expected)

---

## Phase 1: Code Construction (Days 1-7)

**Goal:** Working evaluation pipeline, clean Metropolis curves for all code sizes

### Day 1 (3 hours): GF(4) Library

**Status:** [x] Complete

**Tasks:**
- [x] Python environment setup (numpy, scipy, matplotlib, tqdm)
- [x] GF(4) arithmetic library:
  - [x] Addition table (XOR of 2-bit representations)
  - [x] Multiplication table (F_4 = F_2[ω]/⟨ω²+ω+1⟩)
  - [x] Multiplicative inverse, trace function
  - [x] Unit tests against known values
- [x] Upload gf4_lib.py to Claude Project for review
- [x] Log polynomial representation choice in DECISIONS_LOG.md

**Deliverable:** gf4_lib.py passing all unit tests

**Validation:** `assert gf4_mult(ω, ω) == gf4_add(ω, 1)` (since ω² = ω + 1)

---

### Day 2 (3 hours): BB Code Constructor

**Status:** [x] Complete

**Tasks:**
- [x] Polynomial ring arithmetic in F_4[x,y]/⟨x^ℓ-1, y^m-1⟩
- [x] Matrix representation of polynomials (circulant-like structure)
- [x] BB code constructor: input (A,B), output H = [A|B]
- [x] Choose 4 specific polynomial pairs for (ℓ,m) instances
- [x] Verify H sparsity structure matches expected pattern
- [x] Upload bb_constructor.py to Project

**Deliverable:** bb_constructor.py with 4 working code instances

**Log in DECISIONS_LOG:** Which polynomials chosen for each (ℓ,m) and why

---

### Day 3 (3 hours): Code Parameters

**Status:** [x] Complete

**Tasks:**
- [x] Compute n, k for each code instance (via rank over F_4)
- [x] For smallest code: compute d via brute-force codeword enumeration
- [x] For larger codes: upper bound on d via random search or leave unknown
- [x] Create parameter table [n, k, d]_4 for all instances
- [x] Verify no trivial codewords (all-zeros only)

**Deliverable:** Parameter table ready for paper Section III

**Validation:** k should be ≈ n/2 for most instances (sanity check)

---

### Day 4 (3 hours): Channel and Evaluation Framework

**Status:** [x]  Complete

**Tasks:**
- [x]  Quaternary symmetric channel: i.i.d. GF(4) errors with rate p
- [x]  evaluate(code, decoder, p, num_trials) function:
  - [x]  Returns FER, BER, error bars (95% CI)
  - [x]  Logs trial count and random seed for reproducibility
- [x]  Reproducible random seeding
- [x]  Upload channel.py and evaluation.py to Project

**Deliverable:** Evaluation framework ready for decoder benchmarking

---

### Day 5 (3 hours): Refactor Metropolis Decoder

**Status:** [x] Complete

**Tasks:**
- [x] Wrap existing simulation code in new framework
- [x] Interface: decoder(code, received_word, T, num_sweeps) → estimated_codeword
- [x] Cost function: H(x) = J · |{violated checks}|
- [x] Temperature schedule tuning (if needed)
- [x] Upload decoder_metropolis.py to Project

**Deliverable:** Metropolis decoder callable via evaluate()

**Validation:** Run on smallest code at p=0.01, 0.05, 0.10; verify FER monotonic

---

### Days 6-7: First Results + Buffer

**Status:** [x] Complete

**Tasks:**
- [x] Generate FER vs. p curves for Metropolis on all code sizes
- [x] 1000+ trials per data point
- [x] Save raw data (not just plots)
- [x] Create first draft of Figure 2 (FER curves)
- [x] Debug any unexpected behaviors

**Deliverable:** End-of-week checkpoint plot: Metropolis FER curves

**GO/NO-GO:** Do curves look sensible? If yes → Phase 2. If no → debug before proceeding.

---

## Phase 2: BP Implementation (Days 8-14)

**Goal:** Working sum-product BP over GF(4), validated against ML on small codes

**HIGHEST RISK PHASE:** BP bugs are subtle and time-consuming

### Day 8 (3 hours): Reading and Planning

**Status:** [x] Complete

**Tasks:**
- [x] Re-read course notes on binary BP (refresh fundamentals)
- [x] Read Davey & MacKay 1998 Section III (non-binary BP algorithm)
- [x] Skim one quantum LDPC BP paper (Pryadko/Roffe) for Section VII context
- [x] Write BP pseudocode before any implementation
- [x] Upload pseudocode to Project for review

**Deliverable:** BP pseudocode reviewed and approved

**Time budget:** 2h reading, 1h pseudocode

---

### Days 9-10 (6 hours): Implement BP

**Status:** [x] Complete

**Tasks:**
- [x] Message structure: length-4 probability vectors over F_4
- [x] Variable node update: pointwise product + normalization
- [x] Check node update (THE HARD PART):
  - [x] Edge weight permutation: m(x) → m(h^{-1}·x)
  - [x] Convolution over F_4 via FHT or direct method
  - [x] Inverse permutation on outgoing message
- [x] Damping (optional but recommended for stability)
- [x] Convergence check: message change threshold or max iterations
- [x] Upload decoder_bp.py to Project

**Deliverable:** BP implementation complete (not yet validated)

**Common bugs to watch:**
- Message normalization (must sum to 1 after each update)
- Edge weight permutation direction (h^{-1} not h)
- FHT sign conventions

---

### Day 11 (3 hours): ML Decoder for Validation

**Status:** [x] Complete

**Tasks:**
- [x] Brute-force ML decoder for codes with 4^k ≤ 2^14 (k ≤ 7)
- [x]Enumerate all codewords, find max likelihood given received word
- [x] Test on smallest code at p = 0.001, 0.01
- [x] Upload decoder_ml.py to Project

**Deliverable:** Ground-truth ML decoder for BP validation

**Validation:** ML should achieve very low FER at p=0.001

---

### Day 12 (3 hours): BP Validation — CRITICAL CHECKPOINT

**Status:** [x] Complete

**Tasks:**
- [x] Run BP and ML on same trials (smallest code, p = 0.001, 0.005, 0.01, 0.05)
- [x] Measure agreement: % of trials where BP decision = ML decision
- [x] REQUIRED: At p=0.001, agreement >95%
- [x] REQUIRED: At p=0.01, agreement >80%
- [x] If validation fails: upload BP code to Project and request debug help

**GO/NO-GO DECISION:**
- **If validation passes:** Proceed to Days 13-14
- **If validation fails:** STOP, debug, do not proceed to larger codes

**Deliverable:** Validated BP decoder with documented agreement metrics

---

### Days 13-14 (6 hours): Full BP Performance Curves

**Status:** [ ] Complete

**Tasks:**
- [ ] Run BP on all code sizes, p ∈ [0.001, 0.3] on log scale
- [ ] 1000+ trials per point
- [ ] Generate FER curves for both BP and Metropolis on same plot
- [ ] Save raw data
- [ ] Create Figure 2 (final version): FER comparison

**Deliverable:** End-of-Phase-2 plot: BP vs. Metropolis FER curves, all code sizes

---

## Phase 3: EXIT and Scaling (Days 15-21)

**Goal:** Theoretical threshold prediction, finite-size scaling, metastability analysis

### Day 15 (3 hours): EXIT Generalization to GF(4)

**Status:** [ ] Complete

**Tasks:**
- [ ] Review course notes on binary EXIT charts
- [ ] Implement mutual information for GF(4) symbols and length-4 message vectors
- [ ] Variable node EXIT function: I_E^V(I_A, p)
- [ ] Check node EXIT function: I_E^C(I_A)
- [ ] Upload exit_analysis.py to Project

**Deliverable:** EXIT function computation working

---

### Day 16 (3 hours): Threshold from EXIT

**Status:** [ ] Complete

**Tasks:**
- [ ] Find p* where EXIT curves just touch (BP threshold prediction)
- [ ] Plot EXIT curves at p slightly below and above p*
- [ ] Verify tunnel opens/closes as expected
- [ ] Create Figure 3: EXIT chart with threshold

**Deliverable:** Predicted BP threshold p* from EXIT analysis

---

### Day 17 (3 hours): EXIT vs. Simulation Validation

**Status:** [ ] Complete

**Tasks:**
- [ ] Extract simulated threshold p_c^{sim} from FER waterfall (curve crossing)
- [ ] Compare to EXIT prediction p*
- [ ] Agreement within ~20-30%: PASS
- [ ] Disagreement >30%: investigate density evolution assumptions
- [ ] Document comparison in DECISIONS_LOG

**Deliverable:** Validated EXIT threshold vs. simulation

---

### Day 18 (3 hours): Finite-Size Scaling for Metropolis

**Status:** [ ] Complete

**Tasks:**
- [ ] Identify Metropolis threshold p_c^{Metro} via curve crossing
- [ ] Optional: attempt data collapse P_fail(p,L) = f((p - p_c)L^{1/ν})
- [ ] Extract critical exponent ν if data collapse works
- [ ] Compare Metropolis threshold to BP threshold

**Deliverable:** Metropolis threshold characterized

---

### Days 19-20 (6 hours): Metastability and Defect Confinement

**Status:** [ ] Complete

**Tasks:**
- [ ] Metastability: trapping time distributions, temperature dependence
- [ ] Frame as "trapping sets / error floors" (channel coding language)
- [ ] Defect confinement: correlation length ξ vs. p
- [ ] Frame as "error pattern containment"
- [ ] Generate publication-quality figures
- [ ] Create Figure 4: Metastability or Figure 5: Defect confinement

**Deliverable:** Decoder characterization figures ready

---

### Day 21 (3 hours): Buffer and CRSS Verification

**Status:** [ ] Complete

**Tasks:**
- [ ] Skim Yedidia-Freeman-Weiss abstract and intro (20-30 min)
- [ ] Verify Tanner graph of Analog 3 structurally matches CRSS additive code
- [ ] Document structural match in DECISIONS_LOG (needed for Section VII)
- [ ] Catch up on any delayed tasks from Days 15-20

**Deliverable:** All numerical results complete, ready to write

---

## Phase 4: Writing (Days 22-28)

**Goal:** Submittable paper draft, section by section

### Day 22 (4 hours): Outline and Figure Plan

**Status:** [ ] Complete

**Tasks:**
- [ ] Lock section structure:
  - I. Introduction
  - II. Preliminaries (GF(4), LDPC, channel)
  - III. Code Construction
  - IV. Decoders (BP and Metropolis)
  - V. EXIT Analysis
  - VI. Numerical Results
  - VII. Connection to Quantum BB Codes
  - VIII. Discussion and Future Work
- [ ] Plan 4-5 figures, finalize figure order
- [ ] Start LaTeX document structure

**Deliverable:** Paper skeleton with empty sections

---

### Day 23 (4 hours): Preliminaries and Construction

**Status:** [ ] Complete

**Tasks:**
- [ ] Write Section II (Preliminaries): GF(4), BB codes, channel
- [ ] Write Section III (Code Construction): polynomials, parameter table
- [ ] Upload draft to Project for review

**Deliverable:** Sections II-III complete

---

### Day 24 (4 hours): Decoders Section

**Status:** [ ] Complete

**Tasks:**
- [ ] Write Section IV (Decoders):
  - BP algorithm with pseudocode
  - Metropolis algorithm with pseudocode
  - BP-as-Bethe, Metropolis-as-Boltzmann framing [YFW, Sourlas, Nishimori]
  - Complexity analysis
- [ ] Upload to Project for review

**Deliverable:** Section IV complete

---

### Day 25 (4 hours): EXIT and Results

**Status:** [ ] Complete

**Tasks:**
- [ ] Write Section V (EXIT Analysis): EXIT functions, threshold prediction
- [ ] Write Section VI (Numerical Results): FER curves, scaling, metastability
- [ ] Insert all figures in final positions
- [ ] Upload to Project for review

**Deliverable:** Sections V-VI complete with figures

---

### Day 26 (3 hours): Introduction, Section VII, Discussion

**Status:** [ ] Complete

**Tasks:**
- [ ] Write Section I (Introduction): motivation, contributions, structure
- [ ] Write Section VII (Quantum Connection): use SECTION_VII_DRAFT.md as template
- [ ] Write Section VIII (Discussion): trade-offs, applications, future work
- [ ] Write abstract (last)
- [ ] Upload full draft to Project for review

**Deliverable:** Complete first draft

---

### Day 27 (3 hours): Internal Revision

**Status:** [ ] Complete

**Tasks:**
- [ ] Read paper end-to-end, fix flow issues
- [ ] Verify every claim has citation or result support
- [ ] Verify every figure referenced in text
- [ ] Check contribution statement matches deliverables
- [ ] Length check (trim if >4 pages for Letter format)
- [ ] Citation check: Bravyi, CRSS, YFW, Sourlas, Nishimori, Davey-MacKay all cited?

**Deliverable:** Revised draft ready for final polish

---

### Day 28 (3 hours): Final Polish and Submission Prep

**Status:** [ ] Complete

**Tasks:**
- [ ] Spell check, grammar check
- [ ] Format for IEEE Communications Letters (IEEEtran.cls)
- [ ] Prepare arXiv submission (possibly with extended appendix)
- [ ] Write cover letter for journal
- [ ] Send draft to professors for final review
- [ ] Upload camera-ready versions to Project

**Deliverable:** Submission-ready paper

---

## Phase 5: Submission (Days 29-30)

### Day 29: arXiv Submission

**Status:** [ ] Complete

**Tasks:**
- [ ] Submit to arXiv (cs.IT primary, possibly cond-mat.stat-mech secondary)
- [ ] Wait for moderation (~24 hours)
- [ ] Record arXiv ID in PROJECT_BRIEF

**Deliverable:** arXiv preprint live, citable DOI obtained

---

### Day 30: IEEE Communications Letters Submission

**Status:** [ ] Complete

**Tasks:**
- [ ] Format strictly per IEEE guidelines
- [ ] Write cover letter (suggest 3-5 reviewers)
- [ ] Submit via IEEE submission portal
- [ ] Record submission ID and date

**Deliverable:** Paper submitted, waiting for reviews

---

## Ongoing Tasks (Daily, ~30 min)

- [ ] Lab notebook: log what worked, what failed, why
- [ ] Git commit: at least one per work session
- [ ] Backup: push to GitHub/GitLab
- [ ] Weekly check-in: status update in Claude Project

---

## Emergency Contacts and Fallback Plans

**If BP validation fails on Day 12:**
- Upload decoder_bp.py to Claude Project immediately
- Request targeted debugging help
- Fallback: Metropolis-only paper (weaker but publishable)

**If >2 days behind schedule:**
- Identify lowest-priority deliverable and defer to future work
- Candidates: data collapse, metastability, EXIT analysis

**If professor feedback requires major changes:**
- Assess impact on timeline
- Negotiate deadline extension if needed
- Update PROJECT_BRIEF with modified scope

---

**Last Pipeline Update:** [DATE]