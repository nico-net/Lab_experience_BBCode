"""
day5_benchmark.py — Side-by-side FER for the two Metropolis energies.

Decoders:
  - NishimoriMetropolisDecoder(p):       canonical Sourlas-Nishimori energy.
  - MinCostRepairMetropolisDecoder(p):   user's experimental min-cost-repair energy.
  - null_decoder (baseline):              returns received unchanged.

Test bench: tiny code (n=18, k=10, d=6) at p in {0.01, 0.05, 0.10}.
Trials per cell: 500, seed 20260526. The decoder uses correct knowledge
of the channel p (matched-channel setting).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from bb_constructor import make_tiny
from decoder_metropolis import NishimoriMetropolisDecoder
from decoder_metropolis_v2 import MinCostRepairMetropolisDecoder
from evaluation import evaluate, null_decoder


def run_benchmark():
    code = make_tiny()
    p_values = [0.01, 0.05, 0.10]
    num_trials = 500
    seed = 20260526
    num_sweeps = 200
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    print("=" * 96)
    print(f"Day 5 benchmark — tiny code (n={code.n}, k={code.k}, d=6)")
    print(f"  trials per cell = {num_trials},  seed = {seed},  num_sweeps = {num_sweeps}")
    print(f"  Decoders: null, NishimoriMetropolis(p), MinCostRepairMetropolis(p)")
    print("=" * 96)
    header = (
        f"{'p':>6} | {'decoder':<22} | {'FER':>8}  {'95% CI':>16}  | "
        f"{'SER':>8} | {'BER':>8} | {'sec':>6}"
    )
    print(header)
    print("-" * len(header))

    all_records = []
    for p in p_values:
        nishimori = NishimoriMetropolisDecoder(p=p, num_sweeps=num_sweeps)
        mincost = MinCostRepairMetropolisDecoder(p=p, num_sweeps=num_sweeps)

        # Baseline.
        r_null = evaluate(code, null_decoder, p=p, num_trials=num_trials,
                          seed=seed, decoder_name="null")
        print(
            f"{p:>6.3f} | {'null':<22} | "
            f"{r_null.fer:>8.4f}  [{r_null.fer_ci_low:.3f}, {r_null.fer_ci_high:.3f}]  | "
            f"{r_null.symbol_error_rate:>8.4f} | {r_null.bit_error_rate:>8.4f} | "
            f"{r_null.elapsed_sec:>6.2f}"
        )
        # Nishimori (canonical).
        r_nis = evaluate(code, nishimori, p=p, num_trials=num_trials,
                         seed=seed, decoder_name=nishimori.name)
        print(
            f"{p:>6.3f} | {'nishimori (canonical)':<22} | "
            f"{r_nis.fer:>8.4f}  [{r_nis.fer_ci_low:.3f}, {r_nis.fer_ci_high:.3f}]  | "
            f"{r_nis.symbol_error_rate:>8.4f} | {r_nis.bit_error_rate:>8.4f} | "
            f"{r_nis.elapsed_sec:>6.2f}"
        )
        # MinCostRepair (experimental).
        r_mcr = evaluate(code, mincost, p=p, num_trials=num_trials,
                         seed=seed, decoder_name=mincost.name)
        print(
            f"{p:>6.3f} | {'mincost-repair (v2)':<22} | "
            f"{r_mcr.fer:>8.4f}  [{r_mcr.fer_ci_low:.3f}, {r_mcr.fer_ci_high:.3f}]  | "
            f"{r_mcr.symbol_error_rate:>8.4f} | {r_mcr.bit_error_rate:>8.4f} | "
            f"{r_mcr.elapsed_sec:>6.2f}"
        )
        print()

        all_records.append({
            "p": p,
            "null": asdict(r_null),
            "nishimori": asdict(r_nis),
            "mincost_repair": asdict(r_mcr),
        })

    # Monotonicity checks (PIPELINE Day 5 validation).
    print("-" * len(header))
    print("Monotonicity (PIPELINE Day 5 validation criterion):")
    for name, key in [
        ("null", "null"),
        ("nishimori", "nishimori"),
        ("mincost-repair", "mincost_repair"),
    ]:
        fers = [rec[key]["fer"] for rec in all_records]
        monotonic = all(fers[i] <= fers[i + 1] for i in range(len(fers) - 1))
        ci_strict = all(
            all_records[i][key]["fer_ci_high"] < all_records[i + 1][key]["fer_ci_low"]
            for i in range(len(all_records) - 1)
        )
        marker = "PASS" if monotonic else "FAIL"
        print(f"  {name:<18}  FER = {fers}  →  monotonic: {marker}  "
              f"(strict CI: {'yes' if ci_strict else 'no'})")

    (out_dir / "day5_benchmark.json").write_text(
        json.dumps(all_records, indent=2) + "\n"
    )
    print(f"\nResults written to {out_dir}/day5_benchmark.json")
    return all_records


if __name__ == "__main__":
    t0 = time.perf_counter()
    run_benchmark()
    print(f"\nTotal elapsed: {time.perf_counter() - t0:.2f}s")
