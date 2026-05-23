"""
day4_validation.py — End-to-end Day 4 validation sweep.

Runs `null_decoder` across all four code instances at a few error rates and
checks the empirical FER against the analytical value 1 - (1-p)^n. Saves
results/null_sweep.json + results/null_sweep.txt so we have a Day-4 record
the harness was working before any non-trivial decoder was attached.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from bb_constructor import ALL_INSTANCES
from evaluation import evaluate, null_decoder


def main() -> None:
    p_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    num_trials = 5_000
    seed = 20260524
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    all_results = []
    print(f"{'code':>7} {'n':>4} {'p':>8} {'FER_emp':>10} {'FER_ana':>10} "
          f"{'|Δ|/σ':>7} {'SER':>10} {'BER_bit':>10} {'trials':>7}")
    print("-" * 88)

    for make in ALL_INSTANCES:
        code = make()
        for p in p_values:
            r = evaluate(code, null_decoder, p=p, num_trials=num_trials,
                         seed=seed, decoder_name="null")
            fer_ana = 1.0 - (1.0 - p) ** code.n
            sigma = math.sqrt(fer_ana * (1 - fer_ana) / r.num_trials) or 1e-12
            z = abs(r.fer - fer_ana) / sigma
            all_results.append(asdict(r))
            print(f"{code.name:>7} {code.n:>4} {p:>8.4f} {r.fer:>10.4f} "
                  f"{fer_ana:>10.4f} {z:>7.2f} "
                  f"{r.symbol_error_rate:>10.4f} {r.bit_error_rate:>10.4f} "
                  f"{r.num_trials:>7}")

    (out_dir / "null_sweep.json").write_text(
        json.dumps(all_results, indent=2) + "\n", encoding="utf-8"
    )
    summary_lines = [
        "Day 4 null-decoder sweep — empirical FER vs analytical 1-(1-p)^n",
        f"  trials per cell: {num_trials},  seed: {seed}",
        "  |Δ|/σ < 4 expected for all cells (false-positive rate < 1/15000).",
        "",
    ]
    (out_dir / "null_sweep.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()
