"""Reproducible benchmark & ablation script for handin/decoder.py.

Measures per-observable bit MAE, shot-level LER, wall-clock runtime,
and throughput against the all-zeros baseline on freshly sampled Stim shots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import stim

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "handin"))
from decoder import decode  # noqa: E402


def evaluate_basis(circuit_path: Path, shots: int, seed: int) -> dict[str, float]:
    circuit = stim.Circuit.from_file(str(circuit_path))
    events, flips = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )

    t0 = time.perf_counter()
    preds = decode(circuit, events)
    elapsed = time.perf_counter() - t0

    zero_bit_mae = float(np.mean(flips))
    bit_mae = float(np.mean(preds != flips))
    shot_ler = float(np.mean(np.any(preds != flips, axis=1)))
    return {
        "shots": float(shots),
        "zero_bit_mae": zero_bit_mae,
        "bit_mae": bit_mae,
        "shot_ler": shot_ler,
        "elapsed_s": elapsed,
        "shots_per_sec": shots / max(elapsed, 1e-9),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark handin/decoder.py on Stim shots.")
    parser.add_argument("--shots", type=int, default=20_000, help="Shots per basis (X and Z).")
    parser.add_argument("--seed", type=int, default=1, help="Stim sampler seed.")
    args = parser.parse_args()

    data_dir = REPO_ROOT / "data"
    x_stats = evaluate_basis(data_dir / "X_circuit.stim", args.shots, args.seed)
    z_stats = evaluate_basis(data_dir / "Z_circuit.stim", args.shots, args.seed + 1)

    total_time = x_stats["elapsed_s"] + z_stats["elapsed_s"]
    mean_bit_mae = 0.5 * (x_stats["bit_mae"] + z_stats["bit_mae"])
    mean_shot_ler = 0.5 * (x_stats["shot_ler"] + z_stats["shot_ler"])
    mean_zero_mae = 0.5 * (x_stats["zero_bit_mae"] + z_stats["zero_bit_mae"])

    print(f"Shots per basis : {args.shots:,} (total {2 * args.shots:,})")
    print(f"All-zeros MAE   : {mean_zero_mae:.5f}")
    print(f"X basis         : bit_MAE={x_stats['bit_mae']:.5f}  shot_LER={x_stats['shot_ler']:.5f}  time={x_stats['elapsed_s']:.2f}s ({x_stats['shots_per_sec']:,.0f} shots/s)")
    print(f"Z basis         : bit_MAE={z_stats['bit_mae']:.5f}  shot_LER={z_stats['shot_ler']:.5f}  time={z_stats['elapsed_s']:.2f}s ({z_stats['shots_per_sec']:,.0f} shots/s)")
    print(f"Combined        : bit_MAE={mean_bit_mae:.5f}  shot_LER={mean_shot_ler:.5f}  total_time={total_time:.2f}s")


if __name__ == "__main__":
    main()
