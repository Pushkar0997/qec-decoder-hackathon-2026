"""
Validate the hand-in decoder against the labelled `data/train.csv`.

Splits train.csv by basis (preserving row alignment), decodes each block through
`handin/decoder.py`, and reports the exact train shot/bit LER plus runtime.

Run:  python scripts/validate_train.py [--max-rows N]

`pandas` is a development-only dependency (never imported by `handin/`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import stim

REPO_ROOT = Path(__file__).resolve().parent.parent
HANDIN_DIR = REPO_ROOT / "handin"
DATA_DIR = REPO_ROOT / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
X_STIM = DATA_DIR / "X_circuit.stim"
Z_STIM = DATA_DIR / "Z_circuit.stim"

sys.path.insert(0, str(HANDIN_DIR))

import decoder  # pyright: ignore[reportMissingImports]

DETECTOR_COLS = [f"D{i}" for i in range(70)]
LABEL_COLS = ["L0", "L1", "L2", "L3"]


def _load_train(max_rows: int | None) -> pd.DataFrame:
    dtypes: dict[str, str] = {"id": "str", "basis": "str"}
    dtypes.update({column: "int8" for column in DETECTOR_COLS + LABEL_COLS})
    frame = pd.read_csv(TRAIN_CSV, dtype=dtypes, nrows=max_rows)  # pyright: ignore[reportCallIssue, reportArgumentType]
    assert not frame.isna().any().any(), "train.csv contains NaN/null values"
    assert set(frame["basis"].unique()) == {"X", "Z"}
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows", type=int, default=None, help="Optionally subsample train.csv rows")
    args = parser.parse_args()

    wall_start = time.perf_counter()
    train = _load_train(args.max_rows)
    basis = train["basis"].to_numpy()
    x_rows = np.flatnonzero(basis == "X")
    z_rows = np.flatnonzero(basis == "Z")
    print(f"[DATA] rows={len(train):,} | X={x_rows.size:,} | Z={z_rows.size:,}")

    preds = np.empty((len(train), 4), dtype=np.uint8)
    total_decode = 0.0
    for label, rows, stim_path in (("X", x_rows, X_STIM), ("Z", z_rows, Z_STIM)):
        events = np.ascontiguousarray(
            train.iloc[rows][DETECTOR_COLS].to_numpy(dtype=np.uint8), dtype=np.bool_
        )
        circuit = stim.Circuit.from_file(str(stim_path))
        decoder._CIRCUIT_CACHE.clear()

        t0 = time.perf_counter()
        block = decoder.decode(circuit, events)
        elapsed = time.perf_counter() - t0
        total_decode += elapsed
        preds[rows] = block

        truth = train.iloc[rows][LABEL_COLS].to_numpy(dtype=np.uint8)
        shot_ler = float(np.mean(np.any(block != truth, axis=1)))
        bit_ler = float(np.mean(block != truth))
        print(
            f"[TRAIN {label}] {events.shape[0]:,} shots in {elapsed:.3f}s | "
            f"Shot LER={shot_ler:.6f} | Bit LER={bit_ler:.6f}"
        )

    truth_all = train[LABEL_COLS].to_numpy(dtype=np.uint8)
    shot_ler = float(np.mean(np.any(preds != truth_all, axis=1)))
    bit_ler = float(np.mean(preds != truth_all))
    print(
        f"=== TRAIN TOTAL: {len(train):,} shots | Shot LER={shot_ler:.6f} | Bit LER={bit_ler:.6f} "
        f"| decode={total_decode:.3f}s | end-to-end={time.perf_counter() - wall_start:.3f}s ==="
    )


if __name__ == "__main__":
    main()
