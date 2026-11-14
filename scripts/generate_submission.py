"""
Generate `submission.csv` for the [[18, 4, 4]] qLDPC decoding challenge.

Reads `data/test.csv` + the two Stim circuits, decodes each basis separately
through the standalone `handin/decoder.py` entrypoint, then reassembles the
predictions in the exact original row order of `test.csv`.

Run:  python scripts/generate_submission.py

Note: `pandas` is a *development-only* convenience dependency used by this
script. It is intentionally NOT part of `handin/requirements.txt`, so the
hand-in package remains numpy/scipy/numba/stim only (INV-H01).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import stim

REPO_ROOT = Path(__file__).resolve().parent.parent
HANDIN_DIR = REPO_ROOT / "handin"
DATA_DIR = REPO_ROOT / "data"
TEST_CSV = DATA_DIR / "test.csv"
X_STIM = DATA_DIR / "X_circuit.stim"
Z_STIM = DATA_DIR / "Z_circuit.stim"
SUBMISSION_CSV = REPO_ROOT / "submission.csv"

sys.path.insert(0, str(HANDIN_DIR))

import decoder  # pyright: ignore[reportMissingImports]

DETECTOR_COLS = [f"D{i}" for i in range(70)]
LABEL_COLS = ["L0", "L1", "L2", "L3"]
EXPECTED_ROW_COUNT = 200_000


def _load_test_frame() -> pd.DataFrame:
    """Loads test.csv, validating the exact competition schema."""
    test = pd.read_csv(TEST_CSV, dtype={"id": str, "basis": str})
    expected_columns = ["id", "basis", *DETECTOR_COLS]
    assert list(test.columns) == expected_columns, (
        f"Unexpected test.csv columns: {list(test.columns)[:5]}..."
    )
    assert len(test) == EXPECTED_ROW_COUNT, f"Expected {EXPECTED_ROW_COUNT} rows, got {len(test)}"
    assert not test.isna().any().any(), "test.csv contains NaN/null values"
    bases = set(test["basis"].unique())
    assert bases == {"X", "Z"}, f"Unexpected basis values: {bases}"
    return test


def _detector_bits(test: pd.DataFrame, row_idx: np.ndarray) -> np.ndarray:
    """Extracts the (len(row_idx), 70) detector-event block as a C-contiguous bool array."""
    block = test.iloc[row_idx][DETECTOR_COLS].to_numpy(dtype=np.uint8, copy=True)
    events = np.ascontiguousarray(block, dtype=np.bool_)
    assert events.shape == (row_idx.size, 70), f"Bad detector block shape {events.shape}"
    return events


def _decode_basis(
    label: str,
    circuit_path: Path,
    events: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Decodes one basis block; returns (predictions uint8, elapsed seconds)."""
    circuit = stim.Circuit.from_file(str(circuit_path))
    decoder._CIRCUIT_CACHE.clear()  # force a cold, honest timing of the decode call

    t0 = time.perf_counter()
    predictions = decoder.decode(circuit, events)
    elapsed = time.perf_counter() - t0

    assert predictions.shape == (events.shape[0], 4), (
        f"{label}: expected ({events.shape[0]}, 4), got {predictions.shape}"
    )
    predictions = np.ascontiguousarray(predictions, dtype=np.uint8)
    assert predictions.dtype == np.uint8, f"{label}: predictions must be uint8"

    flip_rate = float(np.mean(predictions.any(axis=1)))
    per_observable = predictions.mean(axis=0)
    print(
        f"[DECODE {label}] {events.shape[0]:,} shots in {elapsed:.3f}s | "
        f"any-flip rate={flip_rate:.6f} | "
        f"per-observable={np.array2string(per_observable, precision=6)}"
    )
    return predictions, elapsed


def main() -> None:
    wall_start = time.perf_counter()

    test = _load_test_frame()
    basis = test["basis"].to_numpy()
    x_rows = np.flatnonzero(basis == "X")
    z_rows = np.flatnonzero(basis == "Z")
    assert x_rows.size + z_rows.size == EXPECTED_ROW_COUNT, "Basis split does not partition rows"
    print(
        f"[DATA] test.csv rows={len(test):,} | "
        f"X={x_rows.size:,} | Z={z_rows.size:,}"
    )

    x_events = _detector_bits(test, x_rows)
    z_events = _detector_bits(test, z_rows)

    x_preds, x_seconds = _decode_basis("X", X_STIM, x_events)
    z_preds, z_seconds = _decode_basis("Z", Z_STIM, z_events)

    # Reassemble in the exact original test.csv row order.
    stacked = np.empty((len(test), 4), dtype=np.uint8)
    stacked[x_rows] = x_preds
    stacked[z_rows] = z_preds

    sub = pd.DataFrame(
        {
            "id": test["id"].to_numpy(),
            "L0": stacked[:, 0],
            "L1": stacked[:, 1],
            "L2": stacked[:, 2],
            "L3": stacked[:, 3],
        }
    )

    # ---- Strict pre-save validation gates ----
    assert len(sub) == len(test) == EXPECTED_ROW_COUNT
    assert (sub["id"].to_numpy() == test["id"].to_numpy()).all(), "Row order not preserved!"
    assert list(sub.columns) == ["id", "L0", "L1", "L2", "L3"]

    labels = sub[LABEL_COLS].to_numpy()
    assert labels.dtype == np.uint8, "Labels must be uint8"
    assert np.isin(labels, (0, 1)).all(), "Labels contain values outside {0, 1}"
    assert not sub.isna().to_numpy().any(), "submission contains NaN/null values"
    assert not bool(sub["id"].isna().any()), "id column contains NaN/null values"

    sub.to_csv(SUBMISSION_CSV, index=False)

    decode_seconds = x_seconds + z_seconds
    total_seconds = time.perf_counter() - wall_start
    print(
        f"[TIMING] decode(X)={x_seconds:.3f}s | decode(Z)={z_seconds:.3f}s | "
        f"combined decode={decode_seconds:.3f}s | end-to-end={total_seconds:.3f}s"
    )
    print(f"[WRITE] {SUBMISSION_CSV} ({len(sub):,} data rows + header)")


if __name__ == "__main__":
    main()
