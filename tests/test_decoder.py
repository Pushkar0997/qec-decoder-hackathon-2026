"""Unit tests for handin/decoder.py against CONTRACT.md invariants."""

import ast
from pathlib import Path
import sys
import unittest

import numpy as np
import stim

REPO_ROOT = Path(__file__).resolve().parent.parent
HANDIN_DIR = REPO_ROOT / "handin"
DATA_DIR = REPO_ROOT / "data"

sys.path.insert(0, str(HANDIN_DIR))
from decoder import decode  # noqa: E402


BANNED_PACKAGES = {
    "pymatching",
    "ldpc",
    "stimbposd",
    "beliefmatching",
    "tesseract_decoder",
    "tesseract-decoder",
    "fusion_blossom",
    "fusion-blossom",
    "qecscreen",
}


class TestDecoderContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.x_circuit = stim.Circuit.from_file(str(DATA_DIR / "X_circuit.stim"))
        cls.z_circuit = stim.Circuit.from_file(str(DATA_DIR / "Z_circuit.stim"))

    def test_no_banned_imports_in_handin(self) -> None:
        for py_file in HANDIN_DIR.glob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        self.assertNotIn(top, BANNED_PACKAGES, f"Banned import {top} in {py_file.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    top = node.module.split(".")[0]
                    self.assertNotIn(top, BANNED_PACKAGES, f"Banned import {top} in {py_file.name}")

    def test_api_shape_dtype_and_values(self) -> None:
        events = np.zeros((16, 70), dtype=bool)
        events[1, 0] = True
        events[2, 17] = True
        preds = decode(self.x_circuit, events)
        self.assertEqual(preds.shape, (16, 4))
        self.assertEqual(preds.dtype, np.uint8)
        self.assertTrue(np.isin(preds, (0, 1)).all())

    def test_zero_syndrome_returns_all_zeros(self) -> None:
        events = np.zeros((32, 70), dtype=bool)
        for circuit in (self.x_circuit, self.z_circuit):
            preds = decode(circuit, events)
            np.testing.assert_array_equal(preds, np.zeros((32, 4), dtype=np.uint8))

    def test_bitwise_determinism(self) -> None:
        events, _ = self.x_circuit.compile_detector_sampler(seed=42).sample(
            512, separate_observables=True
        )
        out1 = decode(self.x_circuit, events)
        out2 = decode(self.x_circuit, events)
        np.testing.assert_array_equal(out1, out2)

    def test_beats_zero_baseline_on_sampled_shots(self) -> None:
        for circuit in (self.x_circuit, self.z_circuit):
            events, flips = circuit.compile_detector_sampler(seed=7).sample(
                5_000, separate_observables=True
            )
            preds = decode(circuit, events)
            bit_mae = float(np.mean(preds != flips))
            zero_mae = float(np.mean(flips))
            self.assertLess(bit_mae, zero_mae * 0.4, f"bit_mae={bit_mae:.5f} vs zero={zero_mae:.5f}")


if __name__ == "__main__":
    unittest.main()
