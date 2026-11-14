# CONTRACT.md — Standalone [[18, 4, 4]] qLDPC Syndrome Decoder Invariants

Repository: `Pushkar0997/qec-decoder-hackathon-2026`
Target Environment: Python 3.13 clean offline `venv` (`handin/` directory)

## Immutable System Invariants

- **INV-H01 (Zero Banned Packages & Zero External Coupling)**:
  `handin/` MUST NOT import, reference, or vendor any code from `pymatching`, `ldpc`, `stimbposd`, `beliefmatching`, `tesseract-decoder`, `fusion-blossom`, or any external repository. Allowed third-party libraries in `handin/requirements.txt` are strictly limited to `numpy`, `scipy`, `numba`, and `stim`.
- **INV-H02 (Exact Entrypoint Signature)**:
  `handin/decoder.py` MUST expose the top-level function:
  `decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray`
  which accepts a `stim.Circuit` and a 2D array `detector_events` of shape `(N_shots, 70)` (bool or uint8) and returns a 2D array of predicted logical observables of shape `(N_shots, 4)` with `dtype=np.uint8` (or `bool` castable to `{0, 1}`).
- **INV-H03 (Self-Contained `handin/` Isolation)**:
  `handin/` MUST execute cleanly when copied into an isolated temporary directory with no parent directory access (`sys.path` restricted to `handin/`) and no internet access after `pip install -r handin/requirements.txt`.
- **INV-H04 (GF(2) Syndrome Validity Invariant)**:
  For every shot $s \in \{0, \dots, N_{\text{shots}}-1\}$, any candidate error mechanism vector $\hat{e} \in \{0,1\}^M$ produced by BP or OSD MUST satisfy $H \hat{e} \equiv D_s \pmod 2$ across all 70 detectors before projecting onto the 4 logical observables $\hat{O}_s = (L \hat{e}) \bmod 2$.
- **INV-H05 (128-Bit Packed GF(2) Exactness)**:
  Since $N_{\text{det}} = 70 \le 128$, every column of $H$ and every syndrome vector $D_s$ is packed into a pair of unsigned 64-bit integers `(lo_64, hi_64)` where `lo_64` stores detectors `0..63` and `hi_64` stores detectors `64..69`. Bitwise XOR (`^`) on `(lo_64, hi_64)` MUST match `np.bitwise_xor` on unpacked `uint8[70]` vectors with 0 bit discrepancies.
- **INV-H06 (Import-Time JIT Compilation & Zero Cold-Start Penalty)**:
  All `@njit` Numba kernels inside `handin/decoder.py` MUST execute a synthetic 2-shot warmup at module import time so that JIT compilation time is `0.000s` when `decode(circuit, detector_events)` is timed by the evaluation harness.
- **INV-H07 (Runtime Budget Ceiling)**:
  Combined wall-clock execution of `decode(X_circuit, det_X)` (100,000 shots) and `decode(Z_circuit, det_Z)` (100,000 shots) MUST complete in $t \le 12.0\text{ s}$ (strictly below the $15.4\text{ s}$ penalty threshold).
- **INV-H08 (Strict Determinism)**:
  Calling `decode(circuit, detector_events)` twice on identical inputs MUST return bit-for-bit identical output arrays (`np.array_equal(out1, out2) == True`).