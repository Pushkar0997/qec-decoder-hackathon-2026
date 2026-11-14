# Execution Tasks — Standalone Hand-In Pipeline

- [x] **T01**: Author standalone `CONTRACT.md` (`INV-H01`–`INV-H08`) with zero external repo references.
- [x] **T02**: Author `spec/architecture.md` documenting the 4-stage 128-bit LUT + Damped NMS-BP + RREF OSD-CS (Order-2) pipeline.
- [x] **T03**: Implement `handin/decoder.py` exposing `decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray` and `problem_info(circuit: stim.Circuit)`.
- [x] **T04**: Pin `handin/requirements.txt` for Python 3.13 clean `venv` compatibility (`numpy`, `scipy`, `numba`, `stim`).
- [x] **T05**: Implement `scripts/verify_and_stress_test.py` verifying `INV-H01` through `INV-H08` (isolated import check, determinism check, GF(2) RREF correctness, 200k-shot runtime <= 12.0s, and LER report).
- [x] **T06**: Implement `scripts/generate_submission.py` to generate and validate `submission.csv` (`gen_X_00000`..`gen_Z_99999`).