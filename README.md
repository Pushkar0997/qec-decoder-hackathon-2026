# QEC Decoder Hackathon 2026 — `[[18, 4, 4]]` qLDPC Syndrome Decoder

A fast, deterministic, **from-scratch** quantum error correction (QEC) decoder built for the [Quantum Error Correction: Decoder Hackathon](https://www.kaggle.com/competitions/quantum-error-correction-decoder) (November 2026), hosted by the Niels Bohr Institute / NQCP (University of Copenhagen), NVIDIA, Riverlane, DTU Physics, QMATH, and DeiC.

> **Benchmark Highlight:** Achieved **`0.04446` Public / `0.04483` Private** per-observable Logical Error Rate (MAE) on Kaggle (Late Submission) — a **~6.5× error reduction** compared to the `~0.29` all-zeros baseline, written completely without external QEC decoder libraries (`pymatching`, `ldpc`, `stimbposd`, etc.).

---

## 1. What Is This Problem? (No PhD Required)

Quantum computers are notoriously noisy: every gate, reset, or measurement has a small chance of flipping a qubit's state. To protect quantum information, we encode **4 logical qubits** (the data we actually care about) across **18 physical data qubits** plus **14 helper ("check") qubits** — a `[[18, 4, 4]]` quantum Low-Density Parity-Check (**qLDPC**) code.

Think of it like a Sudoku puzzle with parity checks:
1. You aren't allowed to look directly at the 4 logical qubits while a computation runs (that would destroy the quantum state).
2. Instead, over **5 rounds**, the 14 check qubits report **70 binary "detector" alarms** (`D0`–`D69`): `0` means everything looks consistent, `1` means an error happened nearby in space or time.
3. **The Decoder's Job:** Given only those 70 alarm bits, deduce which of the **4 logical qubits** (`L0`–`L3`) got flipped by the end of the experiment — and do it accurately and fast across **200,000 test runs** (`100,000` X-basis + `100,000` Z-basis).

---

## 2. How Our Decoder Works

Because competition rules strictly forbid importing pre-built QEC decoders (`pymatching`, `ldpc`, `stimbposd`, `beliefmatching`, `tesseract-decoder`, `fusion-blossom`), every component in [`handin/decoder.py`](handin/decoder.py) is implemented from scratch using `numpy`, `scipy`, `numba`, and `stim`:

```text
  70-bit Detector Syndrome (s)
             │
             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Stim Detector Error Model (DEM) Extraction           │
  │    Maps circuit noise into H (70 × M), L (4 × M), & p   │
  └──────────┬──────────────────────────────────────────────┘
             │
             ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 2. Numba-Accelerated float32 Belief Propagation (BP)    │
  │    5-iteration message passing with per-thread          │
  │    reusable workspaces (zero per-shot allocations)      │
  └──────────┬────────────────────────┬─────────────────────┘
             │ Converged (H·ê = s)    │ Non-converged
             ▼                        ▼
      Compute L·ê (mod 2)   ┌───────────────────────────────┐
                            │ 3. Ordered Statistics Decoding │
                            │    GF(2) Gaussian elimination │
                            │    ordered by BP reliability  │
                            └─────────┬─────────────────────┘
                                      │
                                      ▼
                            4-bit Prediction (L0..L3)
```

### Key Engineering Decisions
* **Zero Per-Shot Memory Allocations:** Uses `float32` BP messages/LLRs and pre-allocated per-thread workspaces inside `@njit` kernels, plus **10,000-shot chunked decoding** in `CompiledCircuitDecoder.decode` so the full 200,000-shot pipeline runs comfortably even on an **8 GB RAM** laptop.
* **5-Iteration BP Sweet Spot:** Empirically matched or beat 25-iteration BP accuracy while cutting runtime in half (`2.1×` speedup).
* **100% Deterministic & Offline-Ready:** Fixed execution order with zero internet calls, pinned dependencies in [`handin/requirements.txt`](handin/requirements.txt), and strict adherence to the [`CONTRACT.md`](CONTRACT.md) invariants.

---

## 3. Verified Results

| Evaluation Metric | Score | Notes |
| :--- | :--- | :--- |
| **All-Zeros Baseline (Kaggle)** | `~0.29000` | Predicting no logical flips |
| **Kaggle Public Score (Late Submission)** | **`0.04446`** | Per-observable MAE across 30% of `test.csv` |
| **Kaggle Private Score (Late Submission)** | **`0.04483`** | Per-observable MAE across 70% of `test.csv` |
| **Local `train.csv` Shot LER (200k shots)** | `0.081285` | Fraction of shots with $\ge 1$ wrong logical bit |
| **Local Sampled Shot LER (200k shots)** | `0.079915` | Freshly sampled from `X_circuit.stim` & `Z_circuit.stim` |

*(Note: Kaggle scores per-observable-bit MAE $\frac{1}{4N}\sum |\hat{\ell} - \ell|$, while local validation logs shot-level LER — whether any of the 4 logical bits failed — explaining the ~1.8× ratio between `0.0448` and `0.0813`.)*

---

## 4. Repository Structure

```text
qec-decoder-hackathon-2026/
├── CONTRACT.md                     # Immutable system & API invariants
├── AGENT_LOG.md                    # Multi-agent handover & verification log
├── data/
│   ├── X_circuit.stim              # X-basis 5-round SI1000 noisy circuit (p = 0.002)
│   └── Z_circuit.stim              # Z-basis 5-round SI1000 noisy circuit (p = 0.002)
│   # (train.csv, test.csv, sample_submission.csv are git-ignored locally)
├── handin/
│   ├── decoder.py                  # Standalone decode(circuit, detector_events) module
│   └── requirements.txt            # Exact pinned allowed packages (numpy, scipy, numba, stim)
├── scripts/
│   ├── verify_and_stress_test.py   # Banned-import AST check, parity, determinism & LER gates
│   ├── validate_train.py           # Full 200k-row train.csv LER benchmark
│   └── generate_submission.py      # Generates and validates 200k-row submission.csv
└── spec/
    ├── architecture.md             # Decoder math, GF(2) OSD & complexity budget
    └── tasks.md                    # Spec-Driven Development (SDD) task checklist
```

---

## 5. Quickstart (Run Locally)

1. **Create a clean virtual environment & install pinned dependencies:**
   ```bash
   python -m venv .venv
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   # Linux/macOS:        source .venv/bin/activate
   pip install -r handin/requirements.txt
   ```

2. **Run the stress-test & banned-library verification suite:**
   ```bash
   python scripts/verify_and_stress_test.py
   ```

3. **Benchmark on `train.csv` and generate `submission.csv`:**
   *(Place `train.csv` and `test.csv` from the Kaggle competition into `data/` first.)*
   ```bash
   python scripts/validate_train.py
   python scripts/generate_submission.py
   ```

---

## 6. References
* **Competition Repo:** [`JuliusRye/QEC_Decoder_Hakathon_2026`](https://github.com/JuliusRye/QEC_Decoder_Hakathon_2026)
* **Code:** `[[18, 4, 4]]` qLDPC memory experiment (Wang et al., 2026) under SI1000 circuit-level noise (Gidney et al., 2021).
