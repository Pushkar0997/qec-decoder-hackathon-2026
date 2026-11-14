# QEC Decoder Hackathon 2026 — `[[18, 4, 4]]` qLDPC Syndrome Decoder

A constrained, rule-compliant **BP + OSD** syndrome decoder built for the [Quantum Error Correction: Decoder Hackathon](https://www.kaggle.com/competitions/quantum-error-correction-decoder) (November 2026), hosted by the Niels Bohr Institute / NQCP (University of Copenhagen), NVIDIA, Riverlane, DTU Physics, QMATH, and DeiC.

> **Kaggle Score (Late Submission):** **`0.04446` Public / `0.04483` Private** per-observable MAE (~6.5× lower than the `~0.29` all-zeros baseline), implemented using only `numpy`, `scipy`, `numba`, and `stim` — with zero external QEC decoder packages (`pymatching`, `ldpc`, `stimbposd`, etc.).

---

## 1. The Problem in Plain English

Quantum bits (qubits) are fragile: quantum gates, resets, and measurements all fail at small probabilities. To protect data, this experiment encodes **4 logical qubits** across **18 physical data qubits** monitored by **14 check qubits** — a `[[18, 4, 4]]` quantum Low-Density Parity-Check (**qLDPC**) code.

1. You cannot directly read the 4 logical qubits mid-computation without collapsing them.
2. Instead, over **5 rounds** of SI1000 circuit-level noise ($p = 0.002$), the check qubits emit **70 binary "detector" alarms** (`D0`–`D69`): `0` if local checks agree, `1` if an error occurred nearby in space or time.
3. **The Decoder's Task:** Given those 70 alarm bits, predict which of the **4 logical observables** (`L0`–`L3`) flipped by the end of the circuit across **200,000 test shots** (`100,000` X-basis + `100,000` Z-basis).

---

## 2. Implementation Architecture

Competition rules forbid importing existing QEC decoder packages (`pymatching`, `ldpc`, `stimbposd`, `beliefmatching`, `tesseract-decoder`, `fusion-blossom`). [`handin/decoder.py`](handin/decoder.py) implements a standard BP + OSD cascade in Numba-compiled Python:

```text
  70-bit Detector Syndrome (s)
             │
             ▼
  1. Stim DEM Extraction: circuit.detector_error_model() -> (H, L, priors)
             │
             ▼
  2. Fast-Path Lookup: zero-syndrome + open-addressed weight-1/2 syndrome LUT
             │ (misses)
             ▼
  3. Numba float32 Min-Sum BP: 5 iterations, pre-allocated per-thread workspaces
             │ (if H·ê != s)
             ▼
  4. 128-bit Packed GF(2) OSD: RREF ordered by BP reliability + limited comb. sweep
             │
             ▼
  4-bit Observable Prediction (L0..L3)
```

### Key Engineering Details
* **Bounded-Memory Execution:** Uses `float32` BP messages/LLRs, per-thread reusable workspaces (zero per-shot allocations), and **10,000-shot chunked decoding** so the full 200k-shot run fits inside an **8 GB RAM** laptop.
* **Import-Time JIT Warmup:** Numba `@njit` kernels are compiled on dummy data during `import decoder` (before the competition timer starts).
* **100% Deterministic:** Fixed execution order with no internet calls and exact pinned versions in [`handin/requirements.txt`](handin/requirements.txt).

---

## 3. Verified Results & Honest Caveats

| Evaluation Metric | Score | Notes |
| :--- | :--- | :--- |
| **All-Zeros Baseline (Kaggle)** | `~0.29000` | Predicting no logical flips |
| **Kaggle Public Score (Late Submission)** | **`0.04446`** | Per-observable MAE across 30% of `test.csv` |
| **Kaggle Private Score (Late Submission)** | **`0.04483`** | Per-observable MAE across 70% of `test.csv` |
| **Local `train.csv` Shot LER (200k shots)** | `0.081285` | Fraction of shots with $\ge 1$ wrong logical bit |
| **Local Sampled Shot LER (200k shots)** | `0.079915` | Freshly sampled from `X_circuit.stim` & `Z_circuit.stim` |

*(Kaggle scores per-observable-bit MAE $\frac{1}{4N}\sum |\hat{\ell} - \ell|$, while local validation logs shot-level LER — whether any of the 4 logical bits failed — explaining the ~1.8× ratio between `0.0448` and `0.0813`.)*

### Known Limitations & Headroom
* **Runtime on Low-Power Hardware:** Runs in ~50 s for 200k shots on a 1.2 GHz dual-core i3-10th Gen laptop (above the 15.4 s judging target on 48 EPYC vCPUs + B200, where multi-threading closes much of that gap, though hard OSD-CS shots remain the bottleneck).
* **Hyperparameter Tuning:** Constants (`ms_scale=0.78`, `damping=0.20`, `osd_w1=64`, `osd_w2=20`) were tuned on sampled shots; full sensitivity curves across BP iterations and pure LUT+OSD paths are not yet plotted.
* **Scope:** Built strictly as a self-contained competition artifact, separate from [`Pushkar0997/qecscreen`](https://github.com/Pushkar0997/qecscreen).

---

## 4. Repository Structure

```text
qec-decoder-hackathon-2026/
├── CONTRACT.md                     # Decoder API, determinism, and dependency invariants
├── AGENT_LOG.md                    # Verification and run history
├── data/
│   ├── X_circuit.stim              # X-basis 5-round SI1000 noisy circuit (p = 0.002)
│   └── Z_circuit.stim              # Z-basis 5-round SI1000 noisy circuit (p = 0.002)
├── handin/
│   ├── decoder.py                  # Standalone decode(circuit, detector_events) module
│   └── requirements.txt            # Pinned allowed packages (numpy, scipy, numba, stim)
├── scripts/
│   ├── ablation_benchmark.py       # Per-basis MAE, shot LER, and throughput benchmark
│   ├── verify_and_stress_test.py   # Banned-import AST check, parity, determinism & LER gates
│   ├── validate_train.py           # Full 200k-row train.csv LER check
│   └── generate_submission.py      # Generates and validates 200k-row submission.csv
├── tests/
│   └── test_decoder.py             # Unit tests for contract, determinism, and AST compliance
└── spec/
    ├── architecture.md             # Decoder math & complexity budget
    └── tasks.md                    # Implementation checklist
```

---

## 5. Quickstart (Run Locally)

1. **Create a virtual environment & install pinned dependencies:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r handin/requirements.txt
   ```

2. **Run unit tests, stress checks, and local benchmark:**
   ```powershell
   python -m unittest discover -s tests -v
   python scripts/verify_and_stress_test.py
   python scripts/ablation_benchmark.py --shots 20000
   ```

3. **Validate on `train.csv` and generate `submission.csv`:**
   *(Place `train.csv` and `test.csv` from Kaggle into `data/` first.)*
   ```powershell
   python scripts/validate_train.py
   python scripts/generate_submission.py
   ```
