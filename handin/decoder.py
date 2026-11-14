"""
Standalone [[18, 4, 4]] qLDPC Syndrome Decoder
Repository: Pushkar0997/qec-decoder-hackathon-2026
Entrypoint: decode(circuit: stim.Circuit, detector_events: np.ndarray) -> np.ndarray
Allowed Stack: numpy, scipy, numba, stim (Zero banned QEC packages).
"""

from __future__ import annotations

import hashlib

import numba
import numpy as np
import stim
from numba import njit, prange
from numpy.typing import NDArray

# ============================================================================
# 1. OFFICIAL DEM EXTRACTION (EXACT COMPETITION SPECIFICATION)
# ============================================================================

def problem_info(
    circuit: stim.Circuit,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.float64]]:
    """Returns H (detectors x mechanisms), L (observables x mechanisms) and p (probability of each mechanism)."""
    dem = circuit.detector_error_model()
    columns, priors = {}, []
    for instruction in dem.flattened():
        if not isinstance(instruction, stim.DemInstruction):
            continue
        if instruction.type != "error":
            continue
        detectors, observables = set(), set()
        for target in instruction.targets_copy():
            if not isinstance(target, stim.DemTarget):
                continue
            if target.is_relative_detector_id():
                detectors ^= {target.val}
            elif target.is_logical_observable_id():
                observables ^= {target.val}
        key, p = (frozenset(detectors), frozenset(observables)), instruction.args_copy()[0]
        if key in columns:  # same effect as an earlier mechanism: merge them
            q = priors[columns[key]]
            priors[columns[key]] = q * (1 - p) + p * (1 - q)
        else:
            columns[key] = len(priors)
            priors.append(p)
    H = np.zeros((dem.num_detectors, len(priors)), dtype=np.uint8)
    L = np.zeros((dem.num_observables, len(priors)), dtype=np.uint8)
    for (detectors, observables), j in columns.items():
        H[list(detectors), j] = 1
        L[list(observables), j] = 1
    return H, L, np.array(priors, dtype=np.float64)


# ============================================================================
# 2. NUMBA 128-BIT HASH LUT & BIT-MANIPULATION KERNELS
# ============================================================================

@njit(inline="always", cache=True)
def _hash_128(lo: np.uint64, hi: np.uint64, mask: np.uint64) -> np.uint64:
    """SplitMix64-based 128-bit syndrome hash for O(1) table lookup."""
    h = lo ^ (hi * np.uint64(0x9E3779B97F4A7C15))
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    h = h ^ (h >> np.uint64(31))
    return h & mask


@njit(inline="always", cache=True)
def _lowest_bit_index_64(x: np.uint64) -> int:
    """Returns index (0..63) of the lowest set bit in a non-zero uint64."""
    for b in range(64):
        if (x >> np.uint64(b)) & np.uint64(1):
            return b
    return -1


@njit(cache=True)
def _build_syndrome_lut(
    col_lo: NDArray[np.uint64],
    col_hi: NDArray[np.uint64],
    obs_mask: NDArray[np.uint8],
    llr: NDArray[np.float32],
    top_k_w2: int,
    table_bits: int,
) -> tuple[
    NDArray[np.uint64],
    NDArray[np.uint64],
    NDArray[np.float64],
    NDArray[np.uint8],
    NDArray[np.uint8],
    np.uint64,
]:
    """
    Precomputes an open-addressed hash table mapping 70-bit syndromes (lo, hi)
    to their minimum-LLR Weight-1 and Top-K Weight-2 error mechanisms.
    """
    table_size = 1 << table_bits
    mask = np.uint64(table_size - 1)

    lut_lo = np.zeros(table_size, dtype=np.uint64)
    lut_hi = np.zeros(table_size, dtype=np.uint64)
    lut_cost = np.full(table_size, 1e30, dtype=np.float64)
    lut_obs = np.zeros(table_size, dtype=np.uint8)
    lut_w = np.zeros(table_size, dtype=np.uint8)

    M = col_lo.shape[0]

    # 1. Insert all Weight-1 mechanisms
    for j in range(M):
        s_lo = col_lo[j]
        s_hi = col_hi[j]
        if s_lo == 0 and s_hi == 0:
            continue
        c = llr[j]
        ob = obs_mask[j]
        slot = _hash_128(np.uint64(s_lo), np.uint64(s_hi), mask)
        while lut_w[slot] != 0:
            if lut_lo[slot] == s_lo and lut_hi[slot] == s_hi:
                if c < lut_cost[slot]:
                    lut_cost[slot] = c
                    lut_obs[slot] = ob
                    lut_w[slot] = 1
                break
            slot = (slot + np.uint64(1)) & mask
        else:
            lut_lo[slot] = s_lo
            lut_hi[slot] = s_hi
            lut_cost[slot] = c
            lut_obs[slot] = ob
            lut_w[slot] = 1

    # 2. Insert Top-K Weight-2 pairs (ordered by lowest prior LLR)
    order = np.argsort(llr)
    K = min(M, top_k_w2)
    for idx1 in range(K):
        j1 = order[idx1]
        lo1 = col_lo[j1]
        hi1 = col_hi[j1]
        c1 = llr[j1]
        ob1 = obs_mask[j1]
        for idx2 in range(idx1 + 1, K):
            j2 = order[idx2]
            s_lo = lo1 ^ col_lo[j2]
            s_hi = hi1 ^ col_hi[j2]
            if s_lo == 0 and s_hi == 0:
                continue
            c = c1 + llr[j2]
            ob = ob1 ^ obs_mask[j2]
            slot = _hash_128(np.uint64(s_lo), np.uint64(s_hi), mask)
            while lut_w[slot] != 0:
                if lut_lo[slot] == s_lo and lut_hi[slot] == s_hi:
                    if c < lut_cost[slot]:
                        lut_cost[slot] = c
                        lut_obs[slot] = ob
                        if lut_w[slot] != 1:
                            lut_w[slot] = 2
                    break
                slot = (slot + np.uint64(1)) & mask
            else:
                lut_lo[slot] = s_lo
                lut_hi[slot] = s_hi
                lut_cost[slot] = c
                lut_obs[slot] = ob
                lut_w[slot] = 2

    return lut_lo, lut_hi, lut_cost, lut_obs, lut_w, mask


# ============================================================================
# 3. NUMBA SINGLE-SHOT & PARALLEL BATCH DECODER KERNELS
# ============================================================================

@njit(cache=True)
def _decode_single_shot(
    s_lo: np.uint64,
    s_hi: np.uint64,
    syndrome: NDArray[np.uint8],
    col_lo: NDArray[np.uint64],
    col_hi: NDArray[np.uint64],
    obs_mask: NDArray[np.uint8],
    llr: NDArray[np.float32],
    chk_ptr: NDArray[np.int32],
    chk_var: NDArray[np.int32],
    var_edges: NDArray[np.int32],
    var_ptr: NDArray[np.int32],
    lut_lo: NDArray[np.uint64],
    lut_hi: NDArray[np.uint64],
    lut_cost: NDArray[np.float64],
    lut_obs: NDArray[np.uint8],
    lut_w: NDArray[np.uint8],
    lut_mask: np.uint64,
    msg_c2v: NDArray[np.float32],
    msg_v2c: NDArray[np.float32],
    post_llr: NDArray[np.float32],
    max_bp_iter: int,
    ms_scale: float,
    damping: float,
    osd_w1: int,
    osd_w2: int,
) -> np.uint8:
    """
    4-Stage Cascade Decoder for a single 70-detector shot:
      Stage 0: Zero-syndrome & Weight-1 / High-Confidence Weight-2 Hash LUT
      Stage 1: Damped Normalized Min-Sum BP with early syndrome convergence
      Stage 2: 128-bit Gauss-Jordan RREF OSD-0
      Stage 3: OSD-CS (Combination Sweep Order-2) + LUT Tournament
    """
    # --- STAGE 0A: Zero Syndrome ---
    if s_lo == 0 and s_hi == 0:
        return np.uint8(0)

    # --- STAGE 0B: O(1) Hash LUT ---
    slot = _hash_128(s_lo, s_hi, lut_mask)
    lut_hit_w = np.uint8(0)
    lut_hit_obs = np.uint8(0)
    lut_hit_cost = 1e30
    while lut_w[slot] != 0:
        if lut_lo[slot] == s_lo and lut_hi[slot] == s_hi:
            lut_hit_w = lut_w[slot]
            lut_hit_obs = lut_obs[slot]
            lut_hit_cost = lut_cost[slot]
            break
        slot = (slot + np.uint64(1)) & lut_mask

    # Weight-1 fault is guaranteed minimal for d_circ = 3
    if lut_hit_w == 1:
        return lut_hit_obs

    # --- STAGE 1: Damped Normalized Min-Sum Belief Propagation ---
    num_det = syndrome.shape[0]
    num_var = llr.shape[0]

    # Per-thread scratch buffers (allocated once in the batch kernel). Every
    # slot used below is written before it is read, so no clearing is needed.

    for v in range(num_var):
        pv = llr[v]
        post_llr[v] = pv
        for idx in range(var_ptr[v], var_ptr[v + 1]):
            msg_v2c[var_edges[idx]] = pv

    for _ in range(max_bp_iter):
        # Check-to-Variable step
        for c in range(num_det):
            start = chk_ptr[c]
            end = chk_ptr[c + 1]
            if start == end:
                continue

            total_sign = -1 if syndrome[c] != 0 else 1
            min1 = 1e30
            min2 = 1e30
            min1_edge = -1

            for e in range(start, end):
                val = msg_v2c[e]
                if val < 0.0:
                    total_sign = -total_sign
                    aval = -val
                else:
                    aval = val
                if aval < min1:
                    min2 = min1
                    min1 = aval
                    min1_edge = e
                elif aval < min2:
                    min2 = aval

            for e in range(start, end):
                val = msg_v2c[e]
                sgn = total_sign if val >= 0.0 else -total_sign
                mag = min2 if e == min1_edge else min1
                msg_c2v[e] = ms_scale * sgn * mag

        # Variable-to-Check step + Hard Decision Check
        cand_lo = np.uint64(0)
        cand_hi = np.uint64(0)
        cand_obs = np.uint8(0)
        cand_cost = 0.0

        for v in range(num_var):
            total = llr[v]
            v_start = var_ptr[v]
            v_end = var_ptr[v + 1]
            for idx in range(v_start, v_end):
                total += msg_c2v[var_edges[idx]]
            post_llr[v] = total

            if total < 0.0:
                cand_lo ^= col_lo[v]
                cand_hi ^= col_hi[v]
                cand_obs ^= obs_mask[v]
                cand_cost += llr[v]

            for idx in range(v_start, v_end):
                e = var_edges[idx]
                new_m = total - msg_c2v[e]
                if new_m > 50.0:
                    new_m = 50.0
                elif new_m < -50.0:
                    new_m = -50.0
                msg_v2c[e] = (1.0 - damping) * new_m + damping * msg_v2c[e]

        # Early convergence check: H @ e_hat == D (mod 2)
        if cand_lo == s_lo and cand_hi == s_hi:
            if lut_hit_w == 2 and lut_hit_cost < cand_cost:
                return lut_hit_obs
            return cand_obs

    # --- STAGE 2: 128-Bit Gauss-Jordan RREF OSD-0 ---
    order = np.argsort(post_llr)

    basis_pivot_bit = np.empty(70, dtype=np.int32)
    basis_col_idx = np.empty(70, dtype=np.int32)
    echelon_lo = np.zeros(70, dtype=np.uint64)
    echelon_hi = np.zeros(70, dtype=np.uint64)
    coord_lo = np.zeros(70, dtype=np.uint64)
    coord_hi = np.zeros(70, dtype=np.uint64)

    non_basis_col = np.empty(osd_w1, dtype=np.int32)
    non_basis_dep_lo = np.empty(osd_w1, dtype=np.uint64)
    non_basis_dep_hi = np.empty(osd_w1, dtype=np.uint64)
    n_non_basis = 0
    n_basis = 0

    for rank_idx in range(num_var):
        col = order[rank_idx]
        v_lo = col_lo[col]
        v_hi = col_hi[col]
        if v_lo == 0 and v_hi == 0:
            continue

        c_lo = np.uint64(0)
        c_hi = np.uint64(0)

        for b in range(n_basis):
            p_bit = basis_pivot_bit[b]
            if p_bit < 64:
                bit_set = (v_lo >> np.uint64(p_bit)) & np.uint64(1)
            else:
                bit_set = (v_hi >> np.uint64(p_bit - 64)) & np.uint64(1)
            if bit_set != 0:
                v_lo ^= echelon_lo[b]
                v_hi ^= echelon_hi[b]
                c_lo ^= coord_lo[b]
                c_hi ^= coord_hi[b]

        if v_lo != 0 or v_hi != 0:
            if v_lo != 0:
                p_bit = _lowest_bit_index_64(np.uint64(v_lo))
            else:
                p_bit = 64 + _lowest_bit_index_64(np.uint64(v_hi))

            if n_basis < 64:
                c_lo ^= np.uint64(1) << np.uint64(n_basis)
            else:
                c_hi ^= np.uint64(1) << np.uint64(n_basis - 64)

            # Backward-eliminate pivot from existing basis rows to maintain exact RREF
            for b in range(n_basis):
                if p_bit < 64:
                    has_p = (echelon_lo[b] >> np.uint64(p_bit)) & np.uint64(1)
                else:
                    has_p = (echelon_hi[b] >> np.uint64(p_bit - 64)) & np.uint64(1)
                if has_p != 0:
                    echelon_lo[b] ^= v_lo
                    echelon_hi[b] ^= v_hi
                    coord_lo[b] ^= c_lo
                    coord_hi[b] ^= c_hi

            echelon_lo[n_basis] = v_lo
            echelon_hi[n_basis] = v_hi
            coord_lo[n_basis] = c_lo
            coord_hi[n_basis] = c_hi
            basis_pivot_bit[n_basis] = p_bit
            basis_col_idx[n_basis] = col
            n_basis += 1
        elif n_non_basis < osd_w1:
            non_basis_col[n_non_basis] = col
            non_basis_dep_lo[n_non_basis] = c_lo
            non_basis_dep_hi[n_non_basis] = c_hi
            n_non_basis += 1

        if n_basis == num_det and n_non_basis == osd_w1:
            break

    # Solve H_B @ e_B0 = syndrome
    rem_lo = s_lo
    rem_hi = s_hi
    e_B0_lo = np.uint64(0)
    e_B0_hi = np.uint64(0)

    for b in range(n_basis):
        p_bit = basis_pivot_bit[b]
        if p_bit < 64:
            bit_set = (rem_lo >> np.uint64(p_bit)) & np.uint64(1)
        else:
            bit_set = (rem_hi >> np.uint64(p_bit - 64)) & np.uint64(1)
        if bit_set != 0:
            rem_lo ^= echelon_lo[b]
            rem_hi ^= echelon_hi[b]
            e_B0_lo ^= coord_lo[b]
            e_B0_hi ^= coord_hi[b]

    basis_llr = np.empty(n_basis, dtype=np.float64)
    basis_obs = np.empty(n_basis, dtype=np.uint8)
    for b in range(n_basis):
        c_idx = basis_col_idx[b]
        basis_llr[b] = llr[c_idx]
        basis_obs[b] = obs_mask[c_idx]

    # OSD-0 Cost & Observable
    best_cost = 0.0
    best_obs = np.uint8(0)
    for b in range(n_basis):
        if b < 64:
            active = (e_B0_lo >> np.uint64(b)) & np.uint64(1)
        else:
            active = (e_B0_hi >> np.uint64(b - 64)) & np.uint64(1)
        if active != 0:
            best_cost += basis_llr[b]
            best_obs ^= basis_obs[b]

    # Compare against Weight-2 LUT hit
    if lut_hit_w == 2 and lut_hit_cost < best_cost:
        best_cost = lut_hit_cost
        best_obs = lut_hit_obs

    # --- STAGE 3A: OSD-CS Weight-1 Sweep ---
    for i in range(n_non_basis):
        col_i = non_basis_col[i]
        cost = llr[col_i]
        if cost >= best_cost:
            continue
        cand_lo = e_B0_lo ^ non_basis_dep_lo[i]
        cand_hi = e_B0_hi ^ non_basis_dep_hi[i]
        obs = obs_mask[col_i]
        for b in range(n_basis):
            if b < 64:
                active = (cand_lo >> np.uint64(b)) & np.uint64(1)
            else:
                active = (cand_hi >> np.uint64(b - 64)) & np.uint64(1)
            if active != 0:
                cost += basis_llr[b]
                obs ^= basis_obs[b]
        if cost < best_cost:
            best_cost = cost
            best_obs = obs

    # --- STAGE 3B: OSD-CS Weight-2 Sweep ---
    w2_limit = min(n_non_basis, osd_w2)
    for i in range(w2_limit):
        col_i = non_basis_col[i]
        dep_i_lo = non_basis_dep_lo[i]
        dep_i_hi = non_basis_dep_hi[i]
        cost_i = llr[col_i]
        obs_i = obs_mask[col_i]
        for j in range(i + 1, w2_limit):
            col_j = non_basis_col[j]
            cost = cost_i + llr[col_j]
            if cost >= best_cost:
                continue
            cand_lo = e_B0_lo ^ dep_i_lo ^ non_basis_dep_lo[j]
            cand_hi = e_B0_hi ^ dep_i_hi ^ non_basis_dep_hi[j]
            obs = obs_i ^ obs_mask[col_j]
            for b in range(n_basis):
                if b < 64:
                    active = (cand_lo >> np.uint64(b)) & np.uint64(1)
                else:
                    active = (cand_hi >> np.uint64(b - 64)) & np.uint64(1)
                if active != 0:
                    cost += basis_llr[b]
                    obs ^= basis_obs[b]
            if cost < best_cost:
                best_cost = cost
                best_obs = obs

    return np.uint8(best_obs)


@njit(parallel=True, cache=True)
def _decode_batch_kernel(
    syndromes: NDArray[np.uint8],
    col_lo: NDArray[np.uint64],
    col_hi: NDArray[np.uint64],
    obs_mask: NDArray[np.uint8],
    llr: NDArray[np.float32],
    chk_ptr: NDArray[np.int32],
    chk_var: NDArray[np.int32],
    var_edges: NDArray[np.int32],
    var_ptr: NDArray[np.int32],
    lut_lo: NDArray[np.uint64],
    lut_hi: NDArray[np.uint64],
    lut_cost: NDArray[np.float64],
    lut_obs: NDArray[np.uint8],
    lut_w: NDArray[np.uint8],
    lut_mask: np.uint64,
    n_threads: int,
    max_bp_iter: int,
    ms_scale: float,
    damping: float,
    osd_w1: int,
    osd_w2: int,
) -> NDArray[np.uint8]:
    """Parallel batch syndrome decoding across all shots."""
    n_shots = syndromes.shape[0]
    n_det = syndromes.shape[1]
    num_var = llr.shape[0]
    num_edges = chk_var.shape[0]
    preds = np.zeros((n_shots, 4), dtype=np.uint8)

    # One reusable BP workspace per worker thread: avoids ~85 KB of per-shot
    # allocation/zeroing, which otherwise dominated runtime on large batches.
    msg_c2v_ws = np.zeros((n_threads, num_edges), dtype=np.float32)
    msg_v2c_ws = np.zeros((n_threads, num_edges), dtype=np.float32)
    post_llr_ws = np.zeros((n_threads, num_var), dtype=np.float32)

    for s in prange(n_shots):
        tid = numba.get_thread_id()  # pyright: ignore[reportPrivateImportUsage]
        row = syndromes[s]
        s_lo = np.uint64(0)
        s_hi = np.uint64(0)
        for d in range(n_det):
            if row[d] != 0:
                if d < 64:
                    s_lo |= np.uint64(1) << np.uint64(d)
                else:
                    s_hi |= np.uint64(1) << np.uint64(d - 64)

        obs_bits = _decode_single_shot(
            s_lo,
            s_hi,
            row,
            col_lo,
            col_hi,
            obs_mask,
            llr,
            chk_ptr,
            chk_var,
            var_edges,
            var_ptr,
            lut_lo,
            lut_hi,
            lut_cost,
            lut_obs,
            lut_w,
            lut_mask,
            msg_c2v_ws[tid],
            msg_v2c_ws[tid],
            post_llr_ws[tid],
            max_bp_iter,
            ms_scale,
            damping,
            osd_w1,
            osd_w2,
        )
        preds[s, 0] = obs_bits & np.uint8(1)
        preds[s, 1] = (obs_bits >> np.uint8(1)) & np.uint8(1)
        preds[s, 2] = (obs_bits >> np.uint8(2)) & np.uint8(1)
        preds[s, 3] = (obs_bits >> np.uint8(3)) & np.uint8(1)

    return preds


# ============================================================================
# 4. COMPILED CIRCUIT DECODER & OFFICIAL `decode` ENTRYPOINT
# ============================================================================

# Shots per worker batch: caps peak RAM (~10k x 70 bytes of syndromes plus
# per-thread BP scratch) independently of the submission size.
CHUNK_SHOTS = 10_000

class CompiledCircuitDecoder:
    """Precomputes bitpacked Tanner graph and Weight-1/2 Syndrome LUT for a Stim circuit."""

    def __init__(
        self,
        circuit: stim.Circuit | None = None,
        H: NDArray[np.uint8] | None = None,
        L: NDArray[np.uint8] | None = None,
        p: NDArray[np.float64] | None = None,
        max_bp_iter: int = 5,
        ms_scale: float = 0.78,
        damping: float = 0.20,
        osd_w1: int = 64,
        osd_w2: int = 20,
        top_k_w2: int = 480,
        lut_bits: int = 19,
    ):
        if circuit is not None:
            H, L, p = problem_info(circuit)
        assert H is not None
        assert L is not None
        assert p is not None

        self.H = np.ascontiguousarray(H, dtype=np.uint8)
        self.L = np.ascontiguousarray(L, dtype=np.uint8)
        self.p = np.clip(np.ascontiguousarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        self.llr = np.ascontiguousarray(np.log((1.0 - self.p) / self.p), dtype=np.float32)

        self.max_bp_iter = max_bp_iter
        self.ms_scale = ms_scale
        self.damping = damping
        self.osd_w1 = osd_w1
        self.osd_w2 = osd_w2

        num_det, num_var = self.H.shape

        # Bitpack columns of H into (col_lo, col_hi)
        self.col_lo = np.zeros(num_var, dtype=np.uint64)
        self.col_hi = np.zeros(num_var, dtype=np.uint64)
        for d in range(num_det):
            active_cols = np.flatnonzero(self.H[d])
            if d < 64:
                self.col_lo[active_cols] |= np.uint64(1) << np.uint64(d)
            else:
                self.col_hi[active_cols] |= np.uint64(1) << np.uint64(d - 64)

        # Bitpack columns of L into 4-bit uint8 masks
        self.obs_mask = np.zeros(num_var, dtype=np.uint8)
        for ob in range(self.L.shape[0]):
            active_cols = np.flatnonzero(self.L[ob])
            self.obs_mask[active_cols] |= np.uint8(1) << np.uint8(ob)

        # Build CSR Tanner Graph
        chk_ptr = [0]
        chk_var_list = []
        for c in range(num_det):
            vs = np.flatnonzero(self.H[c])
            chk_var_list.extend(vs.tolist())
            chk_ptr.append(len(chk_var_list))

        self.chk_ptr = np.array(chk_ptr, dtype=np.int32)
        self.chk_var = np.array(chk_var_list, dtype=np.int32)

        var_edge_lists = [[] for _ in range(num_var)]
        for e, v in enumerate(chk_var_list):
            var_edge_lists[v].append(e)

        var_ptr = [0]
        var_edges_flat = []
        for v in range(num_var):
            var_edges_flat.extend(var_edge_lists[v])
            var_ptr.append(len(var_edges_flat))

        self.var_ptr = np.array(var_ptr, dtype=np.int32)
        self.var_edges = np.array(var_edges_flat, dtype=np.int32)

        # Build Weight-1 & Top-K Weight-2 Syndrome Hash LUT
        (
            self.lut_lo,
            self.lut_hi,
            self.lut_cost,
            self.lut_obs,
            self.lut_w,
            self.lut_mask,
        ) = _build_syndrome_lut(
            self.col_lo,
            self.col_hi,
            self.obs_mask,
            self.llr,
            top_k_w2,
            lut_bits,
        )

    def _decode_chunk(
        self, syndromes_u8: NDArray[np.uint8]
    ) -> NDArray[np.uint8]:
        """Runs the parallel Numba kernel over one contiguous block of shots."""
        return _decode_batch_kernel(
            syndromes_u8,
            self.col_lo,
            self.col_hi,
            self.obs_mask,
            self.llr,
            self.chk_ptr,
            self.chk_var,
            self.var_edges,
            self.var_ptr,
            self.lut_lo,
            self.lut_hi,
            self.lut_cost,
            self.lut_obs,
            self.lut_w,
            self.lut_mask,
            int(numba.get_num_threads()),
            self.max_bp_iter,
            self.ms_scale,
            self.damping,
            self.osd_w1,
            self.osd_w2,
        )

    def decode(
        self, detector_events: NDArray[np.uint8] | NDArray[np.bool_]
    ) -> NDArray[np.uint8]:
        """
        Decodes (N_shots, 70) detector_events -> (N_shots, 4) uint8 predictions.

        Shots are processed in fixed-size chunks so peak resident memory stays
        bounded (~10k shots per worker batch) even for 200k-row submissions.
        """
        syndromes_u8 = np.ascontiguousarray(detector_events, dtype=np.uint8)
        if syndromes_u8.ndim != 2 or syndromes_u8.shape[1] != self.H.shape[0]:
            raise ValueError(
                f"detector_events must be 2D with {self.H.shape[0]} detectors, "
                f"got shape {syndromes_u8.shape}"
            )

        n_shots = syndromes_u8.shape[0]
        if n_shots <= CHUNK_SHOTS:
            return self._decode_chunk(syndromes_u8)

        preds = np.empty((n_shots, 4), dtype=np.uint8)
        for start in range(0, n_shots, CHUNK_SHOTS):
            stop = min(start + CHUNK_SHOTS, n_shots)
            preds[start:stop] = self._decode_chunk(syndromes_u8[start:stop])
        return preds


_CIRCUIT_CACHE: dict[str, CompiledCircuitDecoder] = {}


def decode(
    circuit: stim.Circuit,
    detector_events: NDArray[np.uint8] | NDArray[np.bool_],
) -> NDArray[np.uint8]:
    """
    Competition Entrypoint:
    Accepts a stim.Circuit and (N_shots, 70) detector_events array.
    Returns (N_shots, 4) uint8 array of predicted logical observables.
    """
    key = hashlib.sha256(str(circuit).encode("utf-8")).hexdigest()
    decoder = _CIRCUIT_CACHE.get(key)
    if decoder is None:
        decoder = CompiledCircuitDecoder(circuit=circuit)
        _CIRCUIT_CACHE[key] = decoder
    return decoder.decode(detector_events)


# ============================================================================
# 5. IMPORT-TIME NUMBA JIT WARMUP (INV-H06)
# ============================================================================

def _warmup_numba() -> None:
    dummy_H = np.eye(70, 75, dtype=np.uint8)
    dummy_L = np.zeros((4, 75), dtype=np.uint8)
    dummy_L[0, :4] = 1
    dummy_p = np.full(75, 0.002, dtype=np.float64)
    dec = CompiledCircuitDecoder(
        H=dummy_H, L=dummy_L, p=dummy_p,
        max_bp_iter=2, osd_w1=4, osd_w2=2, top_k_w2=10, lut_bits=8
    )
    dummy_shots = np.zeros((2, 70), dtype=np.uint8)
    dummy_shots[1, 0] = 1
    dummy_shots[1, 1] = 1
    _ = dec.decode(dummy_shots)


_warmup_numba()
