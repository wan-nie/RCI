import os
from typing import Callable, List, Tuple, Optional, Dict
import numpy as np
from .utils import sample_block_coords, plot_heatmap


class BlockWeightedPairSampler:

    def __init__(self, y, num_bins: int = 32, seed: int = 42):
        self.y = y
        self.descend_idx = np.argsort(y)[::-1]
        assert self.check_monotonic(self.descend_idx)

        self.N = self.descend_idx.shape[0]
        self.B = int(num_bins)
        self.rng = np.random.default_rng(seed)

        # binning
        self.bins: List[np.ndarray] = []
        self.bin_slices: List[Tuple[int, int]] = []
        self._prepare_bins()

        # Capacity matrix (upper triangular)
        self.cap_matrix = self._compute_block_capacities()
        self.triu_mask = np.triu(np.ones((self.B, self.B), dtype=bool))

        # Initial weights
        self.weights = np.ones((self.B, self.B), dtype=np.float64)
        # self._apply_upper_triangle_only()
        self.weights[~self.triu_mask] = 0.0
        self.weights = self._normalize_probs(self.weights)
        self._ema_initialized = False

    def check_monotonic(self, idx):
        return np.all(self.y[idx][:-1] >= self.y[idx][1:])
    
    def _prepare_bins(self):
        N, B = self.N, self.B
        base, rem = N // B, N % B
        start = 0
        self.bins.clear()
        self.bin_slices.clear()
        for b in range(B):
            size = base + (1 if b < rem else 0)
            end = start + size
            if end > start:
                sl = self.descend_idx[start:end]
                self.bins.append(sl)
                self.bin_slices.append((start, end))
            start = end

        # If B > number of non-empty bins, shrink B
        if len(self.bins) < self.B:
            self.B = len(self.bins)

    def _compute_block_capacities(self) -> np.ndarray:
        """
        Vectorized computation of the capacity matrix:
        - Diagonal: m*(m-1)//2
        - Upper triangular (off-diagonal): m_a * m_b
        - Lower triangular remains 0
        """
        B = self.B
        m = np.array([len(self.bins[b]) for b in range(B)], dtype=np.int64)  # [B]
        cap = np.zeros((B, B), dtype=np.int64)
        # Outer product to get the product of all block sizes
        prod = np.multiply.outer(m, m)  # [B,B]
        # Replace diagonal with m*(m-1)//2
        diag_vals = m * (m - 1) // 2
        cap = np.triu(prod, k=1)  # Upper triangular excluding diagonal
        np.fill_diagonal(cap, diag_vals)
        return cap

    def _normalize_probs(self, w: np.ndarray, epsilon_explore: float = 0.05, eps: float = 1e-8) -> np.ndarray:
        w = w.copy()
        w[~self.triu_mask] = 0.0
        w = np.maximum(w, 0.0)
        total = w.sum()
        num_blocks = int(self.triu_mask.sum())
        if total <= eps:
            p = np.zeros_like(w)
            p[self.triu_mask] = 1.0 / num_blocks
            return p
        p = w / total
        if epsilon_explore > 0.0:
            uni = np.zeros_like(p)
            uni[self.triu_mask] = 1.0 / num_blocks
            p = (1.0 - epsilon_explore) * p + epsilon_explore * uni
        return p

    def refresh_block_weights(
        self,
        scores_fn: Callable[[np.ndarray], np.ndarray],
        probe_k: int = 2000,
        ema_lambda: float = 0.2,
        tau: float = 1.5,
        eps: float = 1e-8,
        epsilon_explore: float = 0.05,
        mix_err: float = 0.0,
        max_pairs_per_probe: Optional[int] = None,
        plot_path: Optional[str] = None,
        cmap: str = "magma",
    ) -> np.ndarray:
        """
        Probe each block, estimate block difficulty based on current model scores,
        and update weights; returns the upper-triangular block sampling probability matrix p_ab.
        """
        B = self.B

        # Vectorized computation of per-block probe quota: min(cap, probe_k), valid only for upper triangular
        per_block_quota = np.minimum(self.cap_matrix, int(probe_k)).astype(np.int64)
        per_block_quota[~self.triu_mask] = 0

        if max_pairs_per_probe is not None:
            total_need = int(per_block_quota.sum())
            if total_need > max_pairs_per_probe and total_need > 0:
                scale = max_pairs_per_probe / float(total_need)
                per_block_quota = np.floor(per_block_quota.astype(np.float64) * scale).astype(np.int64)
                need_one_mask = (self.cap_matrix > 0) & self.triu_mask & (per_block_quota == 0)
                per_block_quota[need_one_mask] = 1

        # Collect probe pairs and unique indices
        # probe_pairs: List[Tuple[int, int, int, int]] = []
        uniq_idx_set = set()
        probe_pairs = {}
        for a in range(B):
            rows = self.bins[a]
            # m_a = len(rows)
            for b in range(a, B):
                cols = self.bins[b]
                # m_b = len(cols)
                k = int(per_block_quota[a, b])
                if k <= 0:
                    continue

                uptri = a == b
                pairs_k = sample_block_coords(rows, cols, k, rng=self.rng, upper_triangle_only=uptri)
                
                probe_pairs[(a, b)] = pairs_k
                for idx in pairs_k.ravel():
                    uniq_idx_set.add(int(idx))
        
        # Deduplicated forward scores
        uniq_idx = np.fromiter(uniq_idx_set, dtype=np.int64)
        uniq_idx = np.sort(uniq_idx)
        # uniq_idx = np.unique(probe_pairs[:, 2:])
        s_vals = scores_fn(uniq_idx)  # np.ndarray [U]
        assert s_vals.shape[0] == uniq_idx.shape[0]
        idx2score = {int(i): float(s) for i, s in zip(uniq_idx, s_vals)}

        # Compute per-block average loss and error rate (pair-dependent, keep simple loop)
        loss_sum = np.zeros((B, B), dtype=np.float64)
        err_sum = np.zeros((B, B), dtype=np.float64)
        cnt = np.zeros((B, B), dtype=np.int64)

        def softplus(x):
            return np.logaddexp(0.0, x)

        for (a, b), pairs_ab in probe_pairs.items():
            i_idx = pairs_ab[:, 0]
            j_idx = pairs_ab[:, 1]
            si = np.array([idx2score[int(ii)] for ii in i_idx], dtype=np.float64)
            sj = np.array([idx2score[int(jj)] for jj in j_idx], dtype=np.float64)
            margin = si - sj  # (k,)
            loss = softplus(-margin)  # (k,)
            err = (margin < 0).astype(np.float64)  # (k,)

            loss_sum[a, b] += loss.sum()
            err_sum[a, b] += err.sum()
            cnt[a, b] += margin.shape[0]
        
        denom = np.maximum(cnt, 1)
        avg_loss = loss_sum / denom
        avg_err = err_sum / denom
        avg_loss_scaled = np.power(np.maximum(avg_loss, eps), tau)
        avg_err_scaled = np.power(np.maximum(avg_err, eps), tau)

        p_ab_loss = self._normalize_probs(avg_loss_scaled, epsilon_explore=epsilon_explore, eps=eps)
        p_ab_err = self._normalize_probs(avg_err_scaled, epsilon_explore=epsilon_explore, eps=eps)
        p_ab = (1.0 - mix_err) * p_ab_loss + mix_err * p_ab_err

        if not self._ema_initialized:
            self.weights = p_ab
            self._ema_initialized = True
        else:
            self.weights = (1.0 - ema_lambda) * self.weights + ema_lambda * p_ab

        if plot_path is not None:
            plot_heatmap(self.weights, plot_path, title=os.path.basename(plot_path), cmap=cmap)

    def sample_weighted(
            self,
            total_pairs: int,
            min_quota_per_block: int = 0,
            active_idx_ratio: Optional[float] = None,
            max_active_num: int = None
        ) -> np.ndarray:
            """
            Generate total_pairs pairs (i, j) according to current block weights.
            active_idx_ratio:
                - float in (0, 1]: randomly sample this ratio of sample indices for pairing.
                - None: use all samples.
            """
            B = self.B
            K = int(total_pairs)

            # Build index subset mask
            if active_idx_ratio is None:
                mask = np.ones(self.N, dtype=bool)
            else:
                ratio = float(active_idx_ratio)
                if not (0.0 < ratio <= 1.0):
                    raise ValueError("active_idx_ratio must be in (0, 1].")
                num_select = max(1, int(np.ceil(ratio * self.N)))
                if max_active_num is not None:
                    num_select = min(num_select, max_active_num) 
                if num_select >= self.N:
                    mask = np.ones(self.N, dtype=bool)
                else:
                    pos = self.rng.choice(self.N, size=num_select, replace=False)
                    pos.sort()
                    chosen_idx = self.descend_idx[pos]
                    mask = np.zeros(self.N, dtype=bool)
                    mask[chosen_idx] = True

            filtered_bins: List[np.ndarray] = []
            for rows in self.bins:
                filtered_bins.append(rows[mask[rows]] if rows.size > 0 else rows)

            if all(len(b) == 0 for b in filtered_bins):
                return np.zeros((0, 2), dtype=np.int64)

            # Recompute capacities based on filtered bins
            m = np.array([len(filtered_bins[b]) for b in range(B)], dtype=np.int64)
            prod = np.multiply.outer(m, m)
            cap_subset = np.triu(prod, k=1)
            np.fill_diagonal(cap_subset, m * (m - 1) // 2)

            subset_mask = (cap_subset > 0) & self.triu_mask
            if not subset_mask.any():
                return np.zeros((0, 2), dtype=np.int64)

            # Re-normalize weights
            eff_weights = np.where(subset_mask, self.weights, 0.0)
            eff_weights = self._normalize_probs(eff_weights, epsilon_explore=0.0)
            if eff_weights.sum() == 0.0:
                return np.zeros((0, 2), dtype=np.int64)

            # Allocate quotas (using round, no remainder handling)
            alloc = np.round(eff_weights * K).astype(np.int64)

            mask_cap_pos = subset_mask
            alloc[~mask_cap_pos] = 0

            if min_quota_per_block > 0:
                alloc[mask_cap_pos] = np.maximum(alloc[mask_cap_pos], min_quota_per_block)
            alloc[mask_cap_pos] = np.minimum(alloc[mask_cap_pos], cap_subset[mask_cap_pos])

            # Sample per block
            pairs = []
            for a in range(B):
                rows = filtered_bins[a]
                if rows.size == 0:
                    continue
                for b in range(a, B):
                    if not subset_mask[a, b]:
                        continue
                    cols = filtered_bins[b]
                    if cols.size == 0:
                        continue
                    k = int(alloc[a, b])
                    if k <= 0:
                        continue
                    uptri = (a == b)
                    pairs_k = sample_block_coords(
                        rows, cols, k, rng=self.rng, upper_triangle_only=uptri
                    )
                    if pairs_k.size > 0:
                        pairs.append(pairs_k)

            if len(pairs) == 0:
                return np.zeros((0, 2), dtype=np.int64)
            return np.concatenate(pairs, axis=0)

