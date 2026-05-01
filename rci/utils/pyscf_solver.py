import numpy as np
import time
from pyscf import lib
from pyscf.fci import selected_ci, direct_spin1
from pyscf.fci.selected_ci import _as_SCIvector


class SelectedCISolver:
    """
    An arbitrary-basis Davidson solver based on PySCF selected_ci.

    Parameters
    ----------
    basis   : list[SlaterDeterminant]  User-specified determinant basis
    h1eff   : ndarray (ncas, ncas)     One-body integrals (chemist's notation)
    eri_4d  : ndarray (ncas,)*4        Two-body integrals (chemist's notation, restored form)
    ncas    : int                      Number of active orbitals
    nelecas : tuple(int, int)          (n_alpha, n_beta)
    """

    def __init__(self, basis, h1eff, eri_4d, ncas, nelecas):
        self.basis   = basis
        self.h1eff   = h1eff
        self.eri_4d  = eri_4d
        self.ncas    = ncas
        self.nelecas = nelecas
        self.dim     = len(basis)

        # Results (populated after solve)
        self.converged = False
        self.e_ci      = None
        self.civecs    = None

        # Internal cache (populated after _build)
        self._ci_strs_tuple = None
        self._ndet_a        = None
        self._ndet_b        = None
        self._basis_ia      = None
        self._basis_ib      = None
        self._h2eff         = None
        self._link_index    = None
        self._hdiag         = None
        self._civec_buffer  = None

        self._build()

    @staticmethod
    def _occs_to_bitstring(occs):
        s = 0
        for k in occs:
            s |= (1 << k)
        return s

    def _build(self):
        basis = self.basis
        bs    = self._occs_to_bitstring

        # Batch-generate α/β bitstrings for all basis determinants
        all_strs_a = np.array(
            [bs(sd.alpha_occupied_indices()) for sd in basis], dtype=np.int64
        )
        all_strs_b = np.array(
            [bs(sd.beta_occupied_indices()) for sd in basis], dtype=np.int64
        )

        # ci_strs: deduplicate + sort ascending
        ci_strs_a = np.array(sorted(set(all_strs_a.tolist())), dtype=np.int64)
        ci_strs_b = np.array(sorted(set(all_strs_b.tolist())), dtype=np.int64)
        self._ci_strs_tuple = (ci_strs_a, ci_strs_b)
        self._ndet_a = len(ci_strs_a)
        self._ndet_b = len(ci_strs_b)

        print(f"[build] α strings: {self._ndet_a},  β strings: {self._ndet_b}")
        print(f"[build] Cartesian product: {self._ndet_a * self._ndet_b}  "
              f"vs  basis: {self.dim}")

        # basis --> civec[ia, ib] local index
        self._basis_ia = np.searchsorted(ci_strs_a, all_strs_a).astype(np.int64)
        self._basis_ib = np.searchsorted(ci_strs_b, all_strs_b).astype(np.int64)

        # Defensive check: searchsorted does not raise on miss, must verify explicitly
        assert np.all(ci_strs_a[self._basis_ia] == all_strs_a), "α string mapping failed"
        assert np.all(ci_strs_b[self._basis_ib] == all_strs_b), "β string mapping failed"

        # absorb_h1e
        self._h2eff = direct_spin1.absorb_h1e(
            self.h1eff, self.eri_4d, self.ncas, self.nelecas, .5
        )

        # link_index
        print("[build] Precomputing link_index...", end=" ", flush=True)
        t0 = time.time()
        self._link_index = selected_ci._all_linkstr_index(
            self._ci_strs_tuple, self.ncas, self.nelecas
        )
        print(f"{time.time() - t0:.2f}s")

        # hdiag
        print("[build] Computing hdiag...", end=" ", flush=True)
        t0 = time.time()
        hdiag_full = selected_ci.make_hdiag(
            self.h1eff, self.eri_4d,
            self._ci_strs_tuple, self.ncas, self.nelecas
        )
        self._hdiag = hdiag_full.ravel()[
            self._basis_ia * self._ndet_b + self._basis_ib
        ]
        print(f"{time.time() - t0:.2f}s")

        # civec buffer (allocated once, reused throughout _hop)
        self._civec_buffer = np.zeros(
            (self._ndet_a, self._ndet_b), dtype=np.float64
        )

    def _vec_to_civec(self, v):
        self._civec_buffer.fill(0.0)
        self._civec_buffer[self._basis_ia, self._basis_ib] = v
        return self._civec_buffer

    def _civec_to_vec(self, civec):
        return civec[self._basis_ia, self._basis_ib].copy()

    def _hop(self, v):
        civec     = self._vec_to_civec(np.asarray(v, dtype=np.float64))
        civec_sci = _as_SCIvector(civec, self._ci_strs_tuple)
        hc = selected_ci.contract_2e(
            self._h2eff, civec_sci, self.ncas, self.nelecas,
            link_index=self._link_index
        )
        return self._civec_to_vec(np.array(hc))

    def _precond(self, dx, e, x0):
        denom = self._hdiag - e
        denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
        return dx / denom

    def _make_x0(self):
        bs    = self._occs_to_bitstring
        ref_a = bs(range(self.nelecas[0]))
        ref_b = bs(range(self.nelecas[1]))
        hf_ia = int(np.searchsorted(self._ci_strs_tuple[0], ref_a))
        hf_ib = int(np.searchsorted(self._ci_strs_tuple[1], ref_b))
        mask  = (self._basis_ia == hf_ia) & (self._basis_ib == hf_ib)
        hits  = np.where(mask)[0]
        x0    = np.zeros(self.dim, dtype=np.float64)
        if len(hits):
            x0[hits[0]] = 1.0
        else:
            x0[np.argmin(self._hdiag)] = 1.0
            print("[warn] HF determinant not in basis, falling back to hdiag minimum as initial guess")
        return x0

    def solve(self, nroots=1, tol=1e-10, max_cycle=1000, max_space=512,
              x0=None):
        """
        Run the Davidson solver.

        Returns
        -------
        e_ci : list[float]
            Active-space eigenvalue E_CI for each root (excluding e_core).
        civecs : list[ndarray]
            Normalized eigenvector for each root, each of shape (dim,).
        """
        if x0 is None:
            x0 = self._make_x0()

        print(f"\n[solve] Davidson  nroots={nroots} ...")
        t0 = time.time()

        converged, e_roots, e_vecs = lib.davidson1(
            lambda xs: [self._hop(x) for x in xs],
            x0, self._precond,
            tol=tol,
            max_cycle=max_cycle,
            max_space=max_space,
            nroots=nroots,
            verbose=lib.logger.DEBUG
        )

        print(f"[solve] Converged" if converged else '[solve] Not converged!!!')
        print(f"[solve] Elapsed: {time.time() - t0:.2f}s  ")

        self.converged = converged
        self.e_ci      = e_roots
        self.civecs    = e_vecs

        return self.e_ci, self.civecs
