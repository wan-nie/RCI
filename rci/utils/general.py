import os
import numpy as np
from . import solax_tools as tools
import clic_clib as qc
import clic
from .pyscf_solver import SelectedCISolver
from pyscf.tools import fcidump
from pyscf import ao2mo


class QCSolver:
    def __init__(self, H, H_half, h_mo, eri_mo, N_orb, diag_of, batch_sizes, rand_keys):
        """
        eri_mo: <ij|kl>
        """
        self.H = H
        self.H_half = H_half
        self.diag_of = diag_of
                
        self.N_orb = N_orb        
        self.h_mo = h_mo
        self.eri_mo = eri_mo
        self.eri_mo_chem = eri_mo.transpose(0, 2, 1, 3)
        
        # for solax diag
        self.batch_sizes = batch_sizes 
        self.rand_keys = rand_keys

        if diag_of != 'solax':
            self._prepare_clic()

        # trimci warm-start cache
        # (alpha, beta, coeffs) of the last of='trimci' solve; coeffs is (N, n_states).
        self._trimci_warm = None
    
    @classmethod
    def build(cls, integral_dir, integral_path,
              N_orb, N_spinorb, N_el,
              chop_thresh, diag_of, batch_sizes, rand_keys
              ):
        """
        Build a QCSolver by loading or computing integrals,
        then constructing the Hamiltonian operator.
        """
        daggers1p = tuple([1, 0])
        daggers2p = tuple([1, 1, 0, 0])

        # --- Load or compute integrals for PAW basis ---
        if integral_dir is not None:
            print(f'[Integral] load integral from {integral_dir}')
            h_mo = np.loadtxt(os.path.join(integral_dir, "H_nn.dat"))
            eri_mo_raw = np.loadtxt(os.path.join(integral_dir, "Uijkl.dat"))

            eri_mo_dict = {}
            for i, j, k, l, value in eri_mo_raw:
                eri_mo_dict[(int(i), int(k), int(j), int(l))] = value

            H_0   = tools.build_op_0(h_mo, N_orb, daggers1p)
            H_int = tools.build_op_int(eri_mo_dict, N_orb, daggers2p)
            Rho   = np.diag([1] * N_el + [0] * (N_spinorb - N_el))
            H_MF  = tools.build_op_MF(eri_mo_dict, Rho)
            H     = H_0 + H_int - H_MF

        elif integral_path is not None:
            print(f'[Integral] load integral from {integral_path}')
            
            if '.npz' in integral_path:
                data = np.load(integral_path)
                h_mo   = data['h_mo']   if 'h_mo'   in data else data['cas_h_mo_eff']
                eri_mo = data['eri_mo'] if 'eri_mo' in data else data['cas_eri_mo']
            else:
                data = fcidump.read(integral_path)
                h_mo = data['H1']
                eri_mo = ao2mo.restore(1, data['H2'], data['NORB'])

            eri_mo_dict = {
                (i, j, k, l): eri_mo[i, k, j, l]
                for i in range(N_orb)
                for j in range(N_orb)
                for k in range(N_orb)
                for l in range(N_orb)
            }

            H_0   = tools.build_op_0(h_mo, N_orb, daggers1p)
            H_int = tools.build_op_int(eri_mo_dict, N_orb, daggers2p)
            H     = H_0 + H_int

        # --- Post-process H ---
        H = tools.chop(H, chop_thresh)
        if 'scalar' in H:
            H = H.drop("scalar")
        H_half = tools.drop_herm_conj(H)

        # --- Rebuild dense eri_mo from dict ---
        h_mo   = h_mo[:N_orb, :N_orb]
        eri_mo = np.zeros((N_orb, N_orb, N_orb, N_orb), dtype=float)
        for i in range(N_orb):
            for j in range(N_orb):
                for k in range(N_orb):
                    for l in range(N_orb):
                        eri_mo[i, j, k, l] = eri_mo_dict[(i, j, k, l)]

        return cls(H, H_half, h_mo, eri_mo, N_orb, diag_of, batch_sizes, rand_keys)
    
    def _prepare_clic(self, toltables=1e-12):
        N_orb = self.N_orb
        h_mo = self.h_mo
        eri_mo = self.eri_mo

        #
        h_so = clic.double_h(h_core=h_mo, M=N_orb)
        h_so = np.ascontiguousarray(h_so, dtype=np.complex128)
        self.h_so = h_so

        eri_so = clic.umo2so(eri_mo, M=N_orb)
        eri_so = np.ascontiguousarray(eri_so, dtype=np.complex128)
        self.eri_so = eri_so

        #
        self.tables = qc.build_hamiltonian_tables(h_so, eri_so, toltables)
    
    def _solax_to_qc(self, basis):
        bits = basis.to_bits()

        #
        alpha_bits = bits[:, ::2]
        beta_bits = bits[:, 1::2]
        
        n_alpha = int(alpha_bits[0].sum())
        n_beta = int(beta_bits[0].sum())
        alpha_indices = np.argsort(alpha_bits, axis=1, kind='stable')[:, -n_alpha:]
        beta_indices = np.argsort(beta_bits, axis=1, kind='stable')[:, -n_beta:]
        
        alpha_indices.sort(axis=1)
        beta_indices.sort(axis=1)

        # generate qc basis
        def bs(occs):
            s = 0
            for k in occs:
                s |= (1 << k)
            return s
    
        basis = []
        alpha_beta = []
        for i in range(len(bits)):
            basis.append(
                qc.SlaterDeterminant(self.N_orb, alpha_indices[i], beta_indices[i])
            )
            alpha_beta.append([bs(alpha_indices[i]), bs(beta_indices[i])])
        
        alpha_beta = np.array(alpha_beta, dtype=np.uint64)
        return basis, alpha_beta
    
    def shrink_basis_clic(self, basis, basis_sub, hm):
        basis, _ = self._solax_to_qc(basis)
        basis_sub, _ = self._solax_to_qc(basis_sub)
    
        # basis -> index mapping
        index_map = {b: i for i, b in enumerate(basis)}
    
        idx_ls = np.array([index_map[b] for b in basis_sub], dtype=int)
    
        # submatrix
        H_sub = hm[idx_ls][:, idx_ls]
    
        return H_sub

    def _build_initial_guess_davidson(self, basis, H_diag, pre_data=None, diag_k=100_000):
        basis_index = {sd: i for i, sd in enumerate(basis)}
        dim = len(basis)
        if dim == 0:
            raise ValueError("Empty basis in build_initial_guess_davidson.")

        x0 = np.zeros(dim, dtype=np.float64)

        k = min(diag_k, dim)
        idx_diag = np.argsort(H_diag)[:k]
        basis_diag = [basis[i] for i in idx_diag]

        hm = clic.get_ham(basis_diag, self.h_so, self.eri_so, tables=self.tables)
        hm = hm.real
        _, vecs = clic.diagH(hm, num_roots=1, option='arpack')

        v_diag = np.asarray(vecs.real[:, 0], dtype=np.float64)
        x0[idx_diag] = v_diag

        if pre_data is not None:
            basis_pre, _ = self._solax_to_qc(pre_data[0])
            v_pre = pre_data[1].reshape(-1)

            x0_pre = np.zeros(dim, dtype=np.float64)
            for sd, amp in zip(basis_pre, v_pre):
                idx = basis_index.get(sd, None)
                if idx is not None:
                    x0_pre[idx] = np.real(amp)

            overlap = np.dot(x0[idx_diag], x0_pre[idx_diag])
            if overlap < 0:
                x0_pre = -x0_pre

            not_covered = np.ones(dim, dtype=bool)
            not_covered[idx_diag] = False
            x0[not_covered] = x0_pre[not_covered]

        nrm = np.linalg.norm(x0)
        if nrm < 1e-14:
            x0[:] = 0.0
            x0[np.argmin(H_diag)] = 1.0
        else:
            x0 /= nrm

        return x0

    def get_roots(self, basis, basis_base=None, hm_base=None, pre_data=None, diag_of=None):
        if diag_of is None:
            diag_of = self.diag_of
        
        if diag_of == 'solax':
            H_half = self.H_half
            batch_sizes = self.batch_sizes
            rand_keys = self.rand_keys
            
            if basis_base is None:
                hm = H_half.build_matrix(basis, **batch_sizes)
                hm += hm.hconj
                evals, evecs = tools.find_lowest_states(hm.to_scipy(), 1, next(rand_keys))
            else:
                hm = tools.update_matrix(H_half, hm_base, old_basis=basis_base, new_basis=basis % basis_base, **batch_sizes)
                evals, evecs = tools.find_lowest_states(hm.to_scipy().tocsr(), 1, next(rand_keys))

            return evals, evecs, hm
            
        elif diag_of == 'clic':
            basis_clic, _ = self._solax_to_qc(basis)
            hm = clic.get_ham(basis_clic, self.h_so, self.eri_so, tables=self.tables)
            hm = hm.real
            evals, evecs = clic.diagH(hm, num_roots=1, option="arpack")
            
            return evals, evecs.real, hm
        
        elif diag_of == 'davidson':
            basis_clic, _ = self._solax_to_qc(basis)
            sd = basis_clic[0]
            nelec = (len(sd.alpha_occupied_indices()), len(sd.beta_occupied_indices()))
            solver = SelectedCISolver(basis_clic, self.h_mo, self.eri_mo_chem, self.N_orb, nelec)
            
            #
            print('[build] x0 generating')
            x0 = self._build_initial_guess_davidson(basis_clic, solver._hdiag, pre_data=pre_data)
            print('[build] x0 generated')
            evals, evecs = solver.solve(x0=x0)
            evecs = np.expand_dims(evecs[0], axis=-1)

            return evals, evecs, None

        elif diag_of == 'trimci':
            from trimci import trimci_core
            fe = trimci_core.fast_expansion

            # (N, 2) -> 1-D alpha/beta
            _, alpha_beta = self._solax_to_qc(basis)
            alpha = np.ascontiguousarray(alpha_beta[:, 0], dtype=np.uint64)
            beta  = np.ascontiguousarray(alpha_beta[:, 1], dtype=np.uint64)
            N = len(basis)

            # chemist (ij|kl), flattened n_orb^4
            h1  = np.ascontiguousarray(self.h_mo, dtype=np.float64)
            eri = np.ascontiguousarray(self.eri_mo_chem, dtype=np.float64).ravel()

            diag = fe.compute_diagonals(alpha, beta, h1, eri, self.N_orb)

            params = fe.DavidsonParams()
            params.n_states = 1
            params.max_subspace = 60

            def map_coeffs_to_basis(src_alpha, src_beta, src_v, alpha, beta):
                """Map coefficients from a source det list onto (alpha, beta) by det identity.
                
                hit -> old coeff, miss -> 0, then per-row normalize. 
                src_v may be (N_src,) or (N_src, n). N_src: number of SDs, n: number of roots
                Returns (n, N) guess array, or None if every row is zero.
                """
                key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(alpha, beta))}
                N_before = src_alpha.shape[0]
                N = alpha.shape[0]
                src_v = np.asarray(src_v).real
                if src_v.ndim == 1:
                    src_v = src_v[:, None]  # (N, 1)
                
                #
                n = src_v.shape[1]  # num roots
                out = np.zeros((n, N), dtype=np.float64)
                for i in range(N_before):
                    idx = key.get((int(src_alpha[i]), int(src_beta[i])))
                    if idx is not None:
                        out[:, idx] = src_v[i]
                rows = [out[s] / np.linalg.norm(out[s])
                        for s in range(n) if np.linalg.norm(out[s]) > 1e-14]
                
                return np.asarray(rows, dtype=np.float64) if rows else None
    
            guess = np.zeros((0,), dtype=np.float64)
            if self._trimci_warm is not None:
                pa, pb, pc = self._trimci_warm
                g = map_coeffs_to_basis(pa, pb, pc, alpha, beta)
                if g is not None:
                    guess = g[:1]

            if guess.size == 0 and N > 0:
                x0 = np.zeros(N, dtype=np.float64)
                x0[int(np.argmin(diag))] = 1.0
                guess = x0.reshape(1, N)

            # solve
            res = fe.davidson_solve_matfree(alpha, beta, h1, eri, self.N_orb, params, guess)
            evals = np.asarray(res.eigenvalues, dtype=np.float64)
            evecs = np.asarray(res.eigenvectors, dtype=np.float64).T  # (N, n_states)

            # cache for next iteration's warm start
            self._trimci_warm = (alpha, beta, evecs)

            return evals, evecs, None

def basis_to_array(basis):
    x = []
    for b in basis:
        x.append(b.to_bits())
    if len(x) > 0:
        x = np.concatenate(x, axis=0)
    else:
        x = []
    return x


def encoding_to_bitstring(arr, target_length):
    bit_strings = [format(num, '08b') for num in arr]
    combined_bits = ''.join(bit_strings)
    combined_bits = combined_bits[:target_length]
    return combined_bits


def random_sample_basis(basis_pool, target_num):
    sampled = np.random.choice(range(len(basis_pool)), target_num, replace=False)
    return basis_pool[sampled]


def derive_abs_coeff_cut(impt_frac, rand_substate) -> float:
    abs_coeff_srt = np.sort(
        np.abs(rand_substate.coeffs)
    )[::-1]
    impt_num = int(impt_frac * len(abs_coeff_srt))
    abs_coeff_cut = (abs_coeff_srt[impt_num - 1] + abs_coeff_srt[impt_num]) / 2   
    return abs_coeff_cut


def print_state_info(state, hi_thr, lo_thr, title):
    coeffs_abs = np.abs(state.coeffs)
    n_high = (coeffs_abs > hi_thr).sum()
    n_mid = ((coeffs_abs <= hi_thr) & (coeffs_abs >= lo_thr)).sum()
    n_low = (coeffs_abs < lo_thr).sum()
    print('{:<10} {:>15} all; {:>15} high ({:6.2%}); {:>15} mid ({:6.2%}); {:>15} low ({:6.2%})'.format(title, len(state),
                                                                    n_high, n_high/len(state), 
                                                                    n_mid, n_mid/len(state), 
                                                                    n_low, n_low/len(state)))
    return n_high, n_mid, n_low
    
