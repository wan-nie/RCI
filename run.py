import os
TIMEOUT_DAYS = 10
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = str(TIMEOUT_DAYS * 24 * 60 * 60)
import sys
import shutil
import random
import copy
import time
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional
from scipy.stats import spearmanr
import argparse
from datetime import timedelta

import jax
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
torch.set_float32_matmul_precision('high')

import solax as sx
import clic
from rci.models import BitModelTransformerTwoChannels, BlockWeightedPairSampler
from rci.utils import (QCSolver, basis_to_array, encoding_to_bitstring, 
                   random_sample_basis, derive_abs_coeff_cut, print_state_info)
import rci.utils.solax_tools as tools


def setup_hybrid_ddp() -> Tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(days=TIMEOUT_DAYS),
            device_id=torch.device(f"cuda:{local_rank}")
        )
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    gloo_group = dist.new_group(
        ranks=list(range(world_size)),
        backend="gloo",
        timeout=timedelta(days=TIMEOUT_DAYS)
    )

    # --- JAX ---
    if rank == 0:
        jax_devices = jax.devices()
        print(f"[Rank 0] JAX visible devices: {jax_devices}")
    else:
        jax.config.update("jax_platform_name", "cpu")

    # --- PyTorch ---
    num_gpus = torch.cuda.device_count()
    torch_device_id = local_rank
    
    if torch_device_id >= num_gpus:
        raise RuntimeError(f"Rank {rank} trying to use GPU {torch_device_id}, but only {num_gpus} visible.")

    torch.cuda.set_device(torch_device_id)
    device = torch.device("cuda", torch_device_id)
    
    print(f"[Rank {rank}] PyTorch using device: {torch_device_id}")
    
    return rank, local_rank, world_size, device, gloo_group

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

# --- Training ---

class Trainer:
    def __init__(self, args, model: nn.Module):
        self.args = args
        self.model = model
        self.rank = dist.get_rank()
        self.device = args.device
        self.criterion = nn.Softplus()

    def _compute_scores(self, x_gpu: torch.Tensor, input_indices: np.ndarray) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if input_indices.ndim == 1:
            uniq_idx = torch.as_tensor(input_indices, dtype=torch.long)
            inverse = None
        elif input_indices.ndim == 2:
            pairs = torch.as_tensor(input_indices, dtype=torch.long)
            uniq_idx, inverse = torch.unique(pairs, sorted=True, return_inverse=True)
        else:
            raise ValueError(f'Unknown input shape {input_indices.shape}')
        
        score_ls = []
        num_samples = len(uniq_idx)
        batch_size = self.args.batch_size
        
        for i in range(0, num_samples, batch_size):
            batch_idx = uniq_idx[i : i + batch_size]
            batch_x = x_gpu[batch_idx]
            batch_x = batch_x.to(self.device)
            with autocast(device_type='cuda'):
                score = self.model(batch_x).flatten()
            score_ls.append(score)
        
        s = torch.cat(score_ls, dim=0) if score_ls else torch.tensor([], device=self.device)

        if input_indices.ndim == 1:
            return s.detach().cpu().numpy(), None
        else:
            return s, inverse

    def train(self, x_train, y_train_np, x_rand, y_rand_np, n_sel, optimizer, iter_out_dir, seed):
        sampler_seed = seed + self.rank
        
        num_bins = 64
        sampler = BlockWeightedPairSampler(y_train_np, num_bins=num_bins, seed=sampler_seed)
        
        best_spearman = -float('inf')
        best_state = None
        no_improve_count = 0
        eval_interval = self.args.eval_interval

        def scores_fn_wrapper(indices):
            with torch.no_grad():
                s, _ = self._compute_scores(x_train, indices)
            return s

        scaler = GradScaler()
        for step in range(self.args.epochs):
            self.model.train()

            # update W
            if step % eval_interval == 0:
                plot_path = None
                if self.rank == 0 and step % (eval_interval * 20) == 0:
                    weights_dir = os.path.join(iter_out_dir, 'sample_weights')
                    os.makedirs(weights_dir, exist_ok=True)
                    plot_path = os.path.join(weights_dir, f'{step}.csv')

                probe_k = min(100, n_sel // (num_bins**2))
                sampler.refresh_block_weights(
                    scores_fn=scores_fn_wrapper,
                    probe_k=probe_k,
                    plot_path=plot_path
                )

            # sampling
            pairs = sampler.sample_weighted(n_sel, active_idx_ratio=0.2, max_active_num=self.args.max_active_num)
            num_pairs = len(pairs)
            
            # compute scores of unique determinants
            s, inverse = self._compute_scores(x_train, pairs)
            del pairs

            # Pairwise Loss
            total_loss = 0.0
            pairwise_batch_size = int(1e7)
            
            for start in range(0, num_pairs, pairwise_batch_size):
                end = start + pairwise_batch_size
                i_pos = inverse[start:end, 0]
                j_pos = inverse[start:end, 1]
                
                with autocast(device_type='cuda'):
                    diff = s[i_pos] - s[j_pos]
                    chunk_loss = self.criterion(-diff).sum()
                total_loss += chunk_loss
            
            with autocast(device_type='cuda'):
                loss = total_loss / num_pairs

            # back propagation
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # eval and early stopping
            if (step + 1) % eval_interval == 0:
                self.model.eval()
                rho = 0.0
                avg_loss = 0.0
                                
                if self.rank == 0:
                    self.model.eval()
                    with torch.no_grad():
                        s_eval, _ = self._compute_scores(x_rand, np.arange(len(x_rand)))
                    rho = spearmanr(s_eval, y_rand_np).correlation
                
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                avg_loss = loss.item()

                stop_signal = torch.zeros(1, device=self.device)
                if self.rank == 0:
                    if (step + 1) % (eval_interval * 20) == 0:
                        print(f"step {step+1}, loss={avg_loss:.4f}, spearman={rho:.4f}, num_pairs={num_pairs}")

                    if rho > best_spearman + 1e-6:
                        best_spearman = rho
                        best_state = copy.deepcopy(self.model.module.state_dict())
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    if no_improve_count >= self.args.es_patience:
                        print(f"Early stopping at step {step+1}. Best Spearman={best_spearman:.4f}")
                        stop_signal[0] = 1.0
                
                dist.broadcast(stop_signal, src=0)
                if stop_signal.item() > 0.5:
                    break

        if self.rank == 0 and best_state is not None:
            self.model.module.load_state_dict(best_state)
            torch.save(self.model.module, os.path.join(iter_out_dir, "best_model.pt"))
        
        dist.barrier()
        return self.model

# --- Inference ---
@torch.no_grad()
def forward_scores(model, loader, device):
    model.eval()
    score_ls = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        pred = model(x)
        pred = pred.detach().flatten()
        score_ls.append(pred)
    return torch.cat(score_ls, axis=0)

def inference_get_top_k(args, model, x_pool, top_k):
    pool_loader = DataLoader(TensorDataset(x_pool), batch_size=args.batch_size, shuffle=False, pin_memory=True)
    all_scores = forward_scores(model, pool_loader, args.device)
    
    final_indices = []
    if dist.get_rank() == 0:
        k = min(top_k, len(all_scores))
        _, top_indices = torch.topk(all_scores, k)
        final_indices = top_indices.cpu().tolist()
        
    return final_indices

def printr(*args):
    rank = dist.get_rank()
    if rank == 0:
        print(*args)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

# --- Main ---
def parse_args():
    parser = argparse.ArgumentParser(description='Train neural network model for quantum chemistry')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs (default: 5000)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training (default: cuda)')
    parser.add_argument('--es_patience', type=int, default=5, help='Early stopping patience (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')

    parser.add_argument('--seed', type=int, default=1234, help='Random seed (default: 1234)')
    parser.add_argument('--n_iters', type=int, default=10, help='Number of iterations (default: 10)')
    parser.add_argument('--max_sel_pairs', type=int, default=2, help='Maximum number of selected pairs per epoch, in units of 1e8')
    parser.add_argument('--batch_size', type=int, default=8192, help='NN model batch size')
    parser.add_argument('--compile_mode', type=str, default='default', help='torch.compile compile mode')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--max_active_num', type=int, default=None, help='Maximum number of activated determinants to sample pairs from')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden size for both the orbital embedding and the Transformer encoder d_model')

    # System parameters
    parser.add_argument('--mol_name', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--integral_dir', type=str, default=None)
    parser.add_argument('--integral_path', type=str, default=None)
    parser.add_argument('--distance', type=str, default=None)
    parser.add_argument("--basis_name", type=str, default=None)
    parser.add_argument('--N_orb', type=int, help='Number of orbitals')
    parser.add_argument('--impt_expand', type=float, help='Importance expansion factor gamma')
    parser.add_argument('--max_impt_frac', type=float, default=0.2)
    parser.add_argument('--diag_of', type=str, default='solax', choices=['solax', 'clic', 'davidson'])

    # Continue iteration if applicable
    parser.add_argument('--n_from', type=int, default=-1, help='Resume iteration from this index. Use -1 to start from the beginning (default).')
    parser.add_argument('--n_from_root', type=str, default=None, help='Root directory of a previous run to resume from. Must be set together with --n_from.')
        
    args = parser.parse_args()
    return args

def main():
    # setup DDP
    rank, local_rank, world_size, device, gloo_group = setup_hybrid_ddp()
    
    #
    args = parse_args()
    args.device = device
    
    printr(f"Configuration: {args}")

    # seed
    rand_keys = None
    if rank == 0:
        rand_keys = sx.RandomKeys(seed=args.seed)
    set_seed(args.seed + rank)

    # molecule
    mol_name = args.mol_name
    if mol_name in ['N2', 'CO']: N_el = 10
    elif mol_name in ['NH3', 'H2O', 'C2']: N_el = 8
    elif mol_name == 'Fe2S2': N_el = 30
    else: raise Exception(f'Unknown molecule {mol_name}')
    
    N_orb = args.N_orb
    N_spinorb = N_orb * 2
        
    #
    interval = int(1024 * 4 * 2)
    batch_sizes = dict(det_batch_size=interval, op_batch_size=interval)
    printr(batch_sizes)

    if rank == 0:
        printr("Initializing Physics on Rank 0...")
        qcsolver = QCSolver.build(
            integral_dir=args.integral_dir,
            integral_path=args.integral_path,
            N_orb=N_orb,
            N_spinorb=N_spinorb,
            N_el=N_el,
            chop_thresh=1e-11,
            diag_of=args.diag_of,
            batch_sizes=batch_sizes,
            rand_keys=rand_keys
        )
        H = qcsolver.H
        O_ext = tools.chop(H, 2e-2)
        print(f"""\
            Length of operators:
            H : {tools.full_len(H)}
            O_ext : {tools.full_len(O_ext)}
            """)
        
        basis_init = sx.Basis(['1' * N_el + '0' * (N_spinorb - N_el)])
        HF_gs = sx.State(basis_init, np.ones(1))
        E_HF = HF_gs * H(HF_gs, **batch_sizes, multiple_devices=True)
        printr(f"HF energy:\t\t{E_HF}")

        #
        if args.n_from == -1:
            basis_pre_ = basis_init          
            for _ in range(2):
                basis_pre_ = basis_pre_ + O_ext(basis_pre_, **batch_sizes, multiple_devices=True)
            Es, Vs_core, hm_core = qcsolver.get_roots(basis_pre_)

            basis_core = basis_pre_
            basis_selected = basis_pool = None
        else:
            def load_from_n(from_dir, N_spinorb):
                def load_b(name):
                    enc = np.load(f'{from_dir}/basis/{name}/.attrs/_encoding.npy')
                    lst = [encoding_to_bitstring(x, N_spinorb) for x in enc]
                    return sx.Basis(lst)
                return load_b('final_state/.attrs/basis'), load_b('selected'), load_b('pool')
            
            from_dir = f'{args.n_from_root}/{args.n_from}'
            basis_core, basis_selected, basis_pool = load_from_n(from_dir, N_spinorb)
            Es, Vs_core, hm_core = qcsolver.get_roots(basis_core)

        print("*********")
        print(f"Dim:\t{len(basis_core)}")
        print(f"Corr energy (core):\t\t{Es[0] - E_HF}")

    # initialize model and compile
    model = BitModelTransformerTwoChannels(N_orb, N_elec=N_el, hidden_dim=args.hidden_dim)
    model = model.to(args.device)

    # load parameters
    if args.n_from >= 0:
        from_dir = f'{args.n_from_root}/{args.n_from}'
        model_path = f'{from_dir}/best_model.pt'
        saved_model = torch.load(model_path, map_location=args.device, weights_only=False)
        
        if hasattr(saved_model, 'module'):
            state_dict = saved_model.module.state_dict()
        else:
            state_dict = saved_model.state_dict()

        # remove "_orig_mod." prefix if any
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('_orig_mod.'):
                new_key = key.replace('_orig_mod.', '')
            else:
                new_key = key
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
        printr(f"All ranks loaded pretrained model from {from_dir}")

    # compile
    printr(f"Compiling model with mode={args.compile_mode}...")
    model = torch.compile(model, mode=args.compile_mode)

    # DDP
    model = DDP(model, device_ids=[local_rank])
    printr(model)
    dist.barrier()
    
    trainer = Trainer(args, model)

    # Iteration
    start = args.n_from + 1
    end = args.n_iters

    for i in range(start, end):
        printr(f"\n=== Iteration {i} start: {datetime.now()} ===")

        # --- Phase 1: Physics (Rank 0, JAX Multi-GPU) ---        
        if rank == 0:
            if basis_selected:
                basis = basis_core + basis_pool + O_ext(basis_selected, **batch_sizes, multiple_devices=True)
            else:
                basis = basis_core + O_ext(basis_core, **batch_sizes, multiple_devices=True)
            
            printr(f"New total space size:\t{len(basis)}")

            basis_pool = basis % basis_core
            printr(f"Core size:\t\t{len(basis_core)}")
            printr(f"Pool size:\t\t{len(basis_pool)}")

            #
            target_num        = int(np.sqrt(len(basis_pool)) * 50)
            impt_frac         = target_num / len(basis_pool)
            impt_frac_expand       = max(
                impt_frac,
                min(args.max_impt_frac, impt_frac * (args.impt_expand ** i))
            )
            print('impt_frac:\t\t', impt_frac)
            print('impt_frac_expand:\t\t', impt_frac_expand)
            
            random_frac = impt_frac * 2/1.5
            train_sample_size = int(random_frac * len(basis_pool))
            printr(f"Rand sample:\t\t{train_sample_size}")
            printr(f"rand_frac:\t\t{random_frac}")
            
            rand_train_basis = random_sample_basis(basis_pool, train_sample_size)
            basis_for_diag = basis_core + rand_train_basis
            
            pre_data = [basis_core, Vs_core]
            Es, Vs, _ = qcsolver.get_roots(basis_for_diag, basis_base=basis_core, hm_base=hm_core, pre_data=pre_data)
            
            train_state = sx.State(basis_for_diag, Vs[:, 0])
            rand_train_state = train_state % basis_core
            
            impt_frac = target_num / len(basis_pool)
            abs_coeff_cut = derive_abs_coeff_cut(impt_frac, rand_train_state)
            rand_train_state_impt = rand_train_state.chop(abs_coeff_cut)
            
            # print state info
            core_train_state = train_state % rand_train_basis
            printr(len(rand_train_basis), Es[0] - E_HF)
            high_coeff = np.abs(rand_train_state.coeffs).max()
            low_coeff = abs_coeff_cut
            print_state_info(core_train_state, hi_thr=high_coeff, lo_thr=low_coeff, title='Core')
            print_state_info(rand_train_state, hi_thr=high_coeff, lo_thr=low_coeff, title='Rand')
            print_state_info(train_state, hi_thr=high_coeff, lo_thr=low_coeff, title='All')

        # --- Phase 2: Broadcast ---
        if rank == 0:
            dist_objs = [train_state, rand_train_state]
        else:
            dist_objs = [None, None]

        # dist.broadcast_object_list(dist_objs, src=0)
        dist.broadcast_object_list(dist_objs, src=0, group=gloo_group)
        train_state = dist_objs[0]
        rand_train_state = dist_objs[1]

        x_train = torch.tensor(basis_to_array(train_state.basis), dtype=torch.uint8)
        y_train_np = np.abs(train_state.coeffs, dtype=np.float32)

        x_rand = torch.tensor(basis_to_array(rand_train_state.basis), dtype=torch.uint8)
        y_rand_np = np.abs(rand_train_state.coeffs)
        
        # --- Phase 2: DDP Training ---
        iter_out_dir = os.path.join(f'{args.out_dir}/{i}')
        if rank == 0:
            if os.path.exists(iter_out_dir): shutil.rmtree(iter_out_dir)
            os.makedirs(iter_out_dir)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Trainer
        n_total = len(x_train)
        n_pairs = int(n_total * (n_total-1) / 2)
        # n_sel = int(min(n_pairs / 500, 1e8))
        n_sel = int(min(n_pairs / 500, 1e8*args.max_sel_pairs))
        n_sel = n_sel // world_size
        printr(f'N_select_pairs for each epoch:', n_sel)

        printr('Train start:', datetime.now())
        model = trainer.train(
            x_train, y_train_np, 
            x_rand, y_rand_np,
            n_sel=n_sel,
            optimizer=optimizer, 
            iter_out_dir=iter_out_dir, 
            seed=args.seed+i
        )
        printr('Train end:', datetime.now())
        
        if rank == 0:
            # --- Phase 3: Inference ---
            basis_pool_sub = basis_pool % rand_train_basis            
            nn_sel_num = int(impt_frac_expand * len(basis_pool_sub))

            x_pool = torch.tensor(basis_to_array(basis_pool_sub), dtype=torch.uint8)
            nn_sel_indices = inference_get_top_k(
                args, 
                model, 
                x_pool=x_pool, 
                top_k=nn_sel_num
            )
            del x_pool
            printr('Inference end:', datetime.now())

        torch.cuda.empty_cache()
        
        if rank == 0:
            # --- Phase 4: Update Physics (Rank 0) ---
            nn_predicted = basis_pool_sub[nn_sel_indices]
            basis_for_diag = basis_core + rand_train_state_impt.basis + nn_predicted
            
            pre_data = [train_state.basis, train_state.coeffs]
            Es, Vs, hm = qcsolver.get_roots(basis_for_diag, basis_base=basis_core, hm_base=hm_core, pre_data=pre_data)
            
            print(f"Core size:\t\t{len(basis_core)}")
            print(f"Random impt:\t\t{len(rand_train_state_impt)}")
            print(f"NN-selected:\t\t{len(nn_predicted)}")
            print(f"In total accepted:\t{len(nn_predicted) + len(rand_train_state_impt)}\t of {target_num} (target num)")
            print(f"Fin diag on:\t\t{len(basis_for_diag)}")
            print(f"Corr energy (before chopping):\t\t{Es[0] - E_HF}")
            
            # chop false positive
            final_state_chop = sx.State(basis_for_diag, Vs[:, 0]) % basis_core
            final_state_prim = sx.State(basis_for_diag, Vs[:, 0]) % (basis_for_diag % basis_core)
            final_state = (final_state_prim + final_state_chop.chop(abs_coeff_cut)).normalize()
            
            # print info of selected
            print("Most important newly accepted determinants: ")
            basis_selected = final_state % basis_core
            tools.print_leading_coefficients(basis_selected, 5)
            basis_selected = basis_selected.basis
            print(f"Fin size:\t\t{len(final_state)}")
            print(f"Were cut:\t\t{len(basis_for_diag) - len(final_state)}")
            print(f"In total selected:\t{len(basis_selected)}\t of {target_num} (target num)")

            #
            if hm is not None:
                if args.diag_of == 'solax':
                    hm = hm.shrink_basis(basis_for_diag, final_state.basis)
                    Es, Vs = tools.find_lowest_states(hm.to_scipy().tocsr(), 1, next(rand_keys))
                elif args.diag_of == 'clic':
                    hm = qcsolver.shrink_basis_clic(basis_for_diag, final_state.basis, hm)
                    Es, Vs = tools.find_lowest_states(hm, 1, next(rand_keys))
            else:
                pre_data = [final_state.basis, final_state.coeffs]
                Es, Vs, hm = qcsolver.get_roots(final_state.basis, pre_data=pre_data)
                
            print(f'Corr energy (after chopping):\t\t{Es[0] - E_HF}')

            # update
            hm_core = hm
            basis_core = final_state.basis
            Vs_core = Vs

            # save results
            to_be_saved = dict(
                final_state=final_state,
                selected=basis_selected,
                pool=basis_pool
            )

            sx.save(to_be_saved, os.path.join(iter_out_dir, 'basis'))
            print(f"=== Iteration {i} end: {datetime.now()} ===")

        # dist.barrier()
        dist.barrier(group=gloo_group)

    cleanup_ddp()

if __name__ == '__main__':
    main()