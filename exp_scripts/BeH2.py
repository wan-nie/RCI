import os
import subprocess
import time
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Run molecular calculations in series')
    parser.add_argument('--cuda_devices', type=str, default='0,1',
                        help='CUDA devices to use, comma-separated')
    parser.add_argument('--num_threads', type=int, default=36,
                        help='Number of threads for CPU operations')
    parser.add_argument('--base_port', type=int, default=42310,
                        help='Base port for distributed training')
    parser.add_argument('--n_iters', type=int, default=10)
    
    #
    parser.add_argument('--data_root', type=str, default='./data/BeH2')
    parser.add_argument('--out_root', type=str, default='./result/BeH2')
    parser.add_argument('--idx', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    
    num_gpus = len(args.cuda_devices.split(','))
    
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads)
    os.environ['MKL_NUM_THREADS'] = str(args.num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(args.num_threads)
    
    os.chdir('..')
    
    # config
    mol = 'BeH2'
    input_ls = [
        (13, f'{args.data_root}/BeH2_6-31g.fcidump'),
        (24, f'{args.data_root}/BeH2_cc-pvdz.fcidump'),
        (58, f'{args.data_root}/BeH2_cc-pvtz.fcidump')
    ]
    n_iters = args.n_iters
    SCRIPT = 'run.py'
    
    total_start_time = time.time()
    
    print(f"\nConfiguration:")
    print(f"  CUDA devices: {args.cuda_devices} ({num_gpus} GPU(s))")
    print(f"  CPU threads: {args.num_threads}")
    print(f"  Base port: {args.base_port}")
    
    idx = args.idx
    N_orb, integral_path = input_ls[idx]
         
    current_port = args.base_port + idx
    out_dir = f"{args.out_root}/{N_orb}"  # result/BeH2/N_orb
    if os.path.exists(out_dir):
        print(f"\nDirectory {out_dir} already exists.")
    else:
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nCreated directory {out_dir}")
    
    log_file = f"{out_dir}/out.log"
    
    print(f"\n{'='*60}")
    print(f"Starting job for {mol} with {N_orb} orbitals (port={current_port})...")
    print(f"{'='*60}")
    
    # build cmd          
    cmd = [
        'torchrun',
        '--nproc_per_node', str(num_gpus),
        '--master_port', str(current_port),
        SCRIPT,
        '--mol_name', mol,
        '--N_orb', str(N_orb),
        '--out_dir', out_dir,
        '--max_sel_pairs', str(num_gpus),
        '--n_iters', str(n_iters),
        '--max_active_num', str(200000),
        '--hidden_dim', str(256),
        '--integral_path', integral_path,
        '--extend_thd', str(0.01),  # a low threshold for small basis set
        '--impt_expand', str(1.3),
        '--diag_of', str('clic')
    ]
            
    task_start_time = time.time()
    
    with open(log_file, 'w') as log_f:
        process = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT
        )
        
        print(f"Start time: {datetime.now()}")
        print(f"Job running with PID {process.pid} for {mol}")
        print(f"Log file: {log_file}")
        
        return_code = process.wait()
    
    task_duration = time.time() - task_start_time
    
    if return_code == 0:
        print(f"\n✓ Job for {mol} completed successfully in {task_duration:.2f} seconds")
    else:
        print(f"\n✗ Job for {mol} failed with return code {return_code}")
        print(f"  Check log file: {log_file}")

    total_duration = time.time() - total_start_time
    
    print(f"\n{'='*60}")
    print(f"All jobs completed!")
    print(f"Total time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()