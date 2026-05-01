import os
import subprocess
import sys
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Run molecular calculations in series')
    parser.add_argument('--cuda_devices', type=str, default='0,1',
                        help='CUDA devices to use, comma-separated (default: 0,1)')
    parser.add_argument('--num_threads', type=int, default=36,
                        help='Number of threads for CPU operations (default: 36)')
    parser.add_argument('--base_port', type=int, default=49500,
                        help='Base port for distributed training (default: 49500)')
    parser.add_argument('--impt_expand', type=float, default=1.5)
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--diag_of', type=str, default='davidson', choices=['solax', 'clic', 'davidson'])
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
    
    mol_names = ['Fe2S2']
    orbs = [20]
    n_iters = args.n_iters
    IMPT_EXPAND = args.impt_expand
    SCRIPT = 'run.py'
    
    total_start_time = time.time()
    
    print(f"\nConfiguration:")
    print(f"  CUDA devices: {args.cuda_devices} ({num_gpus} GPU(s))")
    print(f"  CPU threads: {args.num_threads}")
    print(f"  Base port: {args.base_port}")
    
    for idx, (mol, N_orb) in enumerate(zip(mol_names, orbs)):
        current_port = args.base_port + idx
        
        out_dir = f"./result/Fe2S2"
        
        if os.path.exists(out_dir):
            print(f"\nDirectory {out_dir} already exists.")
        else:
            os.makedirs(out_dir, exist_ok=True)
            print(f"\nCreated directory {out_dir}")
        
        log_file = f"{out_dir}/out.log"
        
        print(f"\n{'='*60}")
        print(f"Starting job {idx+1}/{len(mol_names)} for {mol} with {N_orb} orbitals (port={current_port})...")
        print(f"{'='*60}")
        
        # 构建命令参数
        cmd = [
            'torchrun',
            '--nproc_per_node', str(num_gpus),
            '--master_port', str(current_port),
            SCRIPT,
            '--mol_name', mol,
            '--N_orb', str(N_orb),
            '--out_dir', out_dir,
            '--impt_expand', str(IMPT_EXPAND),
            '--n_iters', str(n_iters),
            '--max_active_num', str(100000),
            '--integral_path', './data/Fe2S2/fe2s2',
            '--diag_of', args.diag_of
        ]
                
        task_start_time = time.time()
        
        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT
            )
            
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