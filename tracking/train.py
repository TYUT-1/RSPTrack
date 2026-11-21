"""Train the full RSPTrack model by default."""
import argparse
import subprocess
import sys
import _init_paths
from lib.local_paths import PROJECT_ROOT, load_paths
from lib.project import DEFAULT_CONFIG, canonical_experiment

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--script', default='rsptrack')
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--save_dir', default=load_paths()['save_dir'])
    parser.add_argument('--mode', choices=['single', 'multiple'], default='single')
    parser.add_argument('--nproc_per_node', type=int, default=1)
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.script, args.config = canonical_experiment(args.script, args.config)
    if args.script != 'rsptrack':
        parser.error('Only rsptrack is supported')
    if args.nproc_per_node < 1:
        parser.error('--nproc_per_node must be positive')
    return args

def main():
    args = parse_args()
    cmd = [sys.executable]
    if args.mode == 'multiple':
        cmd += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(args.nproc_per_node)]
    cmd += [str(PROJECT_ROOT / 'lib/train/run_training.py'), '--script', args.script,
            '--config', args.config, '--save_dir', args.save_dir,
            '--use_lmdb', str(args.use_lmdb), '--seed', str(args.seed)]
    print('Training RSPTrack:', args.config, flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

if __name__ == '__main__':
    main()
