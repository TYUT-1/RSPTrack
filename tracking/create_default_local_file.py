"""Configure paths once for training, testing and evaluation."""
import argparse
from pathlib import Path
import _init_paths
from lib.local_paths import PROJECT_ROOT, write_paths

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace_dir', default=str(PROJECT_ROOT))
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--save_dir', default=str(PROJECT_ROOT / 'output'))
    args = parser.parse_args()
    if Path(args.workspace_dir).resolve() != PROJECT_ROOT:
        parser.error('--workspace_dir must be the current RSPTrack repository')
    print('Saved paths to', write_paths(args.workspace_dir, args.data_dir, args.save_dir))

if __name__ == '__main__':
    main()
