"""Build an LMDB from a file list using POSIX relative keys."""
import argparse
from pathlib import Path
import lmdb
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--result_file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--map_size_gb', type=int, default=1024)
    args = parser.parse_args()
    root = Path(args.base_dir).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        parser.error('--output must be a new directory')
    names = Path(args.result_file).read_text(encoding='utf-8').splitlines()
    with lmdb.open(str(output), map_size=args.map_size_gb * 1024**3) as env:
        for start in range(0, len(names), 10000):
            with env.begin(write=True) as txn:
                for name in tqdm(names[start:start + 10000]):
                    path = (root / name).resolve()
                    if not path.is_relative_to(root):
                        raise ValueError('File list points outside --base_dir')
                    txn.put(name.replace('\\', '/').encode(), path.read_bytes())

if __name__ == '__main__':
    main()
