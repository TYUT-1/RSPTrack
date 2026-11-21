"""Create portable relative paths for an LMDB build."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--result_file', required=True)
    args = parser.parse_args()
    root = Path(args.base_dir).resolve()
    names = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())
    Path(args.result_file).write_text('\n'.join(names) + '\n', encoding='utf-8')
    print('Files:', len(names))

if __name__ == '__main__':
    main()
