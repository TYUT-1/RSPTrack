"""Export a trusted training checkpoint for RSPTrack release or resumption."""
import argparse
import hashlib
import json
from pathlib import Path
import _init_paths
from lib.local_paths import PROJECT_ROOT
from lib.project import DEFAULT_CONFIG, canonical_experiment

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Your own training checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--keep-training-state', action='store_true', help='Keep optimizer and stats for resuming; omit for release')
    args = parser.parse_args()
    import torch
    source, target = Path(args.input).resolve(), Path(args.output).resolve()
    if target.exists() or target == source:
        parser.error('--output must be a new file')
    _, config_name = canonical_experiment('rsptrack', args.config)
    config_path = PROJECT_ROOT / 'experiments/rsptrack' / (config_name + '.yaml')
    if not config_path.is_file():
        parser.error('Unknown configuration: ' + config_name)
    # Full training files include Python settings objects: use only trusted inputs.
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    state = checkpoint.get('net', checkpoint)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError('Expected a tensor state_dict or a checkpoint containing net')
    state = {k.removeprefix('module.'): v.detach().cpu() for k, v in state.items()}
    result = dict(checkpoint) if args.keep_training_state else {}
    result.update(net=state, net_type='RSPTrack', actor_type='RSPTrackActor',
                  epoch=int(checkpoint.get('epoch', 200)), config=config_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, target)
    digest = hashlib.sha256()
    with target.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    target.with_name(target.name + '.sha256').write_text(digest.hexdigest() + '  ' + target.name + '\n', encoding='utf-8')
    target.with_name(target.name + '.yaml').write_text(config_path.read_text(encoding='utf-8'), encoding='utf-8')
    print(json.dumps({'file': str(target), 'bytes': target.stat().st_size, 'sha256': digest.hexdigest(),
                      'config': config_name, 'training_state': args.keep_training_state}, indent=2))

if __name__ == '__main__':
    main()
