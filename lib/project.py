"""RSPTrack experiment names shared by training, inference and evaluation."""
from pathlib import Path


DEFAULT_CONFIG = 'RSPTrack-B'
DEFAULT_DATASET = 'lasot_extension_subset'
_CONFIG_DIR = Path(__file__).resolve().parents[1] / 'experiments' / 'rsptrack'


def canonical_experiment(script_name='rsptrack', config_name=DEFAULT_CONFIG):
    """Return the tracker module name and an existing RSPTrack YAML stem."""
    script_name = script_name.strip().lower()
    if script_name != 'rsptrack':
        raise ValueError("Unsupported tracker %r; use 'rsptrack'." % script_name)

    config_name = config_name.strip()
    if config_name.lower().endswith(('.yaml', '.yml')):
        config_name = config_name.rsplit('.', 1)[0]
    configs = {path.stem.lower(): path.stem for path in _CONFIG_DIR.glob('*.yaml')}
    if config_name.lower() not in configs:
        raise ValueError('Unknown RSPTrack configuration %r. Available: %s' %
                         (config_name, ', '.join(sorted(configs.values()))))
    return script_name, configs[config_name.lower()]
