"""Public experiment names and compatibility with pre-release commands."""

DEFAULT_CONFIG = 'RSPTrack-B'
DEFAULT_DATASET = 'lasot_extension_subset'
LEGACY_CONFIGS = {
    'SAVLT-B-ReSA-TVPM_512_05': 'RSPTrack-B',
    'SAVLT-B': 'RSPTrack-B-Baseline',
    'SAVLT-B-ReSA': 'RSPTrack-B-LRSA',
    'SAVLT-B-TVPM-Ablation-1': 'RSPTrack-B-TVPM',
    'SAVLT-B-TVPM-Ablation-1-1': 'RSPTrack-B-TVPM-alpha0',
    'SAVLT-B-TVPM-Ablation-1-2': 'RSPTrack-B-TVPM-alpha025',
    'SAVLT-B-TVPM-Ablation-1-3': 'RSPTrack-B-TVPM-alpha1',
    'SAVLT-B-TVPM-Ablation-2-1': 'RSPTrack-B-LRSA-f0',
    'SAVLT-B-TVPM-Ablation-2-2': 'RSPTrack-B-LRSA-f2',
    'SAVLT-B-TVPM-Ablation-2-3': 'RSPTrack-B-LRSA-f8',
}


def canonical_experiment(name, config):
    if name.lower() == 'savlt':
        name = 'rsptrack'
    return name, LEGACY_CONFIGS.get(config, config)
