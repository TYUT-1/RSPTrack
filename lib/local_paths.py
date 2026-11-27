"""Portable RSPTrack paths, optionally overridden by local_paths.json."""
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG_FILE = PROJECT_ROOT / 'local_paths.json'
_FIXED_PATHS = ('workspace_dir', 'prj_dir', 'dataspec_path')
_SECTION_COMMENTS = {
    'data_dir': ('_comment_common', '===== Common paths: dataset root, pretrained weights, logs and outputs ====='),
    'lasot_dir': ('_comment_train', '===== Training datasets (*_dir): --use_lmdb 1 selects the corresponding *_lmdb_dir; TNL2K always uses raw files ====='),
    'lasot_path': ('_comment_test', '===== Test/evaluation datasets (*_path): selected by --dataset_name, independently of training --use_lmdb; some roots may be shared with training ====='),
}


def _absolute(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _directory(root, *names):
    """Use an existing spelling/split, or the first documented default."""
    return next((root / name for name in names if (root / name).is_dir()), root / names[0])


def _default_paths(data_dir='data', save_dir='output'):
    data = _absolute(data_dir)
    output = _absolute(save_dir)
    lasot = _directory(data, 'Lasot', 'LaSOT', 'lasot')
    got10k = _directory(data, 'GOT10k', 'GOT10K', 'got10k')
    coco = _directory(data, 'COCO', 'coco')
    trackingnet = _directory(data, 'Trackingnet', 'TrackingNet', 'trackingnet')
    tnl2k = _directory(data, 'TNL2K', 'tnl2k')
    lasot_lmdb = _directory(data, 'Lasot_lmdb', 'LaSOT_lmdb', 'lasot_lmdb')
    otb99 = _directory(data, 'OTB99', 'OTB99_Lang', 'OTB99_lang', 'otb99_lang')
    paths = {
        # 1. Common paths: project location, pretrained weights, logs and outputs.
        'workspace_dir': PROJECT_ROOT,
        'prj_dir': PROJECT_ROOT,
        'data_dir': data,
        'save_dir': output,
        'tensorboard_dir': output / 'tensorboard',
        'pretrained_networks': PROJECT_ROOT / 'pretrained',
        'network_path': PROJECT_ROOT / 'pretrained',
        'results_path': output / 'test' / 'tracking_results',
        'segmentation_path': output / 'test' / 'segmentation_results',
        'result_plot_path': output / 'test' / 'result_plots',
        'got_reports_path': output / 'test' / 'got10k_reports',
        'dataspec_path': PROJECT_ROOT / 'lib' / 'train' / 'data_specs',

        # 2. Training dataset paths (*_dir): used by training data loaders.
        # Raw GOT-10k points to train/; its LMDB keys already include train/.
        'lasot_dir': lasot,
        'got10k_dir': got10k / 'train',
        'coco_dir': coco,
        'trackingnet_dir': trackingnet,
        'tnl2k_dir': _directory(tnl2k, 'train_set', 'train'),
        'lasot_lmdb_dir': lasot_lmdb,
        'got10k_lmdb_dir': _directory(data, 'GOT10k_lmdb', 'GOT10K_lmdb', 'got10k_lmdb'),
        'coco_lmdb_dir': _directory(data, 'COCO_lmdb', 'coco_lmdb'),
        'trackingnet_lmdb_dir': _directory(data, 'Trackingnet_lmdb', 'TrackingNet_lmdb', 'trackingnet_lmdb'),
        'refcoco_dir': coco,
        'imagenet_dir': _directory(data, 'VID', 'vid'),
        'imagenet_lmdb_dir': _directory(data, 'VID_lmdb', 'vid_lmdb'),
        'imagenet1k_dir': _directory(data, 'imagenet1k', 'ImageNet1K'),
        'imagenet22k_dir': _directory(data, 'imagenet22k', 'ImageNet22K'),
        'depthtrack_dir': _directory(data, 'DepthTrack', 'depthtrack'),
        'lasher_dir': _directory(data, 'LasHeR', 'lasher'),
        'visevent_dir': _directory(data, 'VisEvent', 'visevent'),
        'vasttrack_dir': _directory(data, 'VastTrack', 'vasttrack'),
        'otb99_dir': otb99,

        # 3. Test/evaluation dataset paths (*_path): used by evaluation loaders.
        # dataset_name selects raw files or LMDB, independently of training use_lmdb.
        # Training and evaluation may share a root but select different sequences.
        'lasot_path': lasot,
        'lasot_lmdb_path': lasot_lmdb,
        'lasot_extension_subset_path': _directory(data, 'LaSOT_extension_subset', 'Lasot_ext', 'LaSOT_ext', 'lasot_extension_subset'),
        'got10k_path': got10k,
        'trackingnet_path': trackingnet,
        'tnl2k_path': _directory(tnl2k, 'test_set', 'test'),
        'otblang_path': otb99,
        'otb_path': _directory(data, 'OTB', 'OTB2015', 'otb'),
        'nfs_path': _directory(data, 'NFS', 'nfs'),
        'uav_path': _directory(data, 'UAV123', 'UAV', 'uav'),
        'tc128_path': _directory(data, 'TC128', 'tc128'),
    }
    return {key: str(value) for key, value in paths.items()}


def load_paths():
    """Read shared paths without creating files or requiring ML dependencies.

    Relative JSON values are resolved from the repository, not the process cwd.
    Dataset directories need only exist when their loaders are actually used.
    Keys beginning with _comment_ are section notes, not path settings.
    """
    if not LOCAL_CONFIG_FILE.is_file():
        return _default_paths()
    try:
        config = json.loads(LOCAL_CONFIG_FILE.read_text(encoding='utf-8-sig'))
    except (ValueError, UnicodeError) as error:
        raise ValueError('Invalid JSON in %s: %s' % (LOCAL_CONFIG_FILE, error)) from error
    if not isinstance(config, dict):
        raise ValueError('%s must contain a JSON object.' % LOCAL_CONFIG_FILE)
    config = {key: value for key, value in config.items() if not key.startswith('_comment_')}
    for key, value in config.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Path %r in %s must be a nonempty string.' % (key, LOCAL_CONFIG_FILE))
    paths = _default_paths(config.get('data_dir', 'data'), config.get('save_dir', 'output'))
    unknown = set(config) - set(paths)
    if unknown:
        raise ValueError('Unknown paths in %s: %s' % (LOCAL_CONFIG_FILE, ', '.join(sorted(unknown))))
    for key, value in config.items():
        if key not in _FIXED_PATHS:
            paths[key] = str(_absolute(value))
    return paths


def write_paths(workspace_dir, data_dir, save_dir=None):
    """Generate machine-local paths for the repository's shared settings."""
    if Path(workspace_dir).expanduser().resolve() != PROJECT_ROOT:
        raise ValueError('workspace_dir must be the current RSPTrack repository.')
    paths = _default_paths(data_dir, 'output' if save_dir is None else save_dir)
    config = {}
    for key, value in paths.items():
        if key in _FIXED_PATHS:
            continue
        if key in _SECTION_COMMENTS:
            comment_key, comment = _SECTION_COMMENTS[key]
            config[comment_key] = comment
        config[key] = value
    text = json.dumps(config, indent=2, ensure_ascii=False)
    text = text.replace('\n  "_comment_train"', '\n\n  "_comment_train"')
    text = text.replace('\n  "_comment_test"', '\n\n  "_comment_test"')
    LOCAL_CONFIG_FILE.write_text(text + '\n', encoding='utf-8')
    return LOCAL_CONFIG_FILE
