"""Portable paths shared by training, testing and analysis."""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_paths():
    config = Path(os.environ.get('RSPTRACK_PATHS_FILE', PROJECT_ROOT / 'paths.local.json'))
    overrides = json.loads(config.read_text(encoding='utf-8')) if config.is_file() else {}
    data_dir = Path(os.environ.get('RSPTRACK_DATA_DIR', overrides.get('data_dir', PROJECT_ROOT / 'data'))).expanduser().resolve()
    save_dir = Path(os.environ.get('RSPTRACK_SAVE_DIR', overrides.get('save_dir', PROJECT_ROOT / 'output'))).expanduser().resolve()
    paths = {'prj_dir': str(PROJECT_ROOT), 'save_dir': str(save_dir), 'workspace_dir': str(save_dir),
             'tensorboard_dir': str(save_dir / 'tensorboard'), 'pretrained_networks': str(PROJECT_ROOT / 'pretrained')}
    datasets = {
        'lasot_dir': 'lasot', 'lasot_lmdb_dir': 'lasot_lmdb',
        'got10k_dir': 'got10k/train', 'got10k_lmdb_dir': 'got10k_lmdb',
        'trackingnet_dir': 'trackingnet', 'trackingnet_lmdb_dir': 'trackingnet_lmdb',
        'coco_dir': 'coco', 'coco_lmdb_dir': 'coco_lmdb', 'tnl2k_dir': 'TNL2K/train_set',
        'otb99_dir': 'otb_lang', 'vasttrack_dir': 'vasttrack',
        'imagenet1k_dir': 'imagenet1k', 'imagenet22k_dir': 'imagenet22k',
        'imagenet_dir': 'vid', 'imagenet_lmdb_dir': 'vid_lmdb',
        'depthtrack_dir': 'depthtrack/train', 'lasher_dir': 'lasher/trainingset',
        'visevent_dir': 'visevent/train', 'refcoco_dir': 'refcoco',
        'lasot_path': 'lasot', 'lasot_extension_subset_path': 'LaSOT_Ext',
        'tnl2k_path': 'TNL2K/test_set', 'otblang_path': 'otb_lang',
        'got10k_path': 'got10k', 'got10k_lmdb_path': 'got10k_lmdb',
        'lasot_lmdb_path': 'lasot_lmdb', 'trackingnet_path': 'trackingnet',
        'otb_path': 'OTB2015', 'nfs_path': 'nfs', 'uav_path': 'UAV123',
        'tc128_path': 'TC128', 'vot_path': 'VOT2019',
    }
    paths.update({k: str(data_dir / v) for k, v in datasets.items()})
    paths.update({k: str(save_dir / 'test' / v) for k, v in {
        'results_path': 'tracking_results', 'result_plot_path': 'result_plots',
        'network_path': 'networks', 'segmentation_path': 'segmentation_results'}.items()})
    paths.update({k: '' for k in ['lvis_dir', 'sbd_dir', 'imagenetdet_dir', 'ecssd_dir', 'hkuis_dir',
                                'msra10k_dir', 'davis_dir', 'youtubevos_dir', 'tpl_path',
                                'got_packed_results_path', 'got_reports_path', 'tn_packed_results_path']})
    for k, v in overrides.items():
        if k not in ('data_dir', 'save_dir', 'prj_dir', 'workspace_dir') and k in paths:
            paths[k] = str(Path(v).expanduser().resolve()) if v else ''
    return paths


def write_paths(workspace_dir, data_dir, save_dir=None):
    """Write private JSON without overwriting the portable Python loaders."""
    root = Path(workspace_dir).expanduser().resolve()
    target = root / 'paths.local.json'
    current = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {}
    current.update(data_dir=str(Path(data_dir).expanduser().resolve()),
                   save_dir=str(Path(save_dir or root / 'output').expanduser().resolve()))
    target.write_text(json.dumps(current, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return target
