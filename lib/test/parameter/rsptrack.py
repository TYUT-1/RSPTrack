from copy import deepcopy
from pathlib import Path
from lib.test.utils import TrackerParams
from lib.test.evaluation.environment import env_settings
from lib.config.rsptrack.config import cfg, _update_config
from lib.project import canonical_experiment
from easydict import EasyDict
import yaml

def parameters(yaml_name: str):
    _, yaml_name = canonical_experiment('rsptrack', yaml_name)
    params = TrackerParams()
    env = env_settings()
    config_file = Path(env.prj_dir) / 'experiments/rsptrack' / (yaml_name + '.yaml')
    params.cfg = deepcopy(cfg)
    _update_config(params.cfg, EasyDict(yaml.safe_load(config_file.read_text(encoding='utf-8'))))
    params.yaml_name = yaml_name
    params.template_factor = params.cfg.TEST.TEMPLATE_FACTOR
    params.template_size = params.cfg.TEST.TEMPLATE_SIZE
    params.search_factor = params.cfg.TEST.SEARCH_FACTOR
    params.search_size = params.cfg.TEST.SEARCH_SIZE
    params.checkpoint = str(Path(env.save_dir) / 'checkpoints/train/rsptrack' / yaml_name /
                            ('RSPTrack_ep%04d.pth.tar' % params.cfg.TEST.EPOCH))
    params.save_all_boxes = False
    return params
