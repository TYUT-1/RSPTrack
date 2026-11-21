"""Small CPU checks for public naming, ablations and sparse attention invariants."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from lib.local_paths import load_paths
from lib.project import canonical_experiment


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProtocolTests(unittest.TestCase):
    def test_legacy_command_and_ablations(self):
        self.assertEqual(canonical_experiment('savlt', 'SAVLT-B-ReSA-TVPM_512_05'),
                         ('rsptrack', 'RSPTrack-B'))
        configs = {p.stem: yaml.safe_load(p.read_text(encoding='utf-8'))
                   for p in (ROOT / 'experiments/rsptrack').glob('*.yaml')}
        self.assertEqual(len(configs), 10)
        for config in configs.values():
            self.assertEqual(config['TRAIN']['EPOCH'], 200)
            self.assertEqual(config['TRAIN']['BATCH_SIZE'], 32)
            self.assertEqual(config['MODEL']['ENCODER']['TVPM']['HIDDEN_DIM'], 512)
            self.assertFalse(config['DATA']['USE_NLP']['LASOT'])
            self.assertFalse(config['TEST']['USE_NLP']['DEFAULT'])
            self.assertFalse(config['TEST']['USE_NLP']['LASOT'])
            self.assertTrue(config['TEST']['USE_NLP']['TNL2K'])
        full = configs['RSPTrack-B']['MODEL']['ENCODER']
        self.assertTrue(full['TVPM']['ENABLED'] and full['LRSA']['ENABLED'])
        self.assertEqual(full['LRSA']['RECTIFY_INTERVAL'], 4)
        self.assertEqual(full['TVPM']['ROLE_SCALE'], 0.5)
        for suffix, enabled in [('Baseline', (False, False)), ('TVPM', (True, False)), ('LRSA', (False, True))]:
            enc = configs['RSPTrack-B-' + suffix]['MODEL']['ENCODER']
            self.assertEqual((enc['TVPM']['ENABLED'], enc['LRSA']['ENABLED']), enabled)
        for f in (0, 2, 8):
            self.assertEqual(configs['RSPTrack-B-LRSA-f%d' % f]['MODEL']['ENCODER']['LRSA']['RECTIFY_INTERVAL'], f)

    def test_portable_paths_and_no_lasot_text_path(self):
        with tempfile.TemporaryDirectory() as temp:
            config = Path(temp) / 'paths.json'
            config.write_text(json.dumps({'data_dir': temp, 'save_dir': temp + '/runs'}), encoding='utf-8')
            with patch.dict(os.environ, {'RSPTRACK_PATHS_FILE': str(config)}, clear=True):
                paths = load_paths()
            self.assertEqual(Path(paths['results_path']), Path(temp) / 'runs/test/tracking_results')
            self.assertNotIn('lasotlang_path', paths)
            self.assertNotIn('lasotextlang_path', paths)

    def test_all_experiments_merge_into_schema(self):
        from copy import deepcopy
        from easydict import EasyDict
        from lib.config.rsptrack.config import cfg, _update_config
        for file in (ROOT / 'experiments/rsptrack').glob('*.yaml'):
            merged = deepcopy(cfg)
            _update_config(merged, EasyDict(yaml.safe_load(file.read_text(encoding='utf-8'))))
            self.assertTrue(merged.TEST.MULTI_MODAL_LANGUAGE.DEFAULT)
            self.assertFalse(merged.TEST.USE_NLP.LASOT)
            self.assertTrue(merged.TEST.USE_NLP.OTB99_LANG)

    def test_lasot_loaders_and_placeholder_tokens_need_no_real_text(self):
        from lib.test.evaluation.lasotdataset import LaSOTDataset
        from lib.test.evaluation.lasotextensionsubsetdataset import LaSOTExtensionSubsetDataset
        from lib.test.tracker.rsptrack import RSPTrack
        from lib.train.data.sampler import TrackingSampler
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'airplane/airplane-1'
            directory.mkdir(parents=True)
            (directory / 'groundtruth.txt').write_text('10,10,20,20\n' * 3)
            for filename in ['full_occlusion.txt', 'out_of_view.txt']:
                (directory / filename).write_text('0,0,0')
            for cls in [LaSOTDataset, LaSOTExtensionSubsetDataset]:
                loader = cls.__new__(cls)
                loader.base_path = temp
                sequence = loader._construct_sequence('airplane-1')
                self.assertNotIn('init_nlp', sequence.init_info())
        train_ids, _, train_length = TrackingSampler.extract_token_from_nlp_clip(None, None)
        test_ids, _, test_length = RSPTrack.extract_token_from_nlp_clip(None, None)
        self.assertEqual(train_ids.numel(), 77)
        self.assertTrue(torch.equal(train_ids, torch.zeros(77, dtype=torch.long)))
        self.assertTrue(torch.equal(train_ids, test_ids))
        self.assertEqual((train_length, test_length), (0, 0))

    def test_lasot_lmdb_rejects_unsupported_real_text(self):
        from lib.train.dataset.lasot_lmdb import Lasot_lmdb
        for language_enabled in (True, False):
            with self.assertRaisesRegex(ValueError, 'DATA.USE_NLP.LASOT=False'):
                Lasot_lmdb(root='unused', split='train', use_nlp=True,
                           multi_modal_language=language_enabled)
        with patch.object(Lasot_lmdb, '_build_sequence_list', return_value=['airplane-1']):
            dataset = Lasot_lmdb(root='unused', split='train', use_nlp=False,
                                 multi_modal_language=True)
            self.assertFalse(dataset.use_nlp)

    def test_sparse_mask_padding_protection_and_gradient(self):
        sparse = load_file('lrsa_test', ROOT / 'lib/models/rsptrack/lrsa_attention.py')
        module = sparse.LRSABlockSparseMask(block_size=4, active_ratio=0.25, min_blocks=1, local_blocks=1)
        torch.manual_seed(17)
        query = torch.randn(2, 2, 17, 8, requires_grad=True)
        key = torch.randn(2, 2, 17, 8, requires_grad=True)
        padding = torch.zeros(2, 17, dtype=torch.bool)
        padding[:, -1] = True
        protected = torch.zeros_like(padding)
        protected[:, 12] = True
        keep = module.build_keep_mask(query, key, padding, protected)
        self.assertFalse(keep[..., -1].any())
        self.assertTrue(keep[..., 12:16].all())
        self.assertTrue(keep.diagonal(dim1=-2, dim2=-1)[..., :16].all())
        logits = query @ key.transpose(-2, -1)
        probs = module(logits, query, key, padding, protected).softmax(-1)
        self.assertTrue(torch.isfinite(probs).all())
        self.assertTrue(torch.equal(probs[..., -1], torch.zeros_like(probs[..., -1])))
        (probs * torch.randn_like(probs)).sum().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertEqual(module.state_dict(), {})

    def test_checkpoint_export_preserves_tensors_and_removes_training_state(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / 'old.pth.tar', Path(temp) / 'RSPTrack_ep0200.pth.tar'
            tensor = torch.randn(3, 4)
            torch.save({'net': {'module.encoder.weight': tensor}, 'epoch': 200,
                        'optimizer': {'private': 1}, 'settings': {'old_path': 'private'}}, source)
            subprocess.run([sys.executable, str(ROOT / 'tracking/transfer_model.py'),
                            '--input', str(source), '--output', str(target)], check=True, capture_output=True)
            exported = torch.load(target, weights_only=True)
            self.assertTrue(torch.equal(exported['net']['encoder.weight'], tensor))
            self.assertEqual(exported['net_type'], 'RSPTrack')
            self.assertNotIn('optimizer', exported)
            self.assertNotIn('settings', exported)
            self.assertTrue(target.with_name(target.name + '.sha256').exists())
            self.assertTrue(target.with_name(target.name + '.yaml').exists())

    def test_evaluation_cli_reads_canonical_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sequence = root / 'data/TNL2K/test_set/example'
            (sequence / 'imgs').mkdir(parents=True)
            for i in range(3):
                (sequence / 'imgs' / ('%05d.jpg' % i)).touch()
            (sequence / 'groundtruth.txt').write_text('10,10,20,20\n' * 3, encoding='utf-8')
            (sequence / 'language.txt').write_text('the tracked target', encoding='utf-8')
            result = root / 'runs/test/tracking_results/rsptrack/RSPTrack-B/tnl2k/example.txt'
            result.parent.mkdir(parents=True)
            result.write_text('10\t10\t20\t20\n' * 3, encoding='utf-8')
            config = root / 'paths.json'
            config.write_text(json.dumps({'data_dir': str(root / 'data'), 'save_dir': str(root / 'runs')}), encoding='utf-8')
            env = dict(os.environ, RSPTRACK_PATHS_FILE=str(config), MPLBACKEND='Agg')
            completed = subprocess.run([sys.executable, str(ROOT / 'tracking/analysis_results.py'),
                                        '--dataset_name', 'tnl2k', '--plot'], cwd=ROOT, env=env,
                                       capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn('RSPTrack', completed.stdout)
            self.assertIn('Norm Precision', completed.stdout)
            self.assertTrue(list((root / 'runs/test/result_plots').rglob('*_plot.pdf')))


if __name__ == '__main__':
    unittest.main()
