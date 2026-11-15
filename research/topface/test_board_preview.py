"""Host fixture verification of preview identity, packaging and isolation."""
import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from board.package_trial import package as board_package
from cloud_store import archive_directory, atomic_json, sha
from package_board_preview import package_preview
from prepare_board_preview import prepare_preview
from prepare_full_shard import FOLDER_ID
from continue_board_preview import verify_original_selection


class PreviewPackageFixture:
    def __init__(self, root):
        self.root = root
        self.conversion = root / 'conversion'
        self.compact = self.conversion / 'compact-v1'
        self.evaluation = self.compact / 'evaluation'
        self.models = root / 'models'
        self.models.mkdir()
        self.compiled = root / 'compiled'
        self.compiled.mkdir()
        (self.compiled / 'topface.kmodel').write_bytes(b'host-fixture-not-native-model')
        self.export = root / 'export'
        self.export.mkdir()
        (self.export / 'topface.onnx').write_bytes(b'host-fixture-not-onnx')
        (self.models / 'topface-best.pt').write_bytes(b'host-fixture-checkpoint')
        self.identity = {'checkpoint_sha256': sha(self.models / 'topface-best.pt'),
                         'onnx_sha256': sha(self.export / 'topface.onnx'),
                         'kmodel_sha256': sha(self.compiled / 'topface.kmodel')}
        self.exported = {'experiment': 'formal_full', 'status': 'PYTORCH_ONNX_PARITY_PASS',
                         'epoch': 86, 'checkpoint_sha256': self.identity['checkpoint_sha256'],
                         'onnx_sha256': self.identity['onnx_sha256'],
                         'actual_train_image_pytorch_ort': {'images': 12, 'pass': True}}
        self.compiled_report = {'status': 'TRAINED_MODEL_COMPILED_AND_PAIRED', 'identity': self.identity,
                                'calibration_members': [{'split': 'train'} for _ in range(64)],
                                'paired_members': [{'split': 'train'} for _ in range(12)],
                                'paired_measurements': [{} for _ in range(12)]}
        atomic_json(self.compiled / 'identity.json', self.identity)
        atomic_json(self.compiled / 'output_contract.json', {'scope': 'fixture'})
        atomic_json(self.compiled / 'RESULT.json', self.compiled_report)
        atomic_json(self.export / 'RESULT.json', self.exported)
        board = root / 'board'
        board_package(self.compiled, self.export, board)
        archive_directory(board, self.models / 'topface-board-bundle.tar.gz')
        for source in (self.export / 'topface.onnx', self.compiled / 'topface.kmodel',
                       self.compiled / 'identity.json', self.compiled / 'output_contract.json'):
            (self.models / source.name).write_bytes(source.read_bytes())
        self.run_id = 'k230-topface-v11-conversion-formal-previewfixture'
        self.source = {'ref': 'lotus30/k230-topface-v11-07c-formal-board-preview',
                       'kernel_id': 123, 'version': 1, 'provider_status': 'COMPLETE'}
        self.saved = {'run_id': self.run_id, 'status': 'TRAINED_EXPORT_COMPILE_AND_SIMULATOR_COMPLETE',
                      'scope': 'FORMAL_BOARD_PREVIEW_ONLY', 'full_934_test': 'PENDING_ORIGINAL_07B',
                      'native_board_test': 'NOT_RUN', 'export': self.exported, 'identity': self.identity,
                      'archive': {'bytes': 1234, 'sha256': 'b' * 64},
                      'model_files': [{'path': self.run_id + '-models/' + path.name,
                                       'bytes': path.stat().st_size, 'sha256': sha(path)}
                                      for path in sorted(self.models.iterdir())]}
        self.result = self.compact / (self.run_id + '-RESULT.json')
        self.publication = root / 'publication.json'
        atomic_json(self.publication, {'status': 'DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED',
                    'secret_cleanup': 'CONFIRMED_DELETED', 'independent_metadata_match': True,
                    'run_id': self.run_id, 'folder_id': FOLDER_ID, 'prep': self.source,
                    'uploaded': {'bytes': 1234, 'sha256': 'b' * 64, 'provider_readback': 'FULL_BYTE_SHA256_PASS'}})
        atomic_json(self.conversion / 'MODEL_ARTIFACT_CONFIG.json',
                    {'phase': 'formal_board_preview', 'run_id': self.run_id,
                     'parent': {'kind': 'formal_model', 'completed_epochs': 100,
                                'checkpoint_sha256': self.identity['checkpoint_sha256']}})
        self.output = root / 'delivered'
        self.refresh()

    def refresh(self):
        rows = []
        for name, value in (('compiled/RESULT.json', self.compiled_report), ('exported/RESULT.json', self.exported)):
            target = self.evaluation / name
            atomic_json(target, value)
            rows.append({'path': name, 'bytes': target.stat().st_size, 'sha256': sha(target)})
        self.saved['evaluation_reports'] = rows
        atomic_json(self.result, self.saved)
        atomic_json(self.conversion / 'MODEL_FILES_RECEIPT.json',
                    {'status': 'BOUNDED_MODEL_DELIVERABLES_SHA256_PASS', 'source': self.source,
                     'run_id': self.run_id, 'source_result_sha256': sha(self.result),
                     'publication_receipt_sha256': sha(self.publication),
                     'files': self.saved['model_files'], 'local_directory': str(self.models)})

    def package(self):
        return package_preview(self.conversion, self.publication, self.output)


class PreviewPackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = PreviewPackageFixture(Path(self.temporary.name))

    def test_actual_board_archive_roundtrip_and_complete_hashes(self):
        value = self.fixture.package()
        self.assertEqual(value['status'], 'FORMAL_BOARD_PREVIEW_PACKAGE_VERIFIED')
        self.assertEqual(value['full_934_test'], 'PENDING_ORIGINAL_07B')
        self.assertEqual(value['native_board_test'], 'NOT_RUN')
        self.assertEqual(value['training_completed_epochs'], 100)
        for row in value['files']:
            path = self.fixture.output / row['path']
            self.assertEqual((path.stat().st_size, sha(path)), (row['bytes'], row['sha256']))
        self.assertEqual(sha(self.fixture.output / 'board/topface.kmodel'), self.fixture.identity['kmodel_sha256'])

    def test_undertrained_parent_is_rejected(self):
        config = self.fixture.conversion / 'MODEL_ARTIFACT_CONFIG.json'
        value = json.loads(config.read_text())
        value['parent']['completed_epochs'] = 99
        atomic_json(config, value)
        with self.assertRaises(ValueError):
            self.fixture.package()
        self.assertFalse(self.fixture.output.exists())

    def test_test_calibration_is_rejected(self):
        self.fixture.compiled_report['calibration_members'][0]['split'] = 'synthetic_test'
        self.fixture.refresh()
        with self.assertRaises(ValueError):
            self.fixture.package()

    def test_missing_pair_denominator_is_rejected(self):
        self.fixture.compiled_report['paired_measurements'].pop()
        self.fixture.refresh()
        with self.assertRaises(ValueError):
            self.fixture.package()

    def test_report_tamper_fails_without_publishing_partial_package(self):
        (self.fixture.evaluation / 'compiled/RESULT.json').write_text('{}')
        with self.assertRaises((KeyError, ValueError)):
            self.fixture.package()
        self.assertFalse(self.fixture.output.exists())

    def test_model_tamper_fails_atomically(self):
        (self.fixture.models / 'topface.kmodel').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.fixture.package()
        self.assertFalse(self.fixture.output.exists())

    def test_existing_package_is_preserved(self):
        self.fixture.output.mkdir()
        marker = self.fixture.output / 'keep'
        marker.write_text('preserve')
        with self.assertRaises(FileExistsError):
            self.fixture.package()
        self.assertEqual(marker.read_text(), 'preserve')


class PreviewPreparationTests(unittest.TestCase):
    def test_original_selection_rejects_another_formal_session(self):
        provider = {'ref': 'lotus30/k230-topface-v11-06-formal-training', 'kernel_id': 145144940, 'version': 1}
        parent = {'kind': 'formal_model', 'experiment': 'formal_full', 'completed_epochs': 100,
                  'publication': {'prep': provider}, 'checkpoint_sha256': 'a' * 64}
        accepted = {'run_id': 'k230-topface-v11-formal-a01', 'provider': provider,
                    'completed_epochs': 100, 'status': 'FORMAL_TRAINING_COMPUTE_AND_ARCHIVE_VERIFIED',
                    'selected_checkpoint': {'sha256': 'a' * 64}}
        verify_original_selection(parent, accepted)
        wrong = copy.deepcopy(accepted)
        wrong['provider']['kernel_id'] = 999
        wrong_parent = copy.deepcopy(parent)
        wrong_parent['publication']['prep'] = wrong['provider']
        with self.assertRaises(ValueError):
            verify_original_selection(wrong_parent, wrong)
        wrong = copy.deepcopy(accepted)
        wrong['completed_epochs'] = 101
        with self.assertRaises(ValueError):
            verify_original_selection(parent, wrong)

    def test_binding_changes_only_separate_unsubmitted_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / 'stage'
            frozen = root / 'docs/topface/v11/cloud/conversion-formal-previewfixture'
            original = root / 'docs/topface/v11/cloud/conversion-formal-a01'
            original.mkdir(parents=True)
            marker = original / 'keep'
            marker.write_bytes(b'original-frozen-job')
            def baseline(phase, revision, stage, *unused):
                stage.mkdir()
                frozen.mkdir()
                atomic_json(stage / 'MODEL_ARTIFACT_CONFIG.json', {'run_id': 'separate-fixture'})
                atomic_json(stage / 'dataset-metadata.json', {'id': 'lotus30/previewfixture'})
                (stage / 'model_artifact_job.py').write_text('# old worker retained\n')
                atomic_json(stage / 'INPUT_MANIFEST.json', {'files': []})
                old_digest = sha(stage / 'INPUT_MANIFEST.json')
                kernel = stage.with_name(stage.name + '-kernel')
                kernel.mkdir()
                atomic_json(kernel / 'job.ipynb', {'cells': [{'source':
                    'input_hash=' + repr(old_digest) + '\nfrom model_artifact_job import run\n'}]})
                atomic_json(kernel / 'kernel-metadata.json',
                            {'id': 'lotus30/k230-topface-v11-07b-formal-conversion'})
                return {'notebook_file': 'job.ipynb', 'input_manifest_sha256': old_digest,
                        'run_id': 'separate-fixture', 'dataset': 'lotus30/previewfixture'}
            with patch('prepare_board_preview.ROOT', root), patch('prepare_board_preview.prepare', baseline):
                intent = prepare_preview('previewfixture', staging, *(root / name for name in
                    ('parent', 'runtime', 'qualification', 'native', 'acceptance', 'inventory')))
            self.assertEqual(marker.read_bytes(), b'original-frozen-job')
            self.assertEqual(intent['kernel'], 'lotus30/k230-topface-v11-07c-formal-board-preview')
            self.assertEqual(intent['provider_max_seconds'], 6300)
            manifest = json.loads((staging / 'INPUT_MANIFEST.json').read_text())
            names = {row['path'] for row in manifest['files']}
            self.assertEqual(names, {path.name for path in staging.iterdir()} -
                             {'INPUT_MANIFEST.json', 'dataset-metadata.json'})
            for row in manifest['files']:
                self.assertEqual(row['sha256'], sha(staging / row['path']))
            notebook = frozen / 'job.ipynb'
            self.assertEqual(sha(notebook), intent['notebook_sha256'])
            code = json.loads(notebook.read_text())['cells'][0]['source']
            self.assertIn('from model_board_preview_job import run', code)
            self.assertIn(sha(staging / 'INPUT_MANIFEST.json'), code)


if __name__ == '__main__':
    unittest.main()
