"""Evidence-gate tests only; no GPU training or real cloud publication implied."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from accept_full_downstream import accept_formal,accept_formal_session
from cloud_store import atomic_json,sha
from prepare_full_downstream import LEARNING_FILES,validate_resume
from prepare_full_shard import FOLDER_ID


class FormalFixture:
    def __init__(self,root,status='FORMAL_SESSION_BOUNDARY',completed=2):
        self.root=root;self.frozen=root/'frozen';self.collected=root/'compact'
        self.frozen.mkdir();(self.collected/'formal_full').mkdir(parents=True)
        self.publication=root/'publication.json';self.output=root/'acceptance.json';self.parent=root/'parent.json'
        self.run_id='k230-topface-v11-formal-fixture'
        self.sources={name:hashlib.sha256(name.encode()).hexdigest() for name in
                      (*LEARNING_FILES,'nanodet-source.bin','nanodet-pretrained.bin')}
        experiment={'experiment_id':'same-logical-training','shards':[], 'scenes':8,
                    'data_seed':20261117,'training_seed':20261114,'width':320,'height':192,
                    'batch_size':2,'max_epochs':100,'ellipse_shape_loss':False}
        self.config={'run_id':self.run_id,'session':1,'experiment':experiment,
            'parent':{'kind':'pilot','completed_epochs':0},'training_source_sha256':self.sources,
            'dataset_acceptance':{'dataset_index_sha256':'a'*64,'split_images':{'train':6,'val':2}}}
        self.checkpoints=[{'name':name,'bytes':20,'sha256':hashlib.sha256(name.encode()).hexdigest()}
                          for name in ('best.pt','last.pt')]
        self.trace={'kind':'formal_full','status':status,'formal_identity':experiment['experiment_id'],
            **{k:experiment[k] for k in ('width','height','batch_size','max_epochs','ellipse_shape_loss')},
            'seed':20261114,'dataset_audit_sha256':'a'*64,'training_scene_count':6,'validation_scene_count':2,
            'real_background_augmentation':None,'best_checkpoint_sha256':self.checkpoints[0]['sha256'],
            'best_epoch':1,'completed_epochs':completed,'optimizer_updates':completed*3,
            'checkpoints':self.checkpoints,'schedule':{'epoch_batches':3,'target_updates':300,'warmup_updates':30},
            'runtime':{'torch':'qualified-fixture','cuda':'fixture'},'validation_selection':'frozen fixture'}
        self.metrics={'images':2}
        self.result={'run_id':self.run_id,'status':status,'session':1,
            'experiment_id':experiment['experiment_id'],'completed_epochs':completed,
            'optimizer_updates':completed*3,'best_epoch':1,'checkpoints':self.checkpoints,
            'resume_eligible':status=='FORMAL_SESSION_BOUNDARY','gate':{'pass':False},
            'archive':{'name':self.run_id+'.tar.gz','bytes':100,'sha256':'b'*64}}
        self.write()

    def write(self):
        notebook=self.frozen/'formal.ipynb';notebook.write_text('{"cells":[]}',encoding='utf8')
        atomic_json(self.frozen/'FORMAL_CONFIG.json',self.config)
        files=[{'path':k,'bytes':1,'sha256':v} for k,v in self.sources.items()]
        files.append({'path':'FORMAL_CONFIG.json','bytes':(self.frozen/'FORMAL_CONFIG.json').stat().st_size,
                      'sha256':sha(self.frozen/'FORMAL_CONFIG.json')})
        atomic_json(self.frozen/'INPUT_MANIFEST.json',{'files':files})
        digest=sha(self.frozen/'INPUT_MANIFEST.json')
        intent={'run_id':self.run_id,'kernel':'owner/fixture','dataset':'owner/input','dataset_version':1,
            'input_manifest_sha256':digest,'notebook_file':'formal.ipynb','notebook_sha256':sha(notebook)}
        atomic_json(self.frozen/'SUBMISSION_INTENT.json',intent)
        atomic_json(self.frozen/'OBSERVATION-v1.json',{'ref':'owner/fixture','kernel_id':123,'version':1,'status':'COMPLETE'})
        prep={'ref':'owner/fixture','kernel_id':123,'version':1,'provider_status':'COMPLETE',
              'source_sha256':sha(notebook)}
        atomic_json(self.publication,{'status':'DRIVE_EXPORT_FULL_BYTE_SHA_COMMITTED',
            'secret_cleanup':'CONFIRMED_DELETED','folder_id':FOLDER_ID,'independent_metadata_match':True,
            'automatic_replay':False,'run_id':self.run_id,'prep':prep,
            'uploaded':{'name':f'{self.run_id}-'+('b'*16)+'.tar.gz','bytes':100,'sha256':'b'*64,
                        'provider_readback':'FULL_BYTE_SHA256_PASS'}})
        directory=self.collected/'formal_full'
        atomic_json(directory/'RESULT.json',self.trace);atomic_json(directory/'BEST_METRICS.json',self.metrics)
        self.result['input_manifest_sha256']=digest
        self.result['compact_training_files']=[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)}
            for p in (directory/'RESULT.json',directory/'BEST_METRICS.json')]
        atomic_json(self.collected/(self.run_id+'-RESULT.json'),self.result)

    def accept(self,final=False):
        return (accept_formal if final else accept_formal_session)(self.frozen,self.collected,
            self.publication,self.output,self.parent)


class FormalSessionTests(unittest.TestCase):
    def test_clean_epoch_acceptance_resumes_last_and_retains_earlier_best(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=FormalFixture(Path(directory));result=fixture.accept()
            parent=json.loads(fixture.parent.read_text())
            self.assertFalse(result['formal_training_complete'])
            self.assertEqual(parent['checkpoint_sha256'],fixture.checkpoints[1]['sha256'])
            self.assertEqual(parent['best_checkpoint']['sha256'],fixture.checkpoints[0]['sha256'])
            self.assertEqual(parent['archive']['name'],fixture.run_id+'.tar.gz')
            self.assertNotEqual(parent['archive']['name'],parent['publication']['uploaded']['name'])
            config=copy.deepcopy(fixture.config);config.update(session=2,parent=parent)
            validate_resume(config,fixture.sources)

    def test_unfinished_epoch_session_cannot_be_a_final_model(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=FormalFixture(Path(directory))
            with self.assertRaisesRegex(ValueError,'final acceptance'):fixture.accept(final=True)
            self.assertFalse(fixture.parent.exists())

    def test_final_model_selects_best_instead_of_resume_last(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=FormalFixture(Path(directory),'FINITE_TRAINING_COMPLETE',100);fixture.accept(final=True)
            model=json.loads(fixture.parent.read_text())
            self.assertEqual(model['kind'],'formal_model')
            self.assertEqual(model['checkpoint_sha256'],fixture.checkpoints[0]['sha256'])

    def test_mid_epoch_or_no_progress_is_not_a_clean_resume_boundary(self):
        for status in ('FORMAL_BUDGET_STOP_MID_EPOCH','FAILED_PRESERVE_CHECKPOINTS','FORMAL_NO_EPOCH_PROGRESS'):
            with self.subTest(status=status),tempfile.TemporaryDirectory() as directory:
                fixture=FormalFixture(Path(directory),status)
                with self.assertRaisesRegex(ValueError,'unfinished epoch boundary'):fixture.accept()
                self.assertFalse(fixture.parent.exists())

    def test_partial_upload_or_tampered_compact_cannot_accept_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=FormalFixture(Path(directory));durable=json.loads(fixture.publication.read_text())
            durable['uploaded']['provider_readback']='METADATA_ONLY';atomic_json(fixture.publication,durable)
            with self.assertRaisesRegex(ValueError,'durable publication'):fixture.accept()
        with tempfile.TemporaryDirectory() as directory:
            fixture=FormalFixture(Path(directory));atomic_json(fixture.collected/'formal_full/RESULT.json',{})
            with self.assertRaisesRegex(ValueError,'saved exact bytes'):fixture.accept()

    def test_epoch_updates_must_be_whole_and_source_config_must_be_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=FormalFixture(Path(directory));fixture.result['optimizer_updates']=7
            fixture.trace['optimizer_updates']=7;fixture.write()
            with self.assertRaisesRegex(ValueError,'frozen experiment'):fixture.accept()
        with tempfile.TemporaryDirectory() as directory:
            fixture=FormalFixture(Path(directory));changed=copy.deepcopy(fixture.config)
            changed['experiment']['max_epochs']=200;atomic_json(fixture.frozen/'FORMAL_CONFIG.json',changed)
            with self.assertRaisesRegex(ValueError,'frozen input manifest'):fixture.accept()

    def test_resume_rejects_budget_background_source_dataset_and_session_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture=FormalFixture(Path(directory));fixture.accept()
            parent=json.loads(fixture.parent.read_text())
            baseline=copy.deepcopy(fixture.config);baseline.update(session=2,parent=parent)
            changes=[lambda c:c['experiment'].update(max_epochs=200),
                     lambda c:c.update(real_background_bank={'identity':'changed'}),
                     lambda c:c.update(body_negative_bank={'identity':'changed'}),
                     lambda c:c['dataset_acceptance'].update(dataset_index_sha256='c'*64),
                     lambda c:c.update(session=3)]
            for change in changes:
                modified=copy.deepcopy(baseline);change(modified)
                with self.assertRaisesRegex(ValueError,'preserve the accepted'):validate_resume(modified,fixture.sources)
            with self.assertRaisesRegex(ValueError,'preserve the accepted'):
                validate_resume(baseline,{**fixture.sources,'train_experiment.py':'d'*64})


if __name__=='__main__':unittest.main()
