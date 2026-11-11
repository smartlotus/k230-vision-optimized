"""Synthetic scratch fixtures; never approve project data or train the task model."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from tools.material_dataset import register,audit,canonical
from tools.material_preprocess import model_uint8
from tools.train_material_pilot import (plan_training,stage_dataset,verify_staging,OptimizerLedger,
    PilotBudgetReached,run_training,verify_training_run,trainer_class,sha,write_receipt)


class MaterialTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        environment=patch.dict(os.environ,{'YOLO_CONFIG_DIR':str(Path(__file__).resolve().parents[2]/'.ops/local/yolo-test-settings-l'),
                                          'YOLO_OFFLINE':'true'})
        environment.start();self.addCleanup(environment.stop)
        sources=[]
        for i,value in enumerate((0xf800,0x07e0,0x001f,0xffff)):
            (self.root/f'{i}.rgb565').write_bytes(np.full((360,640),value,dtype='<u2').tobytes())
            cv2.imwrite(str(self.root/f'{i}.jpg'),np.full((360,640,3),i*60,np.uint8))
            (self.root/f'{i}.json').write_text(json.dumps(dict(source='REAL_CAMERA_NATIVE_LAB',is_simulated=False,
                camera_id=2,width=640,height=360,frame_id=i+1,timestamp_ms=100+i*100)))
            sources.append(dict(jpeg_path=f'{i}.jpg',rgb565_path=f'{i}.rgb565',metadata_path=f'{i}.json'))
        self.manifest=register(self.root,dict(object_set_final_confirmed_by_operator=True,sources=sources))
        for row,split in zip(self.manifest['images'],('train','train','val','test')):
            row.update(scene_group_id='SYNTHETIC_UNIT_FIXTURE_'+row['image_id'],scene_group_confirmed=True,split=split)
            row['review'].update(status='APPROVED',reviewer='TEST_ONLY_NOT_REAL_GT',all_visible_tops_checked=True,
                objects=[dict(class_id=0,bbox_xywh=[100,80,40,20],color_id=1)])
        self.initial=self.root/'initial.pt';self.initial.write_bytes(b'TEST_LOCAL_WEIGHT_NOT_A_TORCH_MODEL')
        self.manifest_path=self.root/'reviewed.json';write_receipt(self.manifest_path,self.manifest)

    def test_plan_is_bounded_and_only_complete_real_review_format_enters(self):
        plan=plan_training(self.root,self.manifest,self.initial)
        self.assertEqual(plan['planned_optimizer_updates'],1)
        self.assertEqual(plan['real_optimizer_updates'],0)
        self.assertEqual(plan['status'],'PLAN_ONLY_NOT_STARTED')
        self.assertFalse(plan['held_out_test_consumed'])
        for kwargs in (dict(epochs=0),dict(epochs=6),dict(batch=True),dict(batch=5),dict(max_seconds=601),
                       dict(max_updates=0),dict(epochs=5,batch=1,max_updates=1)):
            with self.assertRaises(ValueError):plan_training(self.root,self.manifest,self.initial,**kwargs)
        with self.assertRaises(ValueError):plan_training(self.root,self.manifest,self.root/'missing.pt')

    def test_project_unreviewed_frames_reject_before_weights_or_any_run_directory(self):
        root=Path(__file__).resolve().parents[1]
        target=self.root/'not-started'
        with patch('tools.train_material_pilot.trainer_class') as backend:
            with self.assertRaisesRegex(ValueError,'Reviewed complete'):
                run_training(root,root/'data/ai_pilot/material_manifest_r3.json',self.root/'missing.pt',target)
            backend.assert_not_called()
        self.assertFalse(target.exists())

    def test_stage_exact_pixels_and_transformed_labels_never_stage_test_images(self):
        from tools.material_dataset import verified_rgb
        for size in (224,320):
            output=self.root/f'dataset-{size}'
            result=stage_dataset(self.root,self.manifest,output,size)
            self.assertEqual(len(result['images']),3)
            self.assertFalse((output/'images/test').exists())
            self.assertNotIn('test:',(output/'dataset.yaml').read_text())
            for row in result['images']:
                source=next(r for r in self.manifest['images'] if r['image_id']==row['image_id'])
                expected,_=model_uint8(verified_rgb(self.root,source),size)
                rgb=cv2.cvtColor(cv2.imread(str(output/row['image'])),cv2.COLOR_BGR2RGB)
                self.assertTrue(np.array_equal(rgb,expected[0].transpose(1,2,0)))
                fields=(output/row['label']).read_text().split()
                self.assertEqual(fields[0],'0')
                self.assertAlmostEqual(float(fields[1]),120/640,places=8)
                self.assertAlmostEqual(float(fields[2]),90/640+49/224,places=8)
                self.assertAlmostEqual(float(fields[3]),40/640,places=8)
            verify_staging(output,result)
            with self.assertRaises(ValueError):stage_dataset(self.root,self.manifest,output,size)

    def test_staging_mutation_extra_files_and_yaml_drift_fail_closed(self):
        for i,mutation in enumerate(('image','label','extra','yaml')):
            output=self.root/f'dataset-{i}';result=stage_dataset(self.root,self.manifest,output,224)
            if mutation in ('image','label'):(output/result['images'][0][mutation]).write_bytes(b'changed')
            elif mutation=='extra':(output/'images/train/rogue.png').write_bytes(b'rogue')
            else:(output/'dataset.yaml').write_text('test: images/test')
            with self.assertRaises(ValueError):verify_staging(output,result)

    def test_optimizer_count_is_actual_torch_step_not_batches_or_unused_callback(self):
        import torch
        value=torch.nn.Parameter(torch.tensor(1.))
        optimizer=torch.optim.SGD([value],lr=.1)
        receipt={'real_optimizer_updates':0};clock=[0.]
        ledger=OptimizerLedger(receipt,self.root/'ledger.json',2,10,clock=lambda:clock[0])
        ledger.attach(optimizer)
        optimizer.zero_grad();(value*value).backward()
        self.assertEqual(receipt['real_optimizer_updates'],0)
        optimizer.step();optimizer.zero_grad();(value*value).backward();optimizer.step()
        self.assertEqual(receipt['real_optimizer_updates'],2)
        with self.assertRaises(PilotBudgetReached):optimizer.step()
        self.assertEqual(receipt['real_optimizer_updates'],2)
        ledger.close();self.assertFalse(ledger.handles)
        second=OptimizerLedger({'real_optimizer_updates':0},self.root/'timeout.json',2,10,clock=lambda:clock[0])
        clock[0]=11
        with self.assertRaises(PilotBudgetReached):second.check_time()
        self.assertFalse((self.root/'timeout.json').exists())

    @unittest.skipIf(importlib.util.find_spec('ultralytics') is None,'Pinned training environment unavailable')
    def test_real_pinned_dataset_loading_is_square_no_augmentation_pixel_exact(self):
        from ultralytics.cfg import get_cfg
        from tools.material_dataset import verified_rgb
        for size in (224,320):
            output=self.root/f'loader-{size}';stage_dataset(self.root,self.manifest,output,size)
            receipt={'real_optimizer_updates':0}
            ledger=OptimizerLedger(receipt,self.root/'unexecuted.json',3,30)
            cls=trainer_class(ledger,lambda:None);trainer=cls.__new__(cls)
            trainer.args=get_cfg(overrides=dict(imgsz=size,rect=False,cache=False,workers=0,batch=2,task='detect'))
            trainer.data={'names':{0:'visible_top'},'nc':1,'channels':3}
            for mode in ('train','val'):
                dataset=trainer.build_dataset(str(output/'images'/mode),mode,2)
                self.assertFalse(dataset.augment);self.assertFalse(dataset.rect)
                for i in range(len(dataset)):
                    sample=dataset[i];array=sample['img'].numpy()
                    key=Path(sample['im_file']).stem
                    record=next(row for row in self.manifest['images'] if row['image_id']==key)
                    expected,_=model_uint8(verified_rgb(self.root,record),size)
                    self.assertEqual(array.shape,(3,size,size));self.assertTrue(np.array_equal(array,expected[0]))
            self.assertEqual(receipt['real_optimizer_updates'],0)

    def fake_factory(self,output,budget=False,drift=False):
        def factory(ledger,guard):
            class Trainer:
                def __init__(self,overrides):self.save_dir=output/'fit'
                def train(self):
                    ledger.receipt['training_loop_started']=True
                    if budget:raise PilotBudgetReached('TEST_BUDGET')
                    # Synthetic callbacks validate receipt plumbing only.
                    ledger.after_step();ledger.receipt['completed_epochs']=1
                    (self.save_dir/'weights').mkdir(parents=True)
                    for name in ('best.pt','last.pt'):(self.save_dir/'weights'/name).write_bytes(b'UNIT_FIXTURE_NOT_A_MODEL')
                    if drift:self_outer.manifest_path.write_text('{}')
            return Trainer
        self_outer=self
        return factory

    def test_runner_uses_new_directory_receipts_and_checkpoint_readback(self):
        output=self.root/'run';old={k:os.environ.get(k) for k in ('YOLO_CONFIG_DIR','YOLO_OFFLINE')}
        with patch('tools.train_material_pilot.trainer_class',side_effect=self.fake_factory(output)):
            result=run_training(self.root,self.manifest_path,self.initial,output)
        self.assertEqual(result['status'],'TRAINING_FINISHED_NOT_TASK_ACCEPTED')
        self.assertEqual(result['real_optimizer_updates'],1)
        self.assertEqual({k:os.environ.get(k) for k in old},old)
        receipt,path,manifest=verify_training_run(self.root,output)
        self.assertEqual(path.name,'best.pt');self.assertEqual(receipt['dataset_manifest_sha256'],audit(self.root,manifest)['manifest_sha256'])
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError):verify_training_run(self.root,output)
        with self.assertRaises(ValueError):run_training(self.root,self.manifest_path,self.initial,output)

    def test_budget_or_source_change_never_becomes_completed_exportable_training(self):
        output=self.root/'budget'
        with patch('tools.train_material_pilot.trainer_class',side_effect=self.fake_factory(output,budget=True)):
            result=run_training(self.root,self.manifest_path,self.initial,output)
        self.assertEqual(result['status'],'BUDGET_REACHED_PARTIAL_NOT_EXPORTABLE')
        with self.assertRaises(ValueError):verify_training_run(self.root,output)

    @unittest.skipIf(importlib.util.find_spec('onnx') is None,'Isolated ONNX environment unavailable')
    def test_completed_training_export_provenance_binds_dataset_weights_and_metadata(self):
        import onnx
        from tools.material_model_contract import model_metadata,TRAINED,verify_training_provenance
        output=self.root/'run'
        with patch('tools.train_material_pilot.trainer_class',side_effect=self.fake_factory(output)):
            training=run_training(self.root,self.manifest_path,self.initial,output)
        export_dir=self.root/'export';export_dir.mkdir()
        tensor=onnx.numpy_helper.from_array(np.zeros((1,5,1029),np.float32))
        graph=onnx.helper.make_graph([onnx.helper.make_node('Constant',[],['out'],value=tensor)],'UNIT_FIXTURE_NOT_TASK',
            [onnx.helper.make_tensor_value_info('images',onnx.TensorProto.FLOAT,[1,3,224,224])],
            [onnx.helper.make_tensor_value_info('out',onnx.TensorProto.FLOAT,[1,5,1029])])
        model=onnx.helper.make_model(graph,opset_imports=[onnx.helper.make_opsetid('',12)],ir_version=8)
        receipt_sha=sha(output/'training.json');weight_sha=sha(output/'fit/weights/best.pt')
        metadata=dict(model_metadata(224,TRAINED),material_training_receipt_sha256=receipt_sha,
            material_dataset_manifest_sha256=training['dataset_manifest_sha256'],material_trained_checkpoint_sha256=weight_sha)
        for key,value in metadata.items():p=model.metadata_props.add();p.key=key;p.value=value
        path=export_dir/'material.onnx';onnx.save(model,str(path))
        export_receipt=dict(status='EXPORTED_NUMERIC_PASS_NOT_TASK_ACCEPTED',onnx_sha256=sha(path),checkpoint='best.pt',
            training_receipt_sha256=receipt_sha,trained_checkpoint_sha256=weight_sha,
            training_dataset_manifest_sha256=training['dataset_manifest_sha256'],real_optimizer_updates=1,
            numeric_limits={'rtol':1e-4,'atol':1e-4},comparisons=[dict(passed=True,pattern=name)
                for name in ('zero','mid','max','rgb_planes','noise')])
        write_receipt(export_dir/'export.json',export_receipt)
        verified=verify_training_provenance(self.root,self.manifest,path,output)
        self.assertEqual(verified['real_optimizer_updates'],1)
        with self.assertRaises(ValueError):verify_training_provenance(self.root,self.manifest,path,None)
        for changes in (dict(real_optimizer_updates=0),dict(onnx_sha256='0'*64),dict(comparisons=[]),
                        dict(status='EXPORT_STARTED'),dict(training_receipt_sha256='0'*64),
                        dict(comparisons=[dict(passed=True,pattern='zero')]*5)):
            write_receipt(export_dir/'export.json',dict(export_receipt,**changes))
            with self.assertRaises(ValueError):verify_training_provenance(self.root,self.manifest,path,output)
        write_receipt(export_dir/'export.json',export_receipt)
        (output/'fit/weights/best.pt').write_bytes(b'changed')
        with self.assertRaises(ValueError):verify_training_provenance(self.root,self.manifest,path,output)

    def test_task_export_refuses_incomplete_receipt_before_creating_onnx_or_loading_torch(self):
        from tools.export_trained_material_onnx import export
        output=self.root/'incomplete';output.mkdir()
        write_receipt(output/'training.json',dict(status='BUDGET_REACHED_PARTIAL_NOT_EXPORTABLE',real_optimizer_updates=0))
        target=self.root/'not-exported'
        with self.assertRaises(ValueError):export(self.root,output,target)
        self.assertFalse(target.exists())

    def test_persistent_manifest_drift_never_becomes_exportable_training(self):
        output=self.root/'drift'
        with patch('tools.train_material_pilot.trainer_class',side_effect=self.fake_factory(output,drift=True)):
            with self.assertRaises(ValueError):run_training(self.root,self.manifest_path,self.initial,output)
        receipt=json.loads((output/'training.json').read_text())
        self.assertEqual(receipt['status'],'FAILED_OR_UNKNOWN_NOT_EXPORTABLE')
        with self.assertRaises(ValueError):verify_training_run(self.root,output)

    def test_training_restores_process_environment_even_when_final_receipt_write_fails(self):
        from tools import train_material_pilot as module
        output=self.root/'receipt-failure'
        old={key:os.environ.get(key) for key in ('YOLO_CONFIG_DIR','YOLO_OFFLINE')}
        original=module.write_receipt
        def fail_final(path,receipt):
            if receipt.get('status')=='TRAINING_FINISHED_NOT_TASK_ACCEPTED':
                raise OSError('SYNTHETIC_FINAL_RECEIPT_FAILURE')
            original(path,receipt)
        with patch.object(module,'write_receipt',side_effect=fail_final), \
             patch.object(module,'trainer_class',side_effect=self.fake_factory(output)):
            with self.assertRaisesRegex(OSError,'SYNTHETIC_FINAL_RECEIPT_FAILURE'):
                module.run_training(self.root,self.manifest_path,self.initial,output)
        self.assertEqual({key:os.environ.get(key) for key in old},old)
        with self.assertRaises(ValueError):verify_training_run(self.root,output)

    @unittest.skipIf(importlib.util.find_spec('ultralytics') is None,'Pinned training environment unavailable')
    def test_export_restores_process_environment_even_when_error_receipt_write_fails(self):
        import torch
        from tools import export_trained_material_onnx as module
        output=self.root/'export-receipt-failure'
        run=self.root/'run-evidence';run.mkdir()
        write_receipt(run/'training.json',{'scope':'UNIT_FIXTURE_NOT_TRAINING'})
        old=os.environ.get('YOLO_CONFIG_DIR')
        old_mkldnn=torch.backends.mkldnn.enabled
        original=module.write_receipt
        fake_receipt=dict(dataset_manifest_sha256='a'*64,real_optimizer_updates=1)
        def fail_final(path,receipt):
            if receipt.get('status')=='FAILED_OR_UNKNOWN_NOT_CONVERTIBLE':
                raise OSError('SYNTHETIC_EXPORT_RECEIPT_FAILURE')
            original(path,receipt)
        with patch.object(module,'verify_training_run',return_value=(fake_receipt,self.initial,{})), \
             patch.object(module,'write_receipt',side_effect=fail_final), \
             patch('ultralytics.nn.tasks.load_checkpoint',side_effect=ValueError('SYNTHETIC_LOAD_FAILURE')):
            with self.assertRaisesRegex(OSError,'SYNTHETIC_EXPORT_RECEIPT_FAILURE'):
                module.export(self.root,run,output)
        self.assertEqual(os.environ.get('YOLO_CONFIG_DIR'),old)
        self.assertEqual(torch.backends.mkldnn.enabled,old_mkldnn)
        receipt=json.loads((output/'export.json').read_text())
        self.assertEqual(receipt['status'],'EXPORT_STARTED')
        self.assertFalse((output/'material.onnx').exists())


if __name__=='__main__':unittest.main()
