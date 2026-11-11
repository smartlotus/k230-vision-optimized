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
from tools.material_model_contract import (FIXTURE,TRAINED,compiler_settings,inspect_onnx,
                                           model_metadata,load_reviewed_calibration,require_nncase_version)
from tools.material_dataset import register,prepare_calibration
from tools.compile_material_kmodel import preflight,compile_model


class MaterialCompilerGateTests(unittest.TestCase):
    def test_exact_uint8_contract_has_only_one_normalization_and_no_output_permute(self):
        for size in (224,320):
            options=compiler_settings(size)
            self.assertEqual(options['input_shape'],[1,3,size,size])
            self.assertEqual(options['input_range'],[0,1])
            self.assertEqual(options['mean'],[0,0,0])
            self.assertEqual(options['std'],[1,1,1])
            self.assertEqual(options['input_type'],'uint8')
            self.assertFalse(options['swapRB'])
            self.assertEqual(options['output_layout'],'')  # YOLO output is rank3.
        for bad in (256,True,320.0):
            with self.assertRaises(ValueError):compiler_settings(bad)

    def test_compiler_and_plugin_versions_must_both_match_runtime(self):
        with patch('tools.material_model_contract.importlib.metadata.version',side_effect=['2.11.0','2.11.0']):
            self.assertEqual(set(require_nncase_version().values()),{'2.11.0'})
        for versions in (['2.10.0','2.11.0'],['2.11.0','2.12.0']):
            with patch('tools.material_model_contract.importlib.metadata.version',side_effect=versions):
                with self.assertRaises(ValueError):require_nncase_version()

    def test_current_unreviewed_dataset_cannot_enter_model_load_or_compile(self):
        root=Path(__file__).resolve().parents[1]
        m=json.loads((root/'data/ai_pilot/material_manifest_r3.json').read_text(encoding='utf-8'))
        with patch('tools.compile_material_kmodel.inspect_onnx') as inspect:
            with self.assertRaisesRegex(ValueError,'Reviewed grouped'):
                preflight(root,m,root/'no-trained-model.onnx',root/'no-approved-calibration',320)
            inspect.assert_not_called()


@unittest.skipIf(importlib.util.find_spec('onnx') is None,'ONNX fixture tests require isolated conversion environment')
class MaterialOnnxContractTests(unittest.TestCase):
    def setUp(self):
        import onnx
        self.onnx=onnx
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def model(self,kind=TRAINED,size=224,mutate=None):
        o=self.onnx;h=o.helper
        count=sum((size//s)**2 for s in (8,16,32))
        constant=o.numpy_helper.from_array(np.zeros((1,5,count),np.float32),name='constant')
        graph=h.make_graph([h.make_node('Constant',[],['output0'],value=constant)],'TEST_FIXTURE_NOT_REAL_MODEL',
            [h.make_tensor_value_info('images',o.TensorProto.FLOAT,[1,3,size,size])],
            [h.make_tensor_value_info('output0',o.TensorProto.FLOAT,[1,5,count])])
        model=h.make_model(graph,opset_imports=[h.make_opsetid('',12)],ir_version=8)
        for key,value in model_metadata(size,kind).items():
            entry=model.metadata_props.add();entry.key=key;entry.value=value
        if mutate:mutate(model)
        path=self.root/'model.onnx';o.save(model,str(path));return path

    def reviewed(self):
        sources=[]
        for i,value in enumerate((0xf800,0x07e0,0x001f,0xffff)):
            raw=self.root/('%d.rgb565'%i);raw.write_bytes(np.full((360,640),value,dtype='<u2').tobytes())
            jpg=self.root/('%d.jpg'%i);cv2.imwrite(str(jpg),np.full((360,640,3),i*60,np.uint8))
            meta=self.root/('%d.json'%i);meta.write_text(json.dumps(dict(source='REAL_CAMERA_NATIVE_LAB',
                width=640,height=360,camera_id=2,is_simulated=False,
                frame_id=i+1,timestamp_ms=i*100+100)))
            sources.append(dict(jpeg_path=jpg.name,rgb565_path=raw.name,metadata_path=meta.name))
        manifest=register(self.root,dict(object_set_final_confirmed_by_operator=True,sources=sources))
        for record,split in zip(manifest['images'],('train','train','val','test')):
            record.update(scene_group_id='HOST_TEST_ONLY_'+record['image_id'],scene_group_confirmed=True,split=split)
            record['review'].update(status='APPROVED',reviewer='SYNTHETIC_UNIT_TEST_NOT_REAL_GT',
                all_visible_tops_checked=True,objects=[dict(class_id=0,bbox_xywh=[100,80,40,20],color_id=1)])
        return manifest

    def test_inspects_fixed_material_geometry_and_rejects_random_fixture_as_task_model(self):
        path=self.model()
        report=inspect_onnx(path,224)
        self.assertEqual(report['input_shape'],[1,3,224,224])
        self.assertEqual(report['output_shape'],[1,5,1029])
        self.assertFalse(report['task_accuracy_verified'])
        path=self.model(FIXTURE)
        with self.assertRaises(ValueError):inspect_onnx(path,224)
        self.assertEqual(inspect_onnx(path,224,FIXTURE)['artifact_kind'],FIXTURE)

    def test_dynamic_batch_layout_classes_dtype_and_opset_are_rejected(self):
        def dynamic(m):
            dim=m.graph.input[0].type.tensor_type.shape.dim[0];dim.ClearField('dim_value');dim.dim_param='batch'
        def batch(m):m.graph.input[0].type.tensor_type.shape.dim[0].dim_value=2
        def layout(m):m.graph.input[0].type.tensor_type.shape.dim[1].dim_value=224
        def classes(m):m.graph.output[0].type.tensor_type.shape.dim[1].dim_value=6
        def dtype(m):m.graph.input[0].type.tensor_type.elem_type=self.onnx.TensorProto.UINT8
        def opset(m):m.opset_import[0].version=13
        for mutate in (dynamic,batch,layout,classes,dtype,opset):
            with self.subTest(mutate=mutate.__name__),self.assertRaises(ValueError):inspect_onnx(self.model(mutate=mutate),224)

    def test_metadata_and_nms_custom_or_control_flow_never_gain_admission(self):
        def missing(m):m.ClearField('metadata_props')
        def duplicate(m):m.metadata_props.append(copy.deepcopy(m.metadata_props[0]))
        for mutate in (missing,duplicate):
            with self.assertRaises(ValueError):inspect_onnx(self.model(mutate=mutate),224)
        for op in ('NonMaxSuppression','If','Loop','Scan'):
            def add(m,op=op):m.graph.node.append(self.onnx.helper.make_node(op,[],['unused']))
            with self.assertRaises(ValueError):inspect_onnx(self.model(mutate=add),224)
        def custom(m):m.graph.node.append(self.onnx.helper.make_node('Unknown',[],['unused'],domain='vendor.custom'))
        with self.assertRaises(ValueError):inspect_onnx(self.model(mutate=custom),224)

    def test_external_or_nonfinite_tensor_constants_are_rejected_before_execution(self):
        def external(m):
            tensor=m.graph.node[0].attribute[0].t
            tensor.ClearField('raw_data');tensor.data_location=self.onnx.TensorProto.EXTERNAL
            entry=tensor.external_data.add();entry.key='location';entry.value='../not-read.bin'
        def nonfinite(m):
            tensor=m.graph.node[0].attribute[0].t
            data=self.onnx.numpy_helper.to_array(tensor).copy();data[0,0,0]=float('nan')
            tensor.CopyFrom(self.onnx.numpy_helper.from_array(data))
        for mutate in (external,nonfinite):
            with self.assertRaises(ValueError):inspect_onnx(self.model(mutate=mutate),224)

    def test_calibration_samples_are_exact_train_only_uint8_and_not_hardcoded_five(self):
        manifest=self.reviewed();folder=self.root/'calib'
        prepared=prepare_calibration(self.root,manifest,folder,224)
        self.assertEqual(prepared['samples_count'],2)
        arrays,receipt=load_reviewed_calibration(self.root,manifest,folder,224)
        self.assertEqual(len(arrays),2)
        self.assertTrue(all(a[0].dtype==np.uint8 and a[0].shape==(1,3,224,224) for a in arrays))
        self.assertTrue(all(r['split']=='train' for r in receipt['samples']))
        with self.assertRaises(ValueError):load_reviewed_calibration(self.root,manifest,folder,320)
        plan,loaded=preflight(self.root,manifest,self.model(),folder,224)
        self.assertEqual(plan['samples_count'],len(loaded))
        self.assertEqual(plan['compiler_options']['input_range'],[0,1])
        self.assertFalse(plan['compiler_executed'])

    def test_prepared_calibration_and_review_drift_reject_before_output(self):
        manifest=self.reviewed();folder=self.root/'calib'
        prepared=prepare_calibration(self.root,manifest,folder,224)
        path=folder/prepared['samples'][0]['file']
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError):preflight(self.root,manifest,self.model(),folder,224)
        changed=copy.deepcopy(manifest);changed['images'][0]['review']['reviewer']='changed'
        manifest_path=self.root/'reviewed.json';manifest_path.write_text(json.dumps(changed))
        target=self.root/'compiled'
        with self.assertRaisesRegex(ValueError,'changed before'):
            compile_model(self.root,manifest,self.model(),folder,224,target,manifest_path)
        self.assertFalse(target.exists())

    def fake_backend(self,initialize_error=False,compile_action=None):
        state={}
        class Options:pass
        class PTQ:
            def set_tensor_data(self,arrays):state['arrays']=arrays
        class Compiler:
            def __init__(self,options):
                if initialize_error:raise RuntimeError('HOST_FIXTURE_INITIALIZE_ERROR')
                state['options']=options
            def import_onnx(self,data,options):state['model_bytes']=data
            def use_ptq(self,ptq):state['ptq']=ptq
            def compile(self):
                if compile_action:compile_action()
            def gencode_tobytes(self):return b'HOST_TEST_NOT_A_REAL_KMODEL'
        return types.SimpleNamespace(CompileOptions=Options,Compiler=Compiler,PTQTensorOptions=PTQ,ImportOptions=Options),state

    def test_converter_records_actual_stage_and_sample_count_not_task_acceptance(self):
        manifest=self.reviewed();folder=self.root/'calib'
        prepare_calibration(self.root,manifest,folder,224)
        manifest_path=self.root/'reviewed.json';manifest_path.write_text(json.dumps(manifest))
        model=self.model();backend,state=self.fake_backend()
        target=self.root/'compiled';before=Path.cwd();old_env={k:os.environ.get(k) for k in ('PATH','NNCASE_PLUGIN_PATH')}
        with patch.dict('sys.modules',{'nncase':backend}), \
             patch('tools.compile_material_kmodel.verify_training_provenance',return_value={'scope':'MOCK_ONLY'}), \
             patch('tools.compile_material_kmodel.require_nncase_version',return_value={'nncase':'2.11.0','nncase-kpu':'2.11.0'}), \
             patch('tools.compile_material_kmodel.importlib.util.find_spec',return_value=types.SimpleNamespace(origin=str(self.root/'_nncase.pyd'))):
            report=compile_model(self.root,manifest,model,folder,224,target,manifest_path)
        self.assertEqual(Path.cwd(),before)
        self.assertEqual({k:os.environ.get(k) for k in old_env},old_env)
        self.assertEqual(report['status'],'COMPILED_NOT_TASK_ACCEPTED')
        self.assertTrue(report['compiler_compile_called']);self.assertTrue(report['compile_completed'])
        self.assertFalse(report['task_accuracy_verified'])
        self.assertEqual(state['ptq'].samples_count,2)
        self.assertEqual(len(state['arrays']),2)
        self.assertEqual(state['options'].input_range,[0,1])
        backend,_=self.fake_backend(initialize_error=True);failed=self.root/'failed'
        with patch.dict('sys.modules',{'nncase':backend}), \
             patch('tools.compile_material_kmodel.verify_training_provenance',return_value={'scope':'MOCK_ONLY'}), \
             patch('tools.compile_material_kmodel.require_nncase_version',return_value={'nncase':'2.11.0','nncase-kpu':'2.11.0'}), \
             patch('tools.compile_material_kmodel.importlib.util.find_spec',return_value=types.SimpleNamespace(origin=str(self.root/'_nncase.pyd'))):
            with self.assertRaises(RuntimeError):compile_model(self.root,manifest,model,folder,224,failed,manifest_path)
        result=json.loads((failed/'conversion.json').read_text())
        self.assertEqual(result['status'],'FAILED_OR_UNKNOWN_NO_ACCEPTANCE')
        self.assertFalse(result['compiler_compile_called']);self.assertFalse(result['compiler_executed'])
        self.assertFalse((failed/'material.kmodel').exists())
        self.assertEqual({k:os.environ.get(k) for k in old_env},old_env)

    def test_review_change_during_compile_keeps_failure_and_no_delivered_model(self):
        manifest=self.reviewed();folder=self.root/'calib'
        prepare_calibration(self.root,manifest,folder,224)
        manifest_path=self.root/'reviewed.json';manifest_path.write_text(json.dumps(manifest))
        def drift():
            changed=copy.deepcopy(manifest);changed['images'][0]['review']['reviewer']='CHANGED'
            manifest_path.write_text(json.dumps(changed))
        backend,_=self.fake_backend(compile_action=drift);target=self.root/'drifted'
        with patch.dict('sys.modules',{'nncase':backend}), \
             patch('tools.compile_material_kmodel.verify_training_provenance',return_value={'scope':'MOCK_ONLY'}), \
             patch('tools.compile_material_kmodel.require_nncase_version',return_value={'nncase':'2.11.0','nncase-kpu':'2.11.0'}), \
             patch('tools.compile_material_kmodel.importlib.util.find_spec',return_value=types.SimpleNamespace(origin=str(self.root/'_nncase.pyd'))):
            with self.assertRaises(ValueError):compile_model(self.root,manifest,self.model(),folder,224,target,manifest_path)
        receipt=json.loads((target/'conversion.json').read_text())
        self.assertTrue(receipt['compile_completed'])
        self.assertEqual(receipt['status'],'FAILED_OR_UNKNOWN_NO_ACCEPTANCE')
        self.assertFalse((target/'material.kmodel').exists())


if __name__=='__main__':unittest.main()
