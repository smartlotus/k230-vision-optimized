import copy,json,tempfile,unittest
from pathlib import Path
import cv2
import numpy as np
from tools.material_dataset import register,audit,review_gate,export_yolo,prepare_calibration,verify_calibration
from tools.material_preprocess import model_uint8,float_reference,rgb565_to_rgb
from vision.yolo_candidates import input_contract,decode_material_proposals,decode_two_class_proposals


class MaterialPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def source(self,name,value,frame=1):
        (self.root/(name+'.rgb565')).write_bytes(np.full((360,640),value,dtype='<u2').tobytes())
        cv2.imwrite(str(self.root/(name+'.jpg')),np.full((360,640,3),80,np.uint8))
        (self.root/(name+'.json')).write_text(json.dumps({'source':'REAL_CAMERA_NATIVE_LAB','is_simulated':False,
            'camera_id':2,'width':640,'height':360,'frame_id':frame,'timestamp_ms':1000+frame}))
        return {'jpeg_path':name+'.jpg','rgb565_path':name+'.rgb565','metadata_path':name+'.json'}

    def manifest(self):
        return register(self.root,{'object_set_final_confirmed_by_operator':True,
            'sources':[self.source('a',0xf800),self.source('b',0x07e0,2),self.source('c',0x001f,3)]})

    def approve(self,record,group,split,objects=None):
        record.update(scene_group_id=group,scene_group_confirmed=True,split=split)
        record['review'].update(status='APPROVED',reviewer='HOST_TEST_FIXTURE_NOT_REAL_GT',
            all_visible_tops_checked=True,objects=objects if objects is not None else
            [{'class_id':0,'bbox_xywh':[100,80,40,20],'color_id':1}])

    def approved(self):
        manifest=self.manifest()
        for record,split in zip(manifest['images'],('train','val','test')):self.approve(record,'group-'+split,split)
        return manifest

    def test_registration_deduplicates_native_bytes_across_jpeg_and_capture_identity(self):
        a=self.source('a',0xf800);b=self.source('b',0xf800,123)
        cv2.imwrite(str(self.root/'b.jpg'),np.full((360,640,3),90,np.uint8),[cv2.IMWRITE_JPEG_QUALITY,40])
        index={'object_set_final_confirmed_by_operator':True,'sources':[a,b]}
        first=register(self.root,index);second=register(self.root,dict(index,sources=[b,a]))
        self.assertEqual(first,second);self.assertEqual(len(first['images']),1)
        self.assertEqual(len(first['images'][0]['captures']),2)
        self.assertFalse(first['images'][0]['captures'][0]['metadata_has_lab_grid'])
        self.assertEqual(audit(self.root,first)['eligible_images'],0)

    def test_missing_or_incomplete_human_labels_never_become_empty_negatives(self):
        record=self.manifest()['images'][0]
        self.assertIn('HUMAN_COMPLETE_REVIEW_REQUIRED',review_gate(record)[0])
        self.approve(record,'group','train',objects=[])
        self.assertEqual(review_gate(record),([],[]))
        for field,value in (('all_visible_tops_checked',False),('objects',None),('status','UNREVIEWED')):
            changed=copy.deepcopy(record);changed['review'][field]=value
            self.assertIn('HUMAN_COMPLETE_REVIEW_REQUIRED',review_gate(changed)[0])
        changed=copy.deepcopy(record);changed['review']['uncertain']=True
        self.assertIn('UNCERTAIN_FRAME_EXCLUDED',review_gate(changed)[0])
        changed=copy.deepcopy(record);changed['review']['ignore']=True
        self.assertIn('UNCERTAIN_FRAME_EXCLUDED',review_gate(changed)[0])
        changed=copy.deepcopy(record);changed['review']['source_native_sha256']='0'*64
        self.assertIn('ANNOTATION_SOURCE_MISMATCH',review_gate(changed)[0])

    def test_source_paths_groups_and_bytes_cannot_be_rebound_silently(self):
        m=self.approved();m['images'][1]['scene_group_id']=m['images'][0]['scene_group_id']
        with self.assertRaises(ValueError):audit(self.root,m)
        m=self.approved();entry=m['images'][0]['captures'][0]['files']['metadata']
        (self.root/entry['path']).write_text('{}')
        with self.assertRaises(ValueError):audit(self.root,m)
        with self.assertRaises(ValueError):register(self.root,{'object_set_final_confirmed_by_operator':True,
            'sources':[{'jpeg_path':'../outside.jpg','rgb565_path':'a.rgb565','metadata_path':'a.json'}]})

    def test_adjacent_frames_from_one_bounded_series_cannot_be_split_by_renaming_groups(self):
        m=self.approved()
        for record in m['images'][:2]:record['captures'][0]['capture_series_id']='one-short-capture-series'
        self.assertNotEqual(m['images'][0]['scene_group_id'],m['images'][1]['scene_group_id'])
        with self.assertRaises(ValueError):audit(self.root,m)

    def test_material_only_boxes_preserve_clipping_reject_paper_and_do_not_truncate_four(self):
        r=self.manifest()['images'][0];self.approve(r,'group','train')
        r['review']['objects'][0]['bbox_xywh']=[-5,10,15,20]
        reasons,objects=review_gate(r);self.assertEqual(reasons,[])
        self.assertEqual(objects[0]['bbox_xywh'],[0,10,10,20]);self.assertTrue(objects[0]['bbox_clipped'])
        r['review']['objects'][0]['geometry_complete_verified']=True
        with self.assertRaises(ValueError):review_gate(r)
        r['review']['objects']=[{'class_id':1,'bbox_xywh':[1,2,3,4]}]
        with self.assertRaises(ValueError):review_gate(r)
        r['review']['objects']=[{'class_id':0,'bbox_xywh':[1,2,3,4]}]*4
        reasons,objects=review_gate(r);self.assertEqual(len(objects),4)
        self.assertIn('MORE_THAN_THREE_TOPS_REQUIRES_SEPARATE_SCENE_REVIEW',reasons)
        for box in ([1,2,0,5],[1,float('nan'),4,5],[700,20,10,10]):
            r['review']['objects']=[{'class_id':0,'bbox_xywh':box}]
            with self.assertRaises(ValueError):review_gate(r)

    def test_export_requires_approved_splits_and_preserves_confirmed_empty_labels(self):
        manifest=self.manifest();target=self.root/'export'
        with self.assertRaises(ValueError):export_yolo(self.root,manifest,target)
        self.assertFalse(target.exists())
        for r,split in zip(manifest['images'],('train','val','test')):
            self.approve(r,'group-'+split,split,objects=[] if split=='val' else None)
        report=export_yolo(self.root,manifest,target)
        self.assertEqual(report['eligible_images'],3)
        self.assertEqual(next((target/'labels/val').glob('*.txt')).read_text(),'')
        values=next((target/'labels/train').glob('*.txt')).read_text().split()
        self.assertEqual(values[0],'0');np.testing.assert_allclose(list(map(float,values[1:])),[.1875,.25,.0625,20/360],atol=1e-8)
        self.assertNotIn('paper_target',(target/'dataset.yaml').read_text())
        with self.assertRaises(ValueError):export_yolo(self.root,manifest,target)

    def test_calibration_uses_only_reviewed_train_and_detects_tampering_and_split_drift(self):
        m=self.approved();folder=self.root/'calibration'
        receipt=prepare_calibration(self.root,m,folder,size=224)
        self.assertEqual(receipt['samples_count'],1);self.assertEqual(receipt['samples'][0]['split'],'train')
        self.assertEqual(receipt['samples'][0]['shape'],[1,3,224,224])
        self.assertTrue(verify_calibration(self.root,m,folder)['verified_host_preparation'])
        self.assertFalse(receipt['native_ai2d_equivalence_verified'])
        changed=copy.deepcopy(m);changed['images'][0]['review']['reviewer']='CHANGED'
        with self.assertRaises(ValueError):verify_calibration(self.root,changed,folder)
        f=folder/receipt['samples'][0]['file'];f.write_bytes(b'changed')
        with self.assertRaises(ValueError):verify_calibration(self.root,m,folder)

    def test_all_empty_labels_are_valid_negatives_but_cannot_be_the_whole_training_set(self):
        manifest=self.approved()
        for record in manifest['images']:record['review']['objects']=[]
        report=audit(self.root,manifest)
        self.assertEqual(report['eligible_images'],3);self.assertFalse(report['export_ready'])
        with self.assertRaises(ValueError):export_yolo(self.root,manifest,self.root/'empty-train')

    def test_mixed_calibration_sizes_cannot_pass_as_one_model_input(self):
        manifest=self.approved()
        a=self.root/'cal224';b=self.root/'cal320'
        first=prepare_calibration(self.root,manifest,a,224);second=prepare_calibration(self.root,manifest,b,320)
        first['samples'][0]=second['samples'][0]
        (a/second['samples'][0]['file']).write_bytes((b/second['samples'][0]['file']).read_bytes())
        (a/'calibration-manifest.json').write_text(json.dumps(first))
        with self.assertRaises(ValueError):verify_calibration(self.root,manifest,a)

    def test_letterbox_rgb_layout_and_single_normalization_have_known_pixels(self):
        rgb=np.empty((360,640,3),np.uint8);rgb[:]=[0,128,255]
        rgb[:40,:40]=[255,0,0];rgb[-40:,-40:]=[0,255,0]
        for size,top in ((224,49),(320,70)):
            tensor,contract=model_uint8(rgb,size)
            self.assertEqual(tensor.shape,(1,3,size,size));self.assertTrue(tensor.flags.c_contiguous)
            np.testing.assert_array_equal(tensor[0,:,0,0],[114,114,114])
            np.testing.assert_array_equal(tensor[0,:,top+2,2],[255,0,0])
            np.testing.assert_array_equal(tensor[0,:,size-top-3,-3],[0,255,0])
            np.testing.assert_array_equal(tensor[0,:,size//2,size//2],[0,128,255])
            np.testing.assert_allclose(float_reference(tensor)[0,:,size//2,size//2],[0,128/255,1],atol=1e-7)
            self.assertEqual(contract['pad_tblr'],[top,top,0,0])
            with self.assertRaises(ValueError):float_reference(float_reference(tensor))
        red=rgb565_to_rgb(np.full((360,640),0xf800,dtype='<u2').tobytes())
        np.testing.assert_array_equal(red[0,0],[248,0,0])

    def test_single_material_output_cannot_be_mistaken_for_two_class_model(self):
        contract=input_contract(224);raw=np.zeros((1,5,1029),np.float32)
        raw[0,:,0]=[112,112,28,21,.9];calls=[]
        def native(values,*args):calls.append(values.shape);return [[[280,150,80,60]],[0],[.9]]
        result=decode_material_proposals(raw,contract,native=native,np=np)
        self.assertEqual(calls,[(1029,5)]);self.assertEqual(result['native_counts'],[1])
        self.assertEqual([p['class_name'] for p in result['proposals']],['visible_top'])
        with self.assertRaises(ValueError):decode_two_class_proposals(raw,contract,native=native,np=np)


if __name__=='__main__':unittest.main()
