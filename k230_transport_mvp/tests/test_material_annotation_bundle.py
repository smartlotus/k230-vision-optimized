import base64,copy,json,tempfile,unittest,subprocess,sys
from pathlib import Path
import cv2
import numpy as np
from tools.material_dataset import register,audit
from tools.material_annotation_bundle import make_bundle,import_bundle,read_rectangle,write_windows_launcher


class MatureAnnotationExchangeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        raw=np.full((360,640),0xf800,dtype='<u2').tobytes();(self.root/'frame.rgb565').write_bytes(raw)
        cv2.imwrite(str(self.root/'frame.jpg'),np.zeros((360,640,3),np.uint8))
        (self.root/'frame.json').write_text(json.dumps({'source':'REAL_CAMERA_NATIVE_LAB','camera_id':2,
            'is_simulated':False,'width':640,'height':360,'frame_id':1,'timestamp_ms':100}))
        self.manifest=register(self.root,{'object_set_final_confirmed_by_operator':True,'sources':[
            {'jpeg_path':'frame.jpg','rgb565_path':'frame.rgb565','metadata_path':'frame.json'}]})
        self.key=self.manifest['images'][0]['image_id']
        self.drafts={'status':'DRAFT_ASSISTED','human_approved':False,'images':[
            {'image_id':self.key,'source_native_sha256':self.key,'objects':[
                {'class_id':0,'bbox_xywh':[100,80,40,30],'color_id':1,'uncertain':False}]}]}

    def bundle(self,format='xanylabeling'):
        path=self.root/format;make_bundle(self.root,self.manifest,self.drafts,path,format)
        return path,path/'images'/(self.key+'.json')

    def test_xany_four_corner_export_and_checked_flag_do_not_create_ground_truth(self):
        path,label=self.bundle();document=json.loads(label.read_text())
        self.assertEqual(document['shapes'][0]['points'],[[100,80],[140,80],[140,110],[100,110]])
        self.assertEqual(document['shapes'][0]['label'],'visible_top');self.assertFalse(document['checked'])
        document['checked']=True;label.write_text(json.dumps(document))
        result,receipt=import_bundle(self.root,self.manifest,path)
        self.assertEqual(receipt['human_approved_images'],0);self.assertEqual(audit(self.root,result)['eligible_images'],0)
        self.assertEqual(result['images'][0]['review']['status'],'UNREVIEWED')

    def test_operator_review_is_explicit_and_does_not_invent_scene_group_or_split(self):
        path,label=self.bundle()
        result,receipt=import_bundle(self.root,self.manifest,path,reviewer='TEST_FIXTURE_NOT_OPERATOR',reviewed_ids=[self.key])
        self.assertEqual(receipt['human_approved_images'],1)
        self.assertEqual(result['images'][0]['review']['objects'][0]['bbox_xywh'],[100,80,40,30])
        report=audit(self.root,result);self.assertEqual(report['eligible_images'],0)
        self.assertIn('SCENE_GROUP_UNCONFIRMED',report['rows'][0]['reasons'])
        self.assertIn('SPLIT_UNASSIGNED',report['rows'][0]['reasons'])
        with self.assertRaises(ValueError):import_bundle(self.root,self.manifest,path,reviewed_ids=[self.key])

    def test_uncertain_four_objects_missing_labels_and_wrong_pixels_are_rejected(self):
        path,label=self.bundle();document=json.loads(label.read_text());original=copy.deepcopy(document)
        document['shapes'][0]['flags']['uncertain']=True;label.write_text(json.dumps(document))
        with self.assertRaises(ValueError):import_bundle(self.root,self.manifest,path,reviewer='TEST',reviewed_ids=[self.key])
        document=copy.deepcopy(original);document['shapes']*=4;label.write_text(json.dumps(document))
        with self.assertRaises(ValueError):import_bundle(self.root,self.manifest,path,reviewer='TEST',reviewed_ids=[self.key])
        document=copy.deepcopy(original);document.pop('shapes');label.write_text(json.dumps(document))
        with self.assertRaises(ValueError):import_bundle(self.root,self.manifest,path)
        label.write_text(json.dumps(original))
        (path/'images'/(self.key+'.png')).write_bytes(b'changed')
        with self.assertRaises(ValueError):import_bundle(self.root,self.manifest,path)

    def test_bundle_cli_imports_from_an_unrelated_working_directory(self):
        script=Path(__file__).resolve().parents[1]/'tools/material_annotation_bundle.py'
        result=subprocess.run([sys.executable,'-B','-X','utf8',str(script),'--help'],cwd=self.root,
                              capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('colored_surfaces',result.stdout)

    def test_windows_launcher_has_literal_paths_and_preserves_saved_annotations(self):
        path,label=self.bundle()
        document=json.loads(label.read_text());document['checked']=True
        label.write_text(json.dumps(document))
        before={p.relative_to(path).as_posix():p.read_bytes() for p in path.rglob('*')
                if p.is_file() and p.name!='start-labeling.cmd'}
        launcher=write_windows_launcher(self.root,path)
        data=launcher.read_bytes()
        self.assertTrue(data.startswith(b'@echo off\r\n'))
        self.assertNotIn(b'\x0b',data)  # A Python \v escape broke the previous launcher.
        self.assertNotIn(b'\t',data)    # \tools must not turn into a tab.
        self.assertIn(b'/.ops/local/venvs/kpu-label/Scripts/python.exe',data)
        self.assertIn(b'/k230_transport_mvp/tools/launch_material_labeling.py',data)
        self.assertIn(b'--bundle "%~dp0."',data)
        self.assertEqual(before,{p.relative_to(path).as_posix():p.read_bytes() for p in path.rglob('*')
                                if p.is_file() and p.name!='start-labeling.cmd'})

    def test_labelme_diagonal_and_embedded_png_roundtrip_remain_source_bound(self):
        path,label=self.bundle('labelme');document=json.loads(label.read_text())
        self.assertEqual(document['shapes'][0]['points'],[[100,80],[140,110]])
        document['imageData']=base64.b64encode((path/'images'/(self.key+'.png')).read_bytes()).decode()
        label.write_text(json.dumps(document))
        result,_=import_bundle(self.root,self.manifest,path)
        self.assertEqual(result['images'][0]['review']['objects'][0]['bbox_xywh'],[100,80,40,30])
        document['imageData']=base64.b64encode(b'not an image').decode();label.write_text(json.dumps(document))
        with self.assertRaises(ValueError):import_bundle(self.root,self.manifest,path)

    def test_polygons_unknown_classes_nonfinite_points_and_draft_source_drift_fail(self):
        shape={'label':'visible_top','shape_type':'rectangle','points':[[1,2],[4,5]],'flags':{}}
        for changed in (dict(shape,label='paper_target'),dict(shape,points=[[1,float('nan')],[4,5]]),
                        dict(shape,points=[[1,2],[4,2],[3,5],[1,5]]),dict(shape,flags={'uncertain':'false'})):
            with self.subTest(changed=changed),self.assertRaises(ValueError):read_rectangle(changed)
        drafts=copy.deepcopy(self.drafts);drafts['images'][0]['source_native_sha256']='0'*64
        with self.assertRaises(ValueError):make_bundle(self.root,self.manifest,drafts,self.root/'bad')


if __name__=='__main__':unittest.main()
