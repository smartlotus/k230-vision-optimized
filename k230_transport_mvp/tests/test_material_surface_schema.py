"""Color/surface editor labels cannot silently expand the top-model classes."""
import copy,json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import cv2
import numpy as np
from tools.material_annotation_schema import SURFACE_LABELS,SURFACE_SCHEMA,LABEL_MAP,label_for
from tools.material_annotation_bundle import make_bundle,import_bundle,read_surface_rectangle
from tools.material_dataset import register,audit


def shape(color=1,surface='top',group=None,x=100):
    return dict(label=label_for(color,surface),shape_type='rectangle',
                points=[[x,100],[x+30,100],[x+30,120],[x,120]],group_id=group,flags={},attributes={})


class MaterialSurfaceSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        (self.root/'source.rgb565').write_bytes(np.full((360,640),0xf800,dtype='<u2').tobytes())
        cv2.imwrite(str(self.root/'source.jpg'),np.zeros((360,640,3),np.uint8))
        (self.root/'source.json').write_text(json.dumps(dict(source='REAL_CAMERA_NATIVE_LAB',is_simulated=False,
            camera_id=2,width=640,height=360,frame_id=1,timestamp_ms=100)))
        self.manifest=register(self.root,dict(object_set_final_confirmed_by_operator=True,sources=[
            dict(jpeg_path='source.jpg',rgb565_path='source.rgb565',metadata_path='source.json')]))
        self.key=self.manifest['images'][0]['image_id']
        self.drafts=dict(status='DRAFT_ASSISTED',human_approved=False,images=[dict(image_id=self.key,
            source_native_sha256=self.key,objects=[dict(class_id=0,bbox_xywh=[100,100,30,20],color_id=1)])])

    def bundle(self):
        folder=self.root/'colored';result=make_bundle(self.root,self.manifest,self.drafts,folder,preset='colored_surfaces')
        self.assertEqual(result['annotation_labels'],list(SURFACE_LABELS))
        return folder,folder/'images'/(self.key+'.json')

    def test_all_twelve_labels_have_explicit_color_and_surface_mapping(self):
        self.assertEqual(len(SURFACE_LABELS),12)
        self.assertEqual(len(set(SURFACE_LABELS)),12)
        for color in range(1,7):
            for surface in ('top','side'):
                result=read_surface_rectangle(shape(color,surface,1))
                self.assertEqual(result['color_id'],color);self.assertEqual(result['surface'],surface)
                self.assertEqual(result['object_group_id'],1)
                if surface=='top':self.assertEqual(result['class_id'],0)
                else:self.assertNotIn('class_id',result)
        self.assertEqual(label_for(6,'side'),'浅蓝色侧面')

    def test_package_preserves_training_class_and_maps_only_unapproved_top_drafts(self):
        folder,label=self.bundle();package=json.loads((folder/'package.json').read_text())
        self.assertEqual(package['classes'],['visible_top'])
        self.assertEqual(package['annotation_schema'],SURFACE_SCHEMA)
        document=json.loads(label.read_text());self.assertEqual(document['shapes'][0]['label'],'红色顶面')
        self.assertEqual(document['shapes'][0]['attributes'],{})
        result,receipt=import_bundle(self.root,self.manifest,folder)
        self.assertEqual(receipt['human_approved_images'],0)
        self.assertEqual(result['images'][0]['review']['objects'][0]['class_id'],0)
        self.assertEqual(result['images'][0]['side_annotations']['objects'],[])
        self.assertEqual(audit(self.root,result)['eligible_images'],0)

    def test_three_tops_and_three_sides_are_three_training_targets_not_six(self):
        folder,label=self.bundle();doc=json.loads(label.read_text())
        doc['shapes']=[shape(color,surface,group,x=80+group*100)
                       for color,group in ((1,1),(4,2),(5,3)) for surface in ('top','side')]
        label.write_text(json.dumps(doc))
        result,_=import_bundle(self.root,self.manifest,folder,reviewer='UNIT_FIXTURE_NOT_REAL_GT',reviewed_ids=[self.key])
        record=result['images'][0]
        self.assertEqual(len(record['review']['objects']),3)
        self.assertEqual([o['color_id'] for o in record['review']['objects']],[1,4,5])
        self.assertEqual(len(record['side_annotations']['objects']),3)
        self.assertFalse(record['side_annotations']['training_enabled'])
        self.assertFalse(record['side_annotations']['all_visible_sides_checked'])
        self.assertEqual(record['side_annotations']['status'],'REVIEWED_AUXILIARY_ONLY')

    def test_side_only_is_empty_top_only_after_explicit_human_review(self):
        folder,label=self.bundle();doc=json.loads(label.read_text());doc['shapes']=[shape(1,'side',1)]
        label.write_text(json.dumps(doc))
        result,_=import_bundle(self.root,self.manifest,folder)
        self.assertEqual(result['images'][0]['review']['status'],'UNREVIEWED')
        self.assertEqual(result['images'][0]['review']['objects'],[])
        result,_=import_bundle(self.root,self.manifest,folder,reviewer='UNIT_FIXTURE_NOT_REAL_GT',reviewed_ids=[self.key])
        self.assertEqual(result['images'][0]['review']['status'],'APPROVED')
        self.assertEqual(result['images'][0]['review']['objects'],[])
        self.assertFalse(audit(self.root,result)['export_ready'])  # Groups remain unknown.

    def test_unknown_labels_polygons_bad_groups_and_explicit_color_conflict_reject(self):
        for row in (dict(shape(),label='紫色顶面'),dict(shape(),shape_type='polygon'),dict(shape(),group_id=True),
                    dict(shape(),group_id=0),dict(shape(),group_id=256),dict(shape(),attributes={'color_id':4}),
                    dict(shape(),points=[[1,float('nan')],[4,5]])):
            with self.subTest(row=row),self.assertRaises(ValueError):read_surface_rectangle(row)
        for color in (True,0,7,None):
            with self.assertRaises(ValueError):label_for(color,'top')

    def test_same_group_color_conflict_duplicate_top_and_uncertainty_reject_approval(self):
        for i,rows in enumerate(([shape(1,'top',1),shape(4,'side',1)],
                                  [shape(1,'top',1),shape(1,'top',1,x=200)],
                                  [shape(1,'top',1),dict(shape(1,'side',1),flags={'uncertain':True})])):
            folder=self.root/f'case-{i}'
            make_bundle(self.root,self.manifest,self.drafts,folder,preset='colored_surfaces')
            label=folder/'images'/(self.key+'.json');doc=json.loads(label.read_text());doc['shapes']=rows
            label.write_text(json.dumps(doc))
            with self.assertRaises(ValueError):import_bundle(self.root,self.manifest,folder,
                reviewer='UNIT_FIXTURE_NOT_REAL_GT',reviewed_ids=[self.key])

    def test_edited_annotation_schema_and_source_pixels_still_fail_closed(self):
        folder,label=self.bundle();package=json.loads((folder/'package.json').read_text())
        original=copy.deepcopy(package);package['annotation_label_map']['红色顶面']['color_id']=4
        (folder/'package.json').write_text(json.dumps(package))
        with self.assertRaises(ValueError):import_bundle(self.root,self.manifest,folder)
        (folder/'package.json').write_text(json.dumps(original))
        (folder/'images'/(self.key+'.png')).write_bytes(b'changed')
        with self.assertRaises(ValueError):import_bundle(self.root,self.manifest,folder)

    def test_launcher_cannot_write_surface_labels_into_a_legacy_package(self):
        from tools import launch_material_labeling as launcher
        folder=self.root/'legacy';make_bundle(self.root,self.manifest,self.drafts,folder)
        executable=self.root/'.ops/local/venvs/kpu-label/Scripts/xanylabeling.exe'
        executable.parent.mkdir(parents=True);executable.write_bytes(b'NOT_EXECUTED_FIXTURE')
        with patch.object(launcher,'ROOT',self.root),patch('sys.argv',['launcher','--bundle',str(folder),
                 '--preset','colored_surfaces','--check-only']):
            with self.assertRaisesRegex(SystemExit,'differs'):launcher.main()

    def test_launcher_autodetects_twelve_labels_and_rejects_unknown_schema(self):
        from tools import launch_material_labeling as launcher
        folder,label=self.bundle()
        executable=self.root/'.ops/local/venvs/kpu-label/Scripts/xanylabeling.exe'
        executable.parent.mkdir(parents=True);executable.write_bytes(b'NOT_EXECUTED_FIXTURE')
        with patch.object(launcher,'ROOT',self.root),patch('sys.argv',['launcher','--bundle',str(folder),'--check-only']), \
             patch('builtins.print') as output:
            launcher.main()
        record=json.loads(output.call_args.args[0]);self.assertEqual(record['labels'],list(SURFACE_LABELS))
        config=Path(record['config']).read_text(encoding='utf-8')
        self.assertIn('shape_color: manual',config);self.assertIn('黑色顶面',config)
        package=json.loads((folder/'package.json').read_text());package['annotation_schema']='UNKNOWN_SCHEMA'
        (folder/'package.json').write_text(json.dumps(package))
        with patch.object(launcher,'ROOT',self.root),patch('sys.argv',['launcher','--bundle',str(folder),'--check-only']):
            with self.assertRaisesRegex(SystemExit,'Unknown'):launcher.main()


if __name__=='__main__':unittest.main()
