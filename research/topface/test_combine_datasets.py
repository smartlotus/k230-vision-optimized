import json
from pathlib import Path
import tempfile
import unittest
from cloud_store import sha
from combine_datasets import combine


def source(root,name,group,seed,split='train',camera='cameraA'):
    root=root/name;scene=root/'scene-00000';scene.mkdir(parents=True)
    label={'scene_group_id':group,'random_seed':seed,'camera_group_id':camera,'split':split}
    (scene/'labels.json').write_text(json.dumps(label),encoding='utf8')
    (scene/'rgb.png').write_bytes(('fixture-'+name).encode())
    audit={'status':'PASS','images':1,'scenes':[{'scene':scene.name,'instances':1,'group':group,'seed':seed,
        'split':split,'camera_group':camera,'label_sha256':sha(scene/'labels.json'),'rgb_sha256':sha(scene/'rgb.png')}]}
    (root/'AUDIT_V11.json').write_text(json.dumps(audit),encoding='utf8');return root


class CompositeTests(unittest.TestCase):
    def test_original_labels_kept_with_unambiguous_parent_prefixes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=source(root,'a','g1',1);b=source(root,'b','g2',2)
            result=combine([('primary',a,'archiveA'),('colorfix',b,'archiveB')],root/'combined')
            self.assertEqual(result['images'],2);self.assertFalse(result['original_labels_modified'])
            for row in result['scenes']:
                self.assertEqual(sha(root/'combined'/row['scene']/'labels.json'),row['label_sha256'])

    def test_camera_split_leakage_and_repeated_seed_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=source(root,'a','g1',1);b=source(root,'b','g2',2,split='val')
            with self.assertRaisesRegex(ValueError,'Camera group crosses'):
                combine([('primary',a,'a'),('colorfix',b,'b')],root/'leak')
            c=source(root,'c','g3',1,camera='cameraB')
            with self.assertRaisesRegex(ValueError,'Duplicated scene/seed/pixels'):
                combine([('primary',a,'a'),('colorfix',c,'c')],root/'repeat')


if __name__=='__main__':unittest.main()
