import json,tempfile,unittest
from pathlib import Path
from k230.tuning.sampling import SampleWriter


class SampleTests(unittest.TestCase):
    def test_full_hd_grid_is_bounded_and_records_explicit_simulated_source(self):
        class Image:
            def width(self):return 1920
            def height(self):return 1080
            def save(self,path,**kwargs):Path(path).write_bytes(b'fake image')
            def get_pixel(self,x,y):return (x,y,0)
        with tempfile.TemporaryDirectory() as root:
            meta={'camera_id':1,'frame_id':5,'is_simulated':True,'image_profile':{'width':1920,'height':1080}}
            writer=SampleWriter(root);result=writer.save(Image(),meta,lambda *v:v)
            self.assertEqual(result['points'],14400);self.assertEqual(result['stride'],12);self.assertTrue(result['complete'])
            record=json.loads(Path(result['grid']).read_text())
            self.assertEqual(record['source'],'SIMULATED_LAB_SAMPLE');self.assertEqual(record['lab_grid'][0][0],[6,6,0])
            second=writer.save(Image(),meta,lambda *v:v)
            self.assertEqual(second['serial'],2);self.assertTrue(Path(result['grid']).exists())

    def test_source_or_geometry_mismatch_is_rejected_before_saving(self):
        class Image:
            def width(self):return 640
            def height(self):return 360
        with tempfile.TemporaryDirectory() as root:
            writer=SampleWriter(root)
            with self.assertRaises(ValueError):writer.save(Image(),{'camera_id':1},lambda *v:v)
            with self.assertRaises(ValueError):writer.save(Image(),{'camera_id':1,'is_simulated':False,'image_profile':{'width':640,'height':360}},lambda *v:v)
            self.assertEqual(list(Path(root).iterdir()),[])
            with self.assertRaises(ValueError):writer.save(Image(),{'camera_id':1,'is_simulated':False,'image_profile':{'width':320,'height':240}},lambda *v:v)
