import io,json,unittest
from k230.capture_io import CaptureIO


class Frame:
    def width(self):return 16
    def height(self):return 16
    def get_pixel(self,x,y):return (x,y,7)
    def save(self,path,**kwargs):pass


class CaptureTests(unittest.TestCase):
    def store(self,request,fail=False):
        writes={}
        class Writer(io.StringIO):
            def __init__(self,path):super().__init__();self.path=path
            def close(self):writes[self.path]=self.getvalue();super().close()
        def opener(path,mode='r'):
            if mode=='r':return io.StringIO(json.dumps(request))
            if fail:raise OSError('I/O error')
            return Writer(path)
        return CaptureIO('/test',opener),writes

    def test_failed_image_or_report_write_does_not_escape(self):
        capture,_=self.store({},True)
        frame=Frame();frame.save=lambda *a,**k:(_ for _ in ()).throw(OSError('I/O error'))
        self.assertFalse(capture.save_image(frame,'/missing'))
        self.assertFalse(capture.save_json({},'/missing'))
        self.assertEqual(capture.error_count,2)
        for _ in range(20):capture.save_json({},'/missing')
        self.assertEqual(len(capture.errors),10)

    def test_capture_pairs_same_frame_native_grid_with_jpeg_and_deduplicates(self):
        capture,writes=self.store({'seq':1,'tag':'top01','camera_id':1})
        result=capture.poll('/request',Frame(),{'camera_id':1,'frame_id':4,'timestamp_ms':80},lambda *v:v)
        self.assertTrue(result['complete'])
        grid=json.loads(writes[result['grid']])
        self.assertEqual(grid['frame_id'],4)
        self.assertEqual(grid['lab_grid'][0][0],[2,2,7])
        self.assertFalse(grid['is_simulated'])
        self.assertIsNone(capture.poll('/request',Frame(),{'camera_id':1,'frame_id':5},lambda *v:v))

    def test_partial_failure_consumed_not_retried_and_other_camera_not_consumed(self):
        capture,_=self.store({'seq':1,'tag':'a','camera_id':1},True)
        self.assertIsNone(capture.poll('/request',Frame(),{'camera_id':2,'frame_id':1},lambda *v:v))
        self.assertEqual(capture.last_seq,-1)
        r=capture.poll('/request',Frame(),{'camera_id':1,'frame_id':2},lambda *v:v)
        self.assertFalse(r['complete']);self.assertTrue(r['jpeg_saved']);self.assertFalse(r['grid_saved'])
        self.assertIsNone(capture.poll('/request',Frame(),{'camera_id':1,'frame_id':3},lambda *v:v))

    def test_untrusted_request_cannot_choose_paths_or_enable_motion(self):
        for request in ({'seq':1,'tag':'../main','camera_id':1},
                        {'seq':True,'tag':'a','camera_id':1},
                        {'seq':1,'tag':'a','camera_id':1,'motion':True}):
            capture,writes=self.store(request)
            self.assertIsNone(capture.poll('/r',Frame(),{'camera_id':1},lambda *v:v))
            self.assertFalse(writes);self.assertEqual(capture.last_seq,-1)

    def test_keyboard_interrupt_is_not_swallowed(self):
        capture,_=self.store({})
        frame=Frame();frame.save=lambda *a,**k:(_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):capture.save_image(frame,'/x')


if __name__=='__main__':unittest.main()
