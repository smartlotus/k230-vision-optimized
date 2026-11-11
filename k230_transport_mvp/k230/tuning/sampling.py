"""One explicit same-frame JPEG/native-LAB sample with bounded grid memory."""
import os
from k230.capture_io import CaptureIO
from .schema import clone
from .store import ProfileStore


class SampleWriter:
    def __init__(self,root='/sdcard/transport_mvp/tuning_samples'):
        self.root=root;self.io=CaptureIO(root)

    def save(self,frame,metadata,rgb_to_lab,exitpoint=None):
        if type(metadata.get('camera_id')) is not int or metadata['camera_id'] not in (1,2):raise ValueError('Explicit healthy camera ID required')
        if type(metadata.get('is_simulated')) is not bool:raise ValueError('Explicit frame source required')
        if type(metadata.get('frame_id')) is not int or metadata['frame_id']<0:raise ValueError('Explicit frame ID required')
        width,height=frame.width(),frame.height();binding=metadata.get('image_profile',{})
        if (width,height)!=(binding.get('width'),binding.get('height')):raise ValueError('Sample frame binding differs')
        ProfileStore(self.root)._prepare()
        serial=max([int(n[7:13]) for n in os.listdir(self.root)
            if n.startswith('sample-') and len(n)>=13 and n[7:13].isdigit()] or [0])+1
        if serial>999999:raise OSError('Sample index exhausted')
        prefix=self.root+'/sample-%06d' % serial
        stride=max(4,(width+159)//160,(height+89)//90);origin=stride//2
        record=clone(metadata)
        record.update(source='SIMULATED_LAB_SAMPLE' if metadata['is_simulated'] else 'REAL_CAMERA_NATIVE_LAB',
            width=width,height=height,x0=origin,y0=origin,stride=stride,jpeg=prefix+'-live.jpg',
            camera_mount_fixed=None,distance_cm=None,motors_enabled=False)
        record['jpeg_saved']=self.io.save_image(frame,record['jpeg'])
        rows=[]
        for y in range(origin,height,stride):
            if exitpoint:exitpoint()
            rows.append([rgb_to_lab(*frame.get_pixel(x,y)) for x in range(origin,width,stride)])
        record['lab_grid']=rows
        path=prefix+'-lab-grid.json';saved=self.io.save_json(record,path)
        return {'serial':serial,'camera_id':metadata['camera_id'],'frame_id':metadata['frame_id'],
            'jpeg':record['jpeg'],'grid':path,'jpeg_saved':record['jpeg_saved'],'grid_saved':saved,
            'complete':saved and record['jpeg_saved'],'stride':stride,'points':sum(len(row) for row in rows)}
