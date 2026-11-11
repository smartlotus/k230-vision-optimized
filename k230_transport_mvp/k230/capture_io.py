"""Optional capture persistence cannot terminate live camera processing."""
import json


class CaptureIO:
    def __init__(self,root='/sdcard/transport_mvp',opener=None):
        self.root=root.rstrip('/');self.opener=opener or open
        self.last_seq=-1;self.errors=[];self.error_count=0;self.last_capture=None

    def _error(self,operation,error):
        self.error_count+=1
        self.errors=(self.errors+[{'operation':operation,'error':str(error)}])[-10:]
        return False

    def save_image(self,image,path):
        try:image.save(path,quality=95);return True
        except OSError as error:return self._error(path,error)

    def save_json(self,value,path):
        try:
            with self.opener(path,'w') as stream:stream.write(json.dumps(value))
            return True
        except OSError as error:return self._error(path,error)

    def poll(self,path,frame,metadata,rgb_to_lab,exitpoint=None):
        try:
            with self.opener(path) as stream:raw=stream.read(1025)
        except OSError:return None
        try:
            if len(raw)>1024:raise ValueError('Capture request too long')
            command=json.loads(raw)
            if not isinstance(command,dict) or set(command)!={'seq','tag','camera_id'}:
                raise ValueError('Capture requires seq, tag and camera_id only')
            seq=command['seq'];tag=command['tag'];camera=command['camera_id']
            if type(seq) is not int or not 0<=seq<=65535:raise ValueError('Invalid capture sequence')
            if seq<=self.last_seq:return None
            if not isinstance(tag,str) or not 1<=len(tag)<=40 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in tag):
                raise ValueError('Invalid capture tag')
            if type(camera) is not int or camera not in (1,2):raise ValueError('Invalid camera ID')
            if camera!=metadata['camera_id']:return None
        except (ValueError,TypeError,KeyError) as error:
            # Bad contents never become a path or change detection state.
            if str(error)!=getattr(self,'last_request_error',None):self._error('capture_request',error)
            self.last_request_error=str(error)
            return None
        # Consume before any side effect. A failed/partial write requires a new explicit sequence.
        self.last_seq=seq
        prefix=self.root+'/'+tag+'-s'+str(seq)
        record=dict(metadata,source='REAL_CAMERA_NATIVE_LAB',is_simulated=False,
            width=frame.width(),height=frame.height(),distance_cm=None,camera_mount_fixed=False,
            request_seq=seq,x0=2,y0=2,stride=4,jpeg=prefix+'-live.jpg',motors_enabled=False)
        saved=self.save_image(frame,record['jpeg'])
        grid=[]
        for y in range(2,frame.height(),4):
            if exitpoint:exitpoint()
            grid.append([rgb_to_lab(*frame.get_pixel(x,y)) for x in range(2,frame.width(),4)])
        record['lab_grid']=grid
        record['jpeg_saved']=saved
        grid_path=prefix+'-lab-grid.json'
        grid_saved=self.save_json(record,grid_path)
        self.last_capture={'seq':seq,'camera_id':camera,'frame_id':metadata['frame_id'],
            'jpeg':record['jpeg'],'grid':grid_path,'jpeg_saved':saved,'grid_saved':grid_saved,
            'complete':saved and grid_saved}
        return self.last_capture

    def status(self):
        return {'error_count':self.error_count,'recent_errors':self.errors,'last_capture':self.last_capture}
