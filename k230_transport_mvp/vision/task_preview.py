"""Real QR -> immutable task plan -> observed target; never infers motion success."""
from core.task_parser import parse_task
from core.task_planner import build_execution_steps
from .qr import QRTracker


class TaskPreview:
    def __init__(self,qr_camera_id=1):
        if type(qr_camera_id) is not int or qr_camera_id not in (1,2):raise ValueError('Healthy QR camera required')
        self.qr_camera_id=qr_camera_id
        self.accepted_camera=None
        self.qr=QRTracker(confirmations=3,timeout_ms=5000,max_gap_ms=1000)
        self.task=None
        self.steps=[]
        self.last_scan={'status':'NOT_FOUND','reason':'QR_WAITING','payload':None,'submitted':False}
        self.accepted_frame=None
        self.restored=False

    def restore_confirmed(self,record):
        """Explicit host-authorized continuation of the same captured QR task."""
        camera=record.get('accepted_camera_id')
        if (self.task is not None or type(camera) is not int or camera not in (1,2)
                or record.get('source')!='REAL_CSI%d_QR' % camera):
            raise ValueError('Confirmed real camera task receipt required')
        if type(record.get('accepted_frame_id')) is not int or record['accepted_frame_id']<1:
            raise ValueError('Confirmed frame identity required')
        self.task=parse_task(record['payload'])
        self.steps=build_execution_steps(self.task)
        self.accepted_frame=record['accepted_frame_id']
        self.accepted_camera=camera
        self.restored=True

    def scan(self,payloads,frame_id,timestamp_ms,camera_id=None,is_simulated=False):
        camera_id=self.qr_camera_id if camera_id is None else camera_id
        if camera_id!=self.qr_camera_id or is_simulated is not False:
            raise ValueError('Real configured QR camera required')
        if self.task is not None:
            return {'status':'NOT_FOUND','reason':'TASK_LOCKED','payload':self.task.payload,'submitted':False}
        update=self.qr.update(payloads,frame_id,timestamp_ms,('CSI%d_QR' % self.qr_camera_id,))
        self.last_scan=update
        if update['reason']=='QR_TIMEOUT':
            # New bounded scan window; submitted-task deduplication is preserved.
            self.qr.reset_scan()
        if update['submitted']:
            self.task=parse_task(update['payload'])
            self.steps=build_execution_steps(self.task)
            self.accepted_frame=frame_id
            self.accepted_camera=camera_id
        return update

    def selected_target(self,candidates,reference=(160,120)):
        if self.task is None:return None
        target=self.task.batch1_colors[0]
        eligible=[c for c in candidates if c['target_id']==target and c['stable'] and c['eligible']]
        if not eligible:return None
        return min(eligible,key=lambda c:sum((c['center'][k]-reference[k])**2 for k in (0,1)))

    def summary(self):
        return {'payload':self.task.payload if self.task else None,
            'batch1_colors':list(self.task.batch1_colors) if self.task else [],
            'batch2_colors':list(self.task.batch2_colors) if self.task else [],
            'batch2_buffer_slots':[self.task.color_to_buffer_slot[c] for c in self.task.batch2_colors] if self.task else [],
            'planned_steps':len(self.steps),'current_preview_target':self.task.batch1_colors[0] if self.task else None,
            'accepted_camera_id':self.accepted_camera,'scan_camera_id':self.qr_camera_id,'accepted_frame_id':self.accepted_frame,
            'confirmed_physical_steps':0,'motion_state':'WAIT_CALIBRATION_AND_CONTROLLER',
            'restored_confirmed_task':self.restored,
            'source':'REAL_CSI%d_QR' % self.accepted_camera if self.task else None}
