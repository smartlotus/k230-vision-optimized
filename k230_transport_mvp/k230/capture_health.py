"""Camera fault history and bounded transition logging; no role changes."""
from vision.common import elapsed_ms


class CaptureHealth:
    def __init__(self,interval_ms=1000):
        self.interval_ms=interval_ms
        self.rows={}
        self.last_log={}

    def failure(self,camera_id,error,now):
        row=self.rows.setdefault(camera_id,{'camera_id':camera_id,'errors':0,'consecutive_errors':0,
            'first_error':None,'first_error_ms':None,'last_error':None,'last_error_ms':None,'status':'UNKNOWN'})
        row['errors']+=1;row['consecutive_errors']+=1
        if row['first_error'] is None:row['first_error']=str(error);row['first_error_ms']=now
        row.update(last_error=str(error),last_error_ms=now,status='UNAVAILABLE')
        first=row['consecutive_errors']==1
        if first or elapsed_ms(now,self.last_log.get(camera_id,now))>=self.interval_ms:
            self.last_log[camera_id]=now
            return dict(row,event='FIRST_FAILURE' if first else 'FAILURE_UPDATE')
        return None

    def success(self,camera_id,now):
        row=self.rows.get(camera_id)
        if row is None:return None
        recovered=row['consecutive_errors']>0
        row.update(consecutive_errors=0,status='LIVE',last_success_ms=now)
        return dict(row,event='RECOVERED') if recovered else None
