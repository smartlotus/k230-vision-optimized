"""Bounded interval telemetry for the continuous camera workbench."""
from vision.common import elapsed_ms


class PreviewMetrics:
    def __init__(self,interval_ms=1000,max_windows=120):
        self.interval_ms=interval_ms
        self.max_windows=max_windows
        self.windows=[]
        self.discarded_partial_windows=0
        self.discarded_partial_frames=0
        self.context=None
        self.baseline=None

    def reset(self,context,now,values):
        if self.baseline is not None and values.get('frames',0)>self.baseline[1].get('frames',0):
            self.discarded_partial_windows+=1
            self.discarded_partial_frames+=values['frames']-self.baseline[1].get('frames',0)
        self.context=context
        self.baseline=(now,dict(values))

    def update(self,context,now,values):
        if self.baseline is None or context!=self.context:
            self.reset(context,now,values)
            return None
        elapsed=elapsed_ms(now,self.baseline[0])
        if elapsed<self.interval_ms:return None
        delta={k:v-self.baseline[1].get(k,0) for k,v in values.items()}
        if elapsed<=0 or any(v<0 for v in delta.values()):
            self.reset(context,now,values)
            return None
        row={'context':context,'start_ms':self.baseline[0],'end_ms':now,'elapsed_ms':elapsed,
            'complete_interval':True,
            'delta':delta,'processing_fps':delta['frames']*1000/elapsed,
            'display_submit_fps':delta.get('display_frames',0)*1000/elapsed,
            'contains_synchronous_capture':delta.get('capture_jobs',0)>0}
        self.windows.append(row)
        if len(self.windows)>self.max_windows:del self.windows[0]
        self.baseline=(now,dict(values))
        return row

    def finish(self,now,values):
        if self.baseline is None or values['frames']<=self.baseline[1].get('frames',0):return None
        minimum=self.interval_ms
        self.interval_ms=1
        try:row=self.update(self.context,now,values)
        finally:self.interval_ms=minimum
        if row is not None:row['complete_interval']=row['elapsed_ms']>=minimum
        return row
