"""Draft/apply/save are distinct; UI never supplies H or motion authority."""
from .schema import defaults,validate,clone,compile_config,CONTEXTS,context_key,SIZES
from vision.common import elapsed_ms


class TuningSession:
    def __init__(self,store,slot=None,demo=False):
        slot=store.selected_slot() if slot is None else slot
        self.store=store;self.slot=slot;self.demo=demo;self.running=False
        self.mode='TOP';self.station='BUFFER';self.target_id=5
        self.saved=store.load(slot)
        self.draft=clone(self.saved['profile']) if self.saved else defaults()
        self.active=clone(self.draft);self.dirty=False;self.adapter=None
        self.observation=None;self.last_error=None;self.last_commit=self.saved
        self.apply_serial=0;self.countdown_started=None
        self.apply()

    def editable(self):
        if self.running:raise ValueError('Controller running; editing forbidden')
        if self.draft['locked']:raise ValueError('Parameters locked')

    def change(self,edit):
        self.editable();candidate=clone(self.draft);edit(candidate)
        candidate['revision']+=1;validate(candidate)
        self.draft=candidate;self.dirty=True;self.last_error=None

    def select(self,mode,station):
        if (mode,station) not in CONTEXTS:raise ValueError('Unknown context')
        if self.running:raise ValueError('Controller owns mode during run')
        self.mode,self.station=mode,station
        self.target_id=1 if mode=='RING' else (self.target_id if mode in ('COLOR','TOP') and self.target_id in range(1,7) else 1)
        if mode in ('QR','LANE'):self.target_id=None
        self.observation=None
        if self.adapter:self.adapter.reset_session()

    def context(self,active=False):
        return (self.active if active else self.draft)['contexts'][context_key(self.mode,self.station)]

    def set_roi(self,roi,numbered=False):
        key=context_key(self.mode,self.station)
        def edit(profile):
            c=profile['contexts'][key]
            if numbered:
                if self.mode!='RING':raise ValueError('Numbered ROI is RING only')
                c['id_rois'][str(self.target_id)]=list(roi);c['ring_id_reviewed']=False
            else:c['roi']=list(roi)
            c['sampling_status']='MANUAL_UNVERIFIED'
        self.change(edit)

    def enable_threshold(self):
        if self.mode not in ('COLOR','TOP'):raise ValueError('LAB only for color modes')
        key=context_key(self.mode,self.station);cid=str(self.target_id)
        def edit(profile):
            c=profile['contexts'][key]
            c['thresholds'][cid]=[0,100,-128,127,-128,127]
            c['sampling_status']='MANUAL_UNVERIFIED'
        self.change(edit)

    def set_resolution(self,width,height):
        if (width,height) not in SIZES:raise ValueError('Unsupported size')
        def edit(profile):
            profile['binding'].update(width=width,height=height,pose_id=None)
            profile['display'].update(pan_x=0,pan_y=0)
            for c in profile['contexts'].values():
                c.update(roi=[0,0,width,height],reference=[width//2,height//2],
                    max_area=min(c['max_area'],width*height),lane_max_width=min(c['lane_max_width'],width),
                    id_rois={'1':None,'2':None,'3':None},ring_id_reviewed=False,sampling_status='MANUAL_UNVERIFIED')
                c['min_area']=min(c['min_area'],c['max_area'])
                c['lane_min_width']=min(c['lane_min_width'],c['lane_max_width'])
        self.change(edit)

    def apply(self):
        if self.running:raise ValueError('Controller running; apply forbidden')
        cfg=compile_config(self.draft)
        from k230.vision_adapter import VisionAdapter
        adapter=VisionAdapter(cfg)
        self.active=clone(self.draft);self.view=clone(self.draft['display'])
        self.adapter=adapter;self.observation=None;self.apply_serial+=1
        return cfg

    def set_display(self,**changes):
        """Display preferences never apply pending detector/camera drafts."""
        self.editable()
        candidate=clone(self.draft)
        candidate['display'].update(changes);validate(candidate)
        self.draft=candidate;self.view=clone(candidate['display'])
        self.active['display']=clone(self.view);self.dirty=True

    def save(self):
        if self.running:raise ValueError('Controller running; save forbidden')
        # Save has no implicit camera/algorithm changes; Apply remains explicit.
        result=self.store.save(self.draft,self.slot)
        self.store.select_slot(self.slot)
        self.saved=self.last_commit=result;self.dirty=False
        return result

    def restore(self,previous=False):
        self.editable();record=self.store.load(self.slot,previous)
        if record is None:raise ValueError('No saved version')
        self.draft=clone(record['profile']);self.draft['locked']=False
        self.dirty=previous;self.saved=record;self.apply()

    def set_locked(self,locked):
        if self.running:raise ValueError('Cannot unlock during controller run')
        if type(locked) is not bool:raise ValueError('Boolean lock required')
        self.draft['locked']=locked;self.dirty=True
        if locked:self.apply()

    def process(self,packet):
        station=None if self.mode in ('QR','LANE') else self.station
        self.observation=self.adapter.detect(self.mode,packet,self.target_id,station)
        return self.observation
