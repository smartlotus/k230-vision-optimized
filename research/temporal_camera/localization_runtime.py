"""Camera-only tracker API for the board and host replay."""
import math

from competition_paths import seed_start_belief
from embedded_filter import SoftGridFilter


class CameraXYTracker:
    def __init__(self, anchors, grid_size=48, temperature=1., memory_seconds=2.,
                 global_floor=.01, startup='competition', acceptance=None,
                 process_scale_mm_s=None):
        if not math.isfinite(temperature) or temperature<=0:
            raise ValueError('Temperature must be finite and positive')
        if startup not in ('competition','global'):
            raise ValueError('Invalid initialization mode')
        if acceptance is not None:
            if not 0 < acceptance['min_local_mass'] <= 1 or not 0 <= acceptance['max_far_mass'] < 1:
                raise ValueError('Invalid acceptance thresholds')
        self.grid_size = grid_size
        self.cell = 2400./grid_size
        self.temperature = temperature
        self.acceptance = acceptance
        self.cells = []
        for x,y in anchors:
            if not (-1200 <= x < 1200 and -1200 <= y < 1200):
                raise ValueError('Anchor outside field map')
            self.cells.append(int((y+1200)/self.cell)*grid_size+int((x+1200)/self.cell))
        if not self.cells:
            raise ValueError('Empty anchor list')
        self.likelihood = [0.]*(grid_size*grid_size)
        self.filter = SoftGridFilter([[True]*grid_size for _ in range(grid_size)],cell_mm=self.cell,
                                     memory_seconds=memory_seconds,global_floor=global_floor,
                                     process_scale_mm_s=process_scale_mm_s)
        if startup=='competition':
            seed_start_belief(self.filter)

    def update(self, logits, timestamp_s, visual_quality=1.):
        for i in range(len(self.likelihood)):
            self.likelihood[i]=0.
        if logits is not None:
            if len(logits)!=len(self.cells):
                raise ValueError('Unexpected number of location logits')
            if all(math.isfinite(value) for value in logits):
                peak=max(logits)
                for cell,value in zip(self.cells,logits):
                    self.likelihood[cell]+=math.exp((value-peak)/self.temperature)
        update=self.filter.update(self.likelihood,timestamp_s,visual_quality=visual_quality)
        x,y=update['camera_x_mm'],update['camera_y_mm']
        local_mass=0.
        far_mass=0.
        for i,value in enumerate(self.filter.belief):
            row,col=divmod(i,self.grid_size)
            dx=(col+.5)*self.cell-1200-x
            dy=(row+.5)*self.cell-1200-y
            d2=dx*dx+dy*dy
            if d2<=150**2:
                local_mass+=value
            if d2>=500**2:
                far_mass+=value
        accepted=(self.acceptance is not None and update['state']=='TRACKING'
                  and local_mass>=self.acceptance['min_local_mass']
                  and far_mass<=self.acceptance['max_far_mass'])
        update['candidate_x_mm']=x
        update['candidate_y_mm']=y
        update['camera_x_mm']=x if accepted else None
        update['camera_y_mm']=y if accepted else None
        update['confidence_local_mass']=local_mass
        update['confidence_far_mass']=far_mass
        update['position_valid']=accepted
        if update['state']=='TRACKING' and not accepted:
            update['state']='DEGRADED'
        return update
