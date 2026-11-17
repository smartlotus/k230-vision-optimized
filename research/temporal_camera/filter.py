"""CPU reference filter. Unknown camera speed => full-support soft prior."""
import math
import numpy as np


class TemporalGridFilter:
    def __init__(self, valid_mask, cell_mm=100., origin=(-1200., -1200.),
                 hard_gate_enabled=False, bad_limit=5, memory_seconds=.5,
                 global_floor=.05, top_k=5):
        self.valid = np.asarray(valid_mask, bool)
        if self.valid.ndim != 2 or not self.valid.any() or cell_mm <= 0:
            raise ValueError('Invalid map')
        if bad_limit < 1 or memory_seconds <= 0 or not 0 < global_floor <= 1 or top_k < 1:
            raise ValueError('Invalid filter parameters')
        self.cell, self.origin = float(cell_mm), origin
        self.hard_gate, self.bad_limit = hard_gate_enabled, int(bad_limit)
        self.memory, self.floor, self.top_k = memory_seconds, global_floor, top_k
        self.uniform = self.valid.astype(float)/self.valid.sum()
        self.belief = self.uniform.copy()
        self.last_ts, self.bad, self.state = None, 0, 'UNINITIALIZED'

    def norm(self, a):
        a = np.asarray(a, float)
        if a.shape != self.valid.shape:
            raise ValueError('Likelihood shape mismatch')
        if not np.isfinite(a).all() or (a < 0).any():
            return None
        a = np.where(self.valid, a, 0.)
        return a/a.sum() if a.sum() > 1e-15 else None

    def reset_global(self):
        self.belief = self.uniform.copy()
        self.last_ts, self.bad, self.state = None, 0, 'LOST'

    def predict(self, dt, speed, margin):
        h, w = self.valid.shape
        if self.hard_gate:
            if speed is None or margin is None or not math.isfinite(speed) or not math.isfinite(margin) or min(speed, margin) < 0:
                raise ValueError('Hard gating requires measured conservative camera bounds')
            radius = speed*dt+margin
            reach = min(max(h,w)-1, int(math.ceil(radius/self.cell))+1)
        else:
            reach, radius = 1, None
        out = np.zeros_like(self.belief)
        count = np.zeros_like(out)
        shifts = []
        for dy in range(-min(h-1,reach),min(h-1,reach)+1):
            for dx in range(-min(w-1,reach),min(w-1,reach)+1):
                if radius is not None and math.hypot(max(0,abs(dx)-1)*self.cell, max(0,abs(dy)-1)*self.cell) > radius:
                    continue
                sy = slice(max(0,-dy),min(h,h-dy)); sx = slice(max(0,-dx),min(w,w-dx))
                ty = slice(max(0,dy),min(h,h+dy)); tx = slice(max(0,dx),min(w,w+dx))
                count[sy,sx] += self.valid[ty,tx]
                shifts.append((sy,sx,ty,tx))
        for sy,sx,ty,tx in shifts:
            out[ty,tx] += np.divide(self.belief[sy,sx],count[sy,sx],
                                    out=np.zeros_like(count[sy,sx]),where=count[sy,sx]>0)*self.valid[ty,tx]
        p = self.norm(out)
        if p is None:
            p = self.uniform.copy()
        if not self.hard_gate:
            retain = (1-self.floor)*math.exp(-dt/self.memory)
            p = retain*p+(1-retain)*self.uniform
        return p

    def update(self, likelihood, timestamp_s, v_camera_max_mm_s=None,
               process_margin_mm=None, visual_quality=1.):
        ts = float(timestamp_s)
        quality = float(visual_quality)
        if not math.isfinite(ts) or not math.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError('Invalid timestamp/quality')
        if self.last_ts is not None and ts < self.last_ts:
            raise ValueError('Timestamp must be monotonic')
        vis = self.norm(likelihood)
        dt = 0 if self.last_ts is None else ts-self.last_ts
        prior = self.predict(dt,v_camera_max_mm_s,process_margin_mm) if self.last_ts is not None else self.uniform.copy()
        relocalized = False
        if vis is None or quality < .15:
            self.bad += 1
            self.state = 'LOST' if self.bad >= self.bad_limit else 'DEGRADED'
            self.belief = self.uniform.copy() if self.state == 'LOST' else prior
        elif self.state in ('UNINITIALIZED','LOST'):
            relocalized = self.state == 'LOST'
            self.belief, self.bad = vis, 0
            self.state = 'TRACKING' if quality >= .5 else 'DEGRADED'
        elif self.hard_gate and float(vis[prior>0].sum()) < .05:
            self.bad += 1
            self.state = 'LOST' if self.bad >= self.bad_limit else 'DEGRADED'
            self.belief = self.uniform.copy() if self.state == 'LOST' else prior
        else:
            posterior = self.norm(prior*np.maximum(vis,1e-15))
            self.belief = prior if posterior is None else posterior
            self.bad = 0
            self.state = 'TRACKING' if quality >= .5 else 'DEGRADED'
        self.last_ts = ts
        ids = np.argsort(np.where(self.valid,self.belief,-np.inf).ravel())[-min(self.top_k,int(self.valid.sum())):][::-1]
        top = []
        for idx in ids:
            y,x = divmod(int(idx),self.valid.shape[1])
            top.append({'camera_x_mm': self.origin[0]+(x+.5)*self.cell,
                        'camera_y_mm': self.origin[1]+(y+.5)*self.cell,
                        'probability': float(self.belief[y,x])})
        nz = self.belief[self.belief>0]
        return {**top[0], 'confidence': top[0]['probability'], 'top_k': top,
                'entropy': float(-(nz*np.log(nz)).sum()), 'state': self.state,
                'relocalized': relocalized, 'hard_gate_enabled': self.hard_gate}
