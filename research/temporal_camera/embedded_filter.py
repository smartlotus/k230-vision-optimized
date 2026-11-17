"""MicroPython-compatible CPU soft filter; no NumPy/UART/motion dependencies.

This is the unknown-motion policy. Measured hard support remains opt-in in
filter.py and is not silently enabled on an uncalibrated device.
"""
import math


class SoftGridFilter:
    def __init__(self, valid_mask, cell_mm=100., origin=(-1200.,-1200.),
                 bad_limit=5, memory_seconds=.5, global_floor=.05, top_k=5,
                 process_scale_mm_s=None):
        self.h=len(valid_mask)
        self.w=len(valid_mask[0]) if self.h else 0
        if not self.w or any(len(r)!=self.w for r in valid_mask):
            raise ValueError('Invalid rectangular mask')
        self.valid=[bool(v) for row in valid_mask for v in row]
        self.n=sum(self.valid)
        if not self.n or cell_mm<=0 or bad_limit<1 or memory_seconds<=0 or not 0<global_floor<=1 or top_k<1:
            raise ValueError('Invalid soft-filter parameters')
        if process_scale_mm_s is not None and (not math.isfinite(process_scale_mm_s) or process_scale_mm_s <= 0):
            raise ValueError('Invalid soft process scale')
        self.cell,self.origin=cell_mm,origin
        self.limit,self.memory,self.floor,self.k=bad_limit,memory_seconds,global_floor,top_k
        self.process_scale=process_scale_mm_s
        self.uniform=[1./self.n if v else 0. for v in self.valid]
        self.belief=self.uniform[:]
        self.prior=[0.]*len(self.valid)
        self.visual=[0.]*len(self.valid)
        if self.process_scale is None:
            self.neighbors=[]
            for i in range(len(self.valid)):
                y,x=divmod(i,self.w)
                self.neighbors.append([yy*self.w+xx for yy in range(max(0,y-1),min(self.h,y+2))
                                       for xx in range(max(0,x-1),min(self.w,x+2)) if self.valid[yy*self.w+xx]])
        else:
            self._radius=None
            self._counts=[0.] * len(self.valid)
            self._weighted=[0.] * len(self.valid)
            self._scratch=[0.] * len(self.valid)
        self.last_ts=None;self.bad=0;self.state='UNINITIALIZED'
        self._pending_prior=False

    def initialize(self, probabilities):
        if len(probabilities)!=len(self.valid):
            raise ValueError('Initial prior shape mismatch')
        if any(not math.isfinite(p) or p<0 for p in probabilities):
            raise ValueError('Invalid initial prior')
        mass=sum(p for p,valid in zip(probabilities,self.valid) if valid)
        if mass<=0:
            raise ValueError('Empty initial prior')
        for i,valid in enumerate(self.valid):
            self.belief[i]=probabilities[i]/mass if valid else 0.
        self.last_ts=None
        self.bad=0
        self.state='DEGRADED'
        self._pending_prior=True

    def reset_global(self):
        self.belief[:]=self.uniform
        self.last_ts=None;self.bad=0;self.state='LOST'
        self._pending_prior=False

    def _box_sum(self, values, output, radius):
        """Separable rolling sums over a clipped square, in O(grid cells)."""
        w,h=self.w,self.h
        scratch=self._scratch
        for row in range(h):
            start=row*w
            total=sum(values[i] for i in range(start,start+min(w,radius+1)))
            for col in range(w):
                scratch[start+col]=total
                if col>=radius:
                    total-=values[start+col-radius]
                if col+radius+1<w:
                    total+=values[start+col+radius+1]
        for col in range(w):
            total=sum(scratch[row*w+col] for row in range(min(h,radius+1)))
            for row in range(h):
                output[row*w+col]=total
                if row>=radius:
                    total-=scratch[(row-radius)*w+col]
                if row+radius+1<h:
                    total+=scratch[(row+radius+1)*w+col]

    def _predict(self, dt):
        if self.process_scale is None:
            for i in range(len(self.prior)):
                self.prior[i]=0.
            for i,targets in enumerate(self.neighbors):
                if self.belief[i] and targets:
                    value=self.belief[i]/len(targets)
                    for target in targets:
                        self.prior[target]+=value
        else:
            radius=max(1,math.ceil(min(max(self.w,self.h)-1,self.process_scale*dt/self.cell)))
            if radius!=self._radius:
                self._box_sum(self.valid,self._counts,radius)
                self._radius=radius
            for i,valid in enumerate(self.valid):
                self._weighted[i]=self.belief[i]/self._counts[i] if valid else 0.
            self._box_sum(self._weighted,self.prior,radius)
        retain=(1-self.floor)*math.exp(-dt/self.memory)
        for i,valid in enumerate(self.valid):
            self.prior[i]=retain*max(0.,self.prior[i])+(1-retain)*self.uniform[i] if valid else 0.

    def update(self, likelihood, timestamp_s, visual_quality=1.):
        ts=float(timestamp_s);quality=float(visual_quality)
        if not math.isfinite(ts) or not math.isfinite(quality) or not 0<=quality<=1:
            raise ValueError('Invalid timestamp/quality')
        if self.last_ts is not None and ts<self.last_ts:raise ValueError('Timestamp must be monotonic')
        if len(likelihood)==self.h and isinstance(likelihood[0],(list,tuple)):
            flat=[v for row in likelihood for v in row]
        else:flat=likelihood
        if len(flat)!=len(self.valid):raise ValueError('Likelihood shape mismatch')
        mass=0.;usable=True
        for i in range(len(self.visual)):
            v=float(flat[i])
            if not math.isfinite(v) or v<0:usable=False
            self.visual[i]=v if self.valid[i] and math.isfinite(v) and v>=0 else 0.
            mass+=self.visual[i]
        usable=usable and mass>1e-15
        if usable:
            for i in range(len(self.visual)):self.visual[i]/=mass
        if self.last_ts is None:
            if self._pending_prior:
                for i in range(len(self.prior)):
                    self.prior[i]=(1-self.floor)*self.belief[i]+self.floor*self.uniform[i]
            else:
                self.prior[:]=self.uniform
        else:
            self._predict(ts-self.last_ts)
        relocalized=False
        if not usable or quality<.15:
            self.bad+=1;self.state='LOST' if self.bad>=self.limit else 'DEGRADED'
            self.belief[:]=self.uniform if self.state=='LOST' else self.prior
        elif self.state in ('UNINITIALIZED','LOST'):
            relocalized=self.state=='LOST';self.belief[:]=self.visual;self.bad=0
            self.state='TRACKING' if quality>=.5 else 'DEGRADED'
        else:
            total=0.
            for i in range(len(self.belief)):
                self.belief[i]=self.prior[i]*max(self.visual[i],1e-15) if self.valid[i] else 0.
                total+=self.belief[i]
            for i in range(len(self.belief)):self.belief[i]/=total
            self.bad=0;self.state='TRACKING' if quality>=.5 else 'DEGRADED'
        self.last_ts=ts
        self._pending_prior=False
        ids=sorted((i for i in range(len(self.belief)) if self.valid[i]),key=lambda i:(self.belief[i],i),reverse=True)[:min(self.k,self.n)]
        top=[]
        for i in ids:
            y,x=divmod(i,self.w)
            top.append({'camera_x_mm':self.origin[0]+(x+.5)*self.cell,
                        'camera_y_mm':self.origin[1]+(y+.5)*self.cell,'probability':self.belief[i]})
        entropy=-sum(p*math.log(p) for p in self.belief if p>0)
        return dict(top[0],confidence=top[0]['probability'],top_k=top,entropy=entropy,
                    state=self.state,relocalized=relocalized,hard_gate_enabled=False)
