"""Host-only image-operation and geometry benchmark; not board FPS."""
import argparse
import json
from pathlib import Path
import statistics
import time
import math
import tracemalloc
import cv2
import numpy as np
from k230.tuning.schema import defaults
from k230.tuning import live, live_optimized
from vision import ellipse_native, ellipse_native_optimized


class Blob:
    def rect(self):return (100, 80, 81, 61)
    def code(self):return 1


class HostImage:
    """Count explicit image copies with an OpenCV binary image stand-in."""
    def __init__(self, data, counters=None, present=True):
        self.data=data
        self.counters=counters if counters is not None else {'copies':0,'copied_pixels':0,'draw_calls':0}
        self.present=present
    def copy(self, roi=None):
        data=self.data
        if roi:
            x,y,w,h=roi;data=data[y:y+h,x:x+w]
        self.counters['copies']+=1;self.counters['copied_pixels']+=data.size
        return HostImage(data.copy(),self.counters,self.present)
    def clear(self):self.data[:]=0
    def binary(self,*args):return self
    def to_grayscale(self):return self
    def to_rgb565(self):return self
    def to_numpy_ref(self):return self.data
    def b_and(self,other):self.data &= other.data
    def draw_image(self,other,x,y):
        self.counters['draw_calls']+=1
        h,w=other.data.shape;self.data[y:y+h,x:x+w]=other.data
    def draw_line(self,*args,**kwargs):self.counters['draw_calls']+=1
    def draw_cross(self,*args,**kwargs):self.counters['draw_calls']+=1
    def find_blobs(self,*args,**kwargs):return [Blob()] if self.present else []


def fixture(present=True):
    data=np.zeros((360,640),np.uint8)
    if present:cv2.ellipse(data,(140,110),(40,30),0,0,360,255,-1)
    profile=defaults()
    profile['contexts']['TOP_BUFFER']['thresholds']['1']=[0,100,-128,127,-128,127]
    profile['contexts']['TOP_BUFFER']['roi']=[0,0,640,360]
    return HostImage(data,present=present),profile


def benchmark():
    rows=[]
    for name,module,render in [('legacy',live,True),('optimized_render',live_optimized,True),
                                ('optimized_recognition_only',live_optimized,False)]:
        frame,profile=fixture()
        samples=[]
        for _ in range(5):
            start=time.perf_counter()
            for i in range(50):
                extra={} if module is live else {'render':render}
                result=module.native_views(frame,profile,'TOP','BUFFER',1,selected=True,**extra)
            samples.append(time.perf_counter()-start)
        assert len(result[2]['accepted'])==1
        rows.append(dict(variant=name,iterations=250,median_50_calls_seconds=statistics.median(samples),
                         samples=samples,operations=frame.counters,geometry=result[2]))
    assert rows[0]['geometry']==rows[1]['geometry']==rows[2]['geometry']
    contour_rows=[]
    points=[[90+35*math.cos(i*2*math.pi/4096),70+24*math.sin(i*2*math.pi/4096)] for i in range(4096)]
    fits=[]
    for module in (ellipse_native,ellipse_native_optimized):
        tracemalloc.start()
        fit=module.contour_fit(points,roi=[0,0,220,180],pixel_area=math.pi*35*24,
                               native_fit=((90,70),(70,48),0))
        peak=tracemalloc.get_traced_memory()[1];tracemalloc.stop()
        fits.append(fit)
        contour_rows.append(dict(variant=module.__name__,points=4096,peak_python_bytes=peak))
    assert fits[0]==fits[1]
    return dict(scope='Synthetic host grayscale image stand-in, excludes sensor/native LAB/display/JPEG; not K230 FPS',
                measurements=rows,geometry_equal=True,contour_allocations=contour_rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=benchmark()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({row['variant']:{'time':row['median_50_calls_seconds'],'operations':row['operations']} for row in report['measurements']}))
    print(json.dumps(report['contour_allocations']))
