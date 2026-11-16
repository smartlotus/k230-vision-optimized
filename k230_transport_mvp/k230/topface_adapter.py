"""Persistent TopFace KPU backend, with explicit model identity and ownership."""
import json
import gc
import time
import math
from vision.topface_decode import decode_tensor

MODEL_SHA='f9bb4021ec0ca812e2dca333953ba72ca775b1c8a18fccf0f9ac0ec87f241f88'


def digest(path):
    import hashlib,binascii
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        while True:
            block=stream.read(4096)
            if not block:break
            h.update(block)
    return binascii.hexlify(h.digest()).decode()


class TopFaceBackend:
    def __init__(self,root):
        import nncase_runtime as nn
        from ulab import numpy as np
        with open(root+'/identity.json') as stream:self.identity=json.load(stream)
        with open(root+'/output_contract.json') as stream:contract=json.load(stream)
        if (self.identity.get('kmodel_sha256')!=MODEL_SHA or
                self.identity.get('model_wh')!=[320,192] or
                self.identity.get('output_shape')!=[1,1275,50] or
                self.identity.get('input')!='NCHW_RGB_UINT8' or
                self.identity.get('score_threshold')!=.25 or self.identity.get('nms_iou')!=.5 or
                contract.get('compiled_input',{}).get('shape')!=[1,3,192,320] or
                contract.get('source_checkpoint_sha256')!=self.identity.get('checkpoint_sha256')):
            raise ValueError('Frozen TopFace identity/contract mismatch')
        if digest(root+'/topface.kmodel')!=MODEL_SHA:raise ValueError('TopFace model byte SHA mismatch')
        self.nn=nn;self.np=np;self.closed=False
        self.kpu=self.ai2d=self.builder=self.destination=self.backing=None
        self.last=None;self.timing={}
        try:
            self.kpu=nn.kpu();self.kpu.load_kmodel(root+'/topface.kmodel')
            self.backing=np.zeros((1,3,192,320),dtype=np.uint8)
            self.destination=nn.from_numpy(self.backing)
            self.kpu.set_input_tensor(0,self.destination)
            self.ai2d=nn.ai2d()
            self.ai2d.set_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8,np.uint8)
            self.ai2d.set_pad_param(True,[0,0,0,0,6,6,0,0],0,[114]*3)
            self.ai2d.set_resize_param(True,nn.interp_method.tf_bilinear,nn.interp_mode.half_pixel)
            self.builder=self.ai2d.build([1,3,360,640],[1,3,192,320])
        except BaseException:
            self.close();raise

    def infer(self,image):
        if self.closed:raise RuntimeError('TopFace backend closed')
        if (image.width(),image.height())!=(640,360):raise ValueError('TopFace requires 640x360')
        array=image.to_numpy_ref()
        if tuple(array.shape)!=(3,360,640):raise ValueError('TopFace requires RGB planar CHW')
        before=time.ticks_us();source=self.nn.from_numpy(array)
        self.builder.run(source,self.destination);prepared=time.ticks_us()
        self.kpu.run();inferred=time.ticks_us()
        output=self.kpu.get_output_tensor(0);raw=output.to_numpy()
        self.last=decode_tensor(raw)
        decoded=time.ticks_us()
        self.timing={'ai2d_ms':time.ticks_diff(prepared,before)/1000,
                     'kpu_ms':time.ticks_diff(inferred,prepared)/1000,
                     'decode_ms':time.ticks_diff(decoded,inferred)/1000}
        raw=output=source=array=None
        return self.last

    def close(self):
        self.last=None
        self.kpu=self.builder=self.ai2d=self.destination=self.backing=None
        self.closed=True;gc.collect()
        if hasattr(self.nn,'shrink_memory_pool'):self.nn.shrink_memory_pool()


class TopFacePreview:
    """Own the packed array until its in-place RVV RGB565 image is drawn."""
    def __init__(self):self.backing=self.image=None

    def convert(self,frame):
        import image
        width,height=frame.width(),frame.height()
        planar=frame.to_numpy_ref()
        if tuple(planar.shape)!=(3,height,width):raise ValueError('Expected CHW camera frame')
        self.backing=planar.reshape((3,height*width)).transpose().copy().reshape((height,width,3))
        self.image=image.Image(width,height,image.RGB888,alloc=image.ALLOC_REF,
                               data=self.backing,cvt_565=True)
        return self.image

    def clear(self):self.image=self.backing=None


def annotate(canvas,decoded,offset_y=0):
    canvas.draw_string_advanced(8,8,20,decoded['capacity_state'],color=(255,255,255))
    for target in decoded['detections']:
        x1,y1,x2,y2=target['bbox_xyxy_visible']
        canvas.draw_rectangle(int(x1),int(y1)+offset_y,int(x2-x1),int(y2-y1),color=(0,255,0),thickness=2)
        cx,cy=target['center_xy_projected_3d']
        if target['center_visible_probability']>=.5 and 0<=cx<640 and 0<=cy<360:
            canvas.draw_cross(int(cx),int(cy)+offset_y,color=(255,0,255))
        canvas.draw_string_advanced(int(x1),max(0,int(y1)-20)+offset_y,16,
            str(target['color_id'])+' '+str(round(target['score'],2)),color=(255,255,255))
        ellipse=target['ellipse']
        if ellipse is None:continue
        ex,ey=ellipse['conic_center_xy'];theta=ellipse['orientation_deg']*math.pi/180;previous=None
        for point in range(25):
            angle=point*2*math.pi/24
            dx=ellipse['semi_major']*math.cos(angle);dy=ellipse['semi_minor']*math.sin(angle)
            current=(int(ex+dx*math.cos(theta)-dy*math.sin(theta)),int(ey+dx*math.sin(theta)+dy*math.cos(theta))+offset_y)
            if previous:canvas.draw_line(previous[0],previous[1],current[0],current[1],color=(255,255,0))
            previous=current
