import sys,time,json,gc
import nncase_runtime as nn
from ulab import numpy as np
k=ai=builder=dst=src=rgb=raw=output=values=None
try:
 print('DIRECT_STAGE:load')
 k=nn.kpu();k.load_kmodel('/sdcard/kmfps1112/material.kmodel')
 print('DIRECT_STAGE:input')
 with open('/sdcard/kmfps1112/frame0.rgb888p','rb') as f:raw=f.read()
 if len(raw)!=691200:raise ValueError('RGB size')
 rgb=np.frombuffer(raw,dtype=np.uint8).reshape((1,3,360,640))
 src=nn.from_numpy(rgb)
 dst=k.get_input_tensor(0)
 print('DIRECT_STAGE:ai2d_build')
 ai=nn.ai2d();ai.set_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8,np.uint8)
 ai.set_pad_param(True,[0,0,0,0,49,49,0,0],0,[114,114,114])
 ai.set_resize_param(True,nn.interp_method.tf_bilinear,nn.interp_mode.half_pixel)
 builder=ai.build([1,3,360,640],[1,3,224,224])
 print('DIRECT_STAGE:ai2d_run')
 builder.run(src,dst)
 print('DIRECT_STAGE:kpu_run')
 k.run()
 print('DIRECT_STAGE:output')
 output=k.get_output_tensor(0);values=output.to_numpy();print('DIRECT_SHAPE:'+json.dumps(list(values.shape)))
 values=output=None
 timings=[]
 for i in range(8):
  t0=time.ticks_us();builder.run(src,dst);t1=time.ticks_us();k.run();t2=time.ticks_us()
  output=k.get_output_tensor(0);values=output.to_numpy();t3=time.ticks_us()
  if i>=3:timings.append([time.ticks_diff(t1,t0)/1000,time.ticks_diff(t2,t1)/1000,time.ticks_diff(t3,t2)/1000])
  values=output=None
 print('DIRECT_RESULT:'+json.dumps({'measured':5,'stages_ms':timings,'scope':'FROZEN_RGB888P_AI2D_KPU_OUTPUT_NO_POST_NO_CAMERA'}))
finally:
 values=output=src=dst=builder=ai=rgb=raw=None
 if k is not None:k.__del__()
 k=None;gc.collect();nn.shrink_memory_pool();gc.collect()
 print('DIRECT_CLEANUP')

