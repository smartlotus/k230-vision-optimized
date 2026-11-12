import time,json,gc,image,nncase_runtime as nn
from ulab import numpy as np
k=ai=builder=dst=src=img=rgb=output=values=None
try:
 print('MMZ_STAGE:image_allocate')
 img=image.Image(640,360,image.RGBP888)
 print('MMZ_STAGE:image_readinto')
 with open('/sdcard/kmfps1112/frame0.rgb888p','rb') as f:count=f.readinto(img.bytearray())
 print('MMZ_STAGE:image_read_complete:'+str(count))
 rgb=img.to_numpy_ref();print('MMZ_SOURCE_SHAPE:'+json.dumps(list(rgb.shape)))
 src=nn.from_numpy(rgb)
 print('MMZ_STAGE:kpu_load')
 k=nn.kpu();k.load_kmodel('/sdcard/kmfps1112/material.kmodel');dst=k.get_input_tensor(0)
 ai=nn.ai2d();ai.set_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8,np.uint8)
 ai.set_pad_param(True,[0,0,0,0,49,49,0,0],0,[114,114,114]);ai.set_resize_param(True,nn.interp_method.tf_bilinear,nn.interp_mode.half_pixel)
 builder=ai.build([1,3,360,640],[1,3,224,224]);print('MMZ_STAGE:ai2d_run')
 builder.run(src,dst);print('MMZ_STAGE:ai2d_done')
 k.run();print('MMZ_STAGE:kpu_done')
 output=k.get_output_tensor(0);values=output.to_numpy();print('MMZ_OUTPUT_SHAPE:'+json.dumps(list(values.shape)))
 values=output=None
 rows=[]
 for i in range(8):
  a=time.ticks_us();builder.run(src,dst);b=time.ticks_us();k.run();c=time.ticks_us()
  output=k.get_output_tensor(0);values=output.to_numpy();d=time.ticks_us()
  if i>=3:rows.append([time.ticks_diff(b,a)/1000,time.ticks_diff(c,b)/1000,time.ticks_diff(d,c)/1000])
  values=output=None
 print('MMZ_RESULT:'+json.dumps({'stages_ms':rows,'scope':'FROZEN_IMAGE_AI2D_KPU_OUTPUT','iterations':5}))
finally:
 values=output=src=dst=builder=ai=rgb=img=None
 if k is not None:k.__del__()
 k=None;gc.collect();nn.shrink_memory_pool();gc.collect();print('MMZ_CLEANUP')

