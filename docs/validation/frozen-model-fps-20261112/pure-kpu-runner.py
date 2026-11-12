import time,json,gc,nncase_runtime as nn
from ulab import numpy as np
k=tensor=output=None
try:
 print('PURE_KPU_LOAD_BEGIN')
 k=nn.kpu();k.load_kmodel('/sdcard/kmfps1112/material.kmodel');print('PURE_KPU_LOADED')
 with open('/sdcard/kmfps1112/input0.bin','rb') as f:raw=f.read()
 data=np.frombuffer(raw,dtype=np.uint8).reshape((1,3,224,224));raw=None;gc.collect()
 tensor=nn.from_numpy(data);k.set_input_tensor(0,tensor);print('PURE_KPU_INPUT_SET')
 times=[];copy=[]
 for i in range(23):
  if i==0:print('PURE_KPU_FIRST_RUN_BEGIN')
  t=time.ticks_us();k.run();elapsed=time.ticks_diff(time.ticks_us(),t)/1000
  if i==0:print('PURE_KPU_FIRST_RUN_END')
  t=time.ticks_us();output=k.get_output_tensor(0);values=output.to_numpy();copied=time.ticks_diff(time.ticks_us(),t)/1000
  if i>=3:times.append(elapsed);copy.append(copied)
  values=output=None
 print('PURE_KPU_RESULT:'+json.dumps({'scope':'FROZEN_REAL_IMAGE_EXISTING_PILOT_KPU_ONLY','iterations':20,'run_ms':times,'copy_ms':copy,'mean_run_ms':sum(times)/len(times),'fps':1000*len(times)/sum(times),'live_camera_fps':False,'accuracy_verified':False}))
finally:
 output=values=tensor=data=None
 if k is not None:k.__del__()
 k=None;nn.shrink_memory_pool();gc.collect();print('PURE_KPU_CLOSED')

