import sys,time,json,gc,nncase_runtime as nn
from ulab import numpy as np
sys.path.insert(0,'/sdcard/kopt_20261112d')
from vision.yolo_candidates import decode_material_proposals,input_contract
k=tensor=output=data=values=None
try:
 print('POST_STAGE:load')
 k=nn.kpu();k.load_kmodel('/sdcard/kmfps1112/material.kmodel');print('POST_STAGE:loaded')
 with open('/sdcard/kmfps1112/input0.bin','rb') as f:raw=f.read()
 print('POST_STAGE:read:'+str(len(raw)))
 data=np.frombuffer(raw,dtype=np.uint8).reshape((1,3,224,224));print('POST_STAGE:array')
 tensor=nn.from_numpy(data);k.set_input_tensor(0,tensor);print('POST_STAGE:input')
 all_rows=[]
 for i in range(13):
  t0=time.ticks_us();k.run();t1=time.ticks_us();output=k.get_output_tensor(0);values=output.to_numpy();t2=time.ticks_us()
  if i==0:print('POST_STAGE:decode_begin')
  proposals=decode_material_proposals(values,input_contract(224),confidence=.25,nms=.45,limit=16,np=np)
  t3=time.ticks_us()
  if i==0:print('POST_STAGE:decode_done')
  if i>=3:all_rows.append([time.ticks_diff(t1,t0)/1000,time.ticks_diff(t2,t1)/1000,time.ticks_diff(t3,t2)/1000,time.ticks_diff(t3,t0)/1000])
  values=output=None
 print('POST_RESULT:'+json.dumps({'stages_ms':all_rows,'proposals':proposals,'scope':'KPU_OUTPUT_POSTPROCESS_PREPARED_REAL_IMAGE_NO_CAPTURE'}))
finally:
 values=output=tensor=data=None
 if k is not None:k.__del__()
 k=None;nn.shrink_memory_pool();gc.collect();print('POST_CLEANUP')

