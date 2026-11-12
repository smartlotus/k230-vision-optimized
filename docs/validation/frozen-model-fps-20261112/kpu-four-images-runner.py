import time,json,gc,hashlib,binascii,nncase_runtime as nn
from ulab import numpy as np
root='/sdcard/kmfps1112'
with open(root+'/identity.json') as f:identity=json.load(f)
all_results=[]
for row in identity['images']:
 k=tensor=output=data=None
 try:
  k=nn.kpu();k.load_kmodel(root+'/material.kmodel')
  with open(root+'/'+row['input_file'],'rb') as f:raw=f.read()
  if binascii.hexlify(hashlib.sha256(raw).digest()).decode()!=row['input_sha256']:raise ValueError('Input SHA mismatch')
  data=np.frombuffer(raw,dtype=np.uint8).reshape((1,3,224,224));raw=None;gc.collect()
  tensor=nn.from_numpy(data);k.set_input_tensor(0,tensor)
  times=[];copies=[]
  for i in range(23):
   t=time.ticks_us();k.run();elapsed=time.ticks_diff(time.ticks_us(),t)/1000
   t=time.ticks_us();output=k.get_output_tensor(0);values=output.to_numpy();copied=time.ticks_diff(time.ticks_us(),t)/1000
   if i>=3:times.append(elapsed);copies.append(copied)
   output=values=None
  result={'image_id':row['image_id'],'iterations':20,'kpu_ms':times,'output_copy_ms':copies,'mean_ms':sum(times)/20,'fps':20000/sum(times)}
  all_results.append(result);print('KPU_FROZEN_IMAGE:'+json.dumps(result))
 finally:
  output=values=tensor=data=None
  if k is not None:k.__del__()
  k=None;nn.shrink_memory_pool();gc.collect()
print('KPU_FROZEN_ALL:'+json.dumps({'scope':'EXISTING_PILOT_FROZEN_REAL_IMAGE_KPU_ONLY','rows':all_results,'images':len(all_results),'new_optimizer_updates':0,'preprocessing_included':False,'display_included':False,'accuracy_verified':False,'postprocess_apis':{n:hasattr(np,n) for n in ('maximum','minimum','any','all','isfinite')}}))

