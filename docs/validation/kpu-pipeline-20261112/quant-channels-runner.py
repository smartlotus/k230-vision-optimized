import sys,time,json,gc,nncase_runtime as nn
from ulab import numpy as np
k=tensor=out=data=values=None
try:
 k=nn.kpu();k.load_kmodel('/sdcard/kmfps1112/material.kmodel')
 with open('/sdcard/kmfps1112/input0.bin','rb') as f:raw=f.read()
 data=np.frombuffer(raw,dtype=np.uint8).reshape((1,3,224,224));tensor=nn.from_numpy(data);k.set_input_tensor(0,tensor);k.run()
 out=k.get_output_tensor(0);values=out.to_numpy()
 stats=[]
 for i in range(5):
  channel=values[0,i,:]
  stats.append({'channel':i,'minimum':float(np.min(channel)),'maximum':float(np.max(channel)),'negative':int(np.sum(channel<0)),'above1':int(np.sum(channel>1))})
 print('QUANT_CHANNELS:'+json.dumps({'shape':list(values.shape),'finite':bool(np.all(np.isfinite(values))),'channels':stats,'sample_scores':values[0,4,:10].tolist()}))
finally:
 channel=values=out=tensor=data=None
 if k is not None:k.__del__()
 k=None;nn.shrink_memory_pool();gc.collect();print('QUANT_CHECK_CLOSED')

