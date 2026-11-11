"""Bounded on-board runtime probe; synthetic input proves no target recognition.

Execute only after the preview's cooperative stop and observed interpreter idle.
The model is an existing vendor example. This script never changes camera roles,
profiles, production startup files or performs physical actions.
"""


def probe(model_path='/sdcard/examples/kmodel/yolov8n_224.kmodel',iterations=20):
    import time,gc,json,hashlib,binascii
    import nncase_runtime as nn
    import ulab.numpy as np
    if type(iterations) is not int or not 1<=iterations<=50:
        raise ValueError('One to fifty measured iterations required')
    digest=hashlib.sha256();model_bytes=0
    with open(model_path,'rb') as stream:
        while True:
            chunk=stream.read(16384)
            if not chunk:break
            digest.update(chunk);model_bytes+=len(chunk)
    result={'scope':'NATIVE_SYNTHETIC_MODEL_RUNTIME_PROBE','model_path':model_path,
        'model_bytes':model_bytes,'model_sha256':binascii.hexlify(digest.digest()).decode(),
        'iterations':iterations,'warmup_iterations':3,'input_shape':[1,3,224,224],
        'input_source':'SYNTHETIC_CONSTANT_UINT8_128','task_model':False,
        'recognition_verified':False,'camera_pipeline_benchmarked':False,
        'physical_action':False,'runtime_version':str(nn.version())}
    kpu=tensor=data=output=values=None
    timings=[];copy_timings=[];cleanup_errors=[]
    try:
        start=time.ticks_us();kpu=nn.kpu();kpu.load_kmodel(model_path)
        result['load_ms']=time.ticks_diff(time.ticks_us(),start)/1000
        if kpu.inputs_size()!=1:raise ValueError('Example must have exactly one input')
        result['output_count']=kpu.outputs_size()
        data=np.ones((1,3,224,224),dtype=np.uint8)*128
        tensor=nn.from_numpy(data);kpu.set_input_tensor(0,tensor)
        for index in range(iterations+3):
            start=time.ticks_us();kpu.run()
            elapsed=time.ticks_diff(time.ticks_us(),start)/1000
            copy_start=time.ticks_us();shapes=[]
            for j in range(kpu.outputs_size()):
                output=kpu.get_output_tensor(j);values=output.to_numpy()
                shapes.append(list(values.shape));values=output=None
            copying=time.ticks_diff(time.ticks_us(),copy_start)/1000
            if index>=3:timings.append(elapsed);copy_timings.append(copying)
        result.update(status='PASS_RUNTIME_ONLY',run_ms=timings,output_copy_ms=copy_timings,
            output_shapes=shapes,run_mean_ms=sum(timings)/len(timings),
            run_min_ms=min(timings),run_max_ms=max(timings),
            output_copy_mean_ms=sum(copy_timings)/len(copy_timings))
    except Exception as error:
        result.update(status='FAILED',error=str(error))
    finally:
        values=output=tensor=data=None
        if kpu is not None:
            try:kpu.__del__()
            except Exception as error:cleanup_errors.append('kpu_destroy: '+str(error))
        kpu=None
        try:nn.shrink_memory_pool()
        except Exception as error:cleanup_errors.append('pool_shrink: '+str(error))
        try:gc.collect()
        except Exception as error:cleanup_errors.append('gc: '+str(error))
    result['cleanup_errors']=cleanup_errors
    if cleanup_errors and result['status']=='PASS_RUNTIME_ONLY':result['status']='FAILED_CLEANUP'
    return result


if __name__=='__main__':
    import json
    try:result=probe()
    except Exception as error:
        result={'status':'FAILED','scope':'NATIVE_SYNTHETIC_MODEL_RUNTIME_PROBE',
                'error':str(error),'recognition_verified':False,'physical_action':False}
    print('KPU_BOUNDED_PROBE:'+json.dumps(result))
