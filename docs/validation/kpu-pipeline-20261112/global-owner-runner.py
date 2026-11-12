"""Bounded frozen-image timing; quantized score projection is benchmark-only."""
import json
import time
import gc
import hashlib
import binascii


def cooperate():
    """Allow the CanMV runtime to handle IDE interruption/USB work."""
    import os
    if hasattr(os,'exitpoint'):os.exitpoint()


def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        while True:
            block=stream.read(4096)
            if not block:break
            h.update(block)
            cooperate()
    return binascii.hexlify(h.digest()).decode()


def latency(values):
    if not values or any(v<0 for v in values):raise ValueError('Nonnegative measured latencies required')
    mean=sum(values)/len(values)
    return dict(count=len(values),mean_ms=mean,min_ms=min(values),max_ms=max(values),
                fps=1000/mean if mean else None)


def timing_proposals(raw,contract,np,trace=None):
    """Retain raw validity separately; never authorizes production observations.

    Tiny quantization overshoots can cross the exact sigmoid range. Saturating
    scores here measures a valid postprocessor workload at unchanged confidence;
    the original strict decoder and task-acceptance gate are untouched.
    """
    if tuple(raw.shape)!=(1,5,1029) or not bool(np.all(np.isfinite(raw))):
        raise ValueError('Finite single-class224 output required')
    scores=raw[0,4,:]
    minimum,maximum=float(np.min(scores)),float(np.max(scores))
    validity=dict(raw_min_score=minimum,raw_max_score=maximum,
        raw_probability_range_valid=minimum>=0 and maximum<=1,
        scores_projected=int(np.sum(scores<0))+int(np.sum(scores>1)),
        projection_scope='TIMING_ONLY',task_accuracy_verified=False)
    if minimum < -.001 or maximum > 1.001:
        raise ValueError('Quantized score overshoot exceeds diagnostic projection bound')
    if trace:trace('OUTPUT_VALIDATED')
    # Benchmark-only scalar NMS avoids native pointer ownership while isolating
    # repeated inference. Subthreshold negative scores are never proposals.
    validity['projection_scope']='FILTER_SUBTHRESHOLD_ONLY_TIMING'
    result=bounded_proposals(raw,contract,np)
    if trace:trace('DECODE_COMPLETE')
    return result,validity


def bounded_proposals(raw,contract,np):
    from vision.yolo_candidates import input_contract
    if contract!=input_contract(224):raise ValueError('Exact224 source transform required')
    if bool(np.any(raw[0,2:4,:]<0)) or bool(np.any(abs(raw[0,0:4,:])>896)):
        raise ValueError('Invalid decoded coordinate range')
    scores=raw[0,4,:]
    if float(np.max(scores))<.25:
        return {'proposals':[],'status':'CONFIDENCE_EMPTY','complete':True}
    candidates=[];top=contract['pad_tblr'][0];gain=contract['gain']
    for index,score in enumerate(scores.tolist()):
        if score<.25:continue
        if len(candidates)>=64:
            return {'proposals':[],'status':'CANDIDATE_CAPACITY_REACHED','complete':False}
        cx,cy,w,h=[float(raw[0,k,index]) for k in range(4)]
        x0=max(0.,(cx-w/2)/gain);x1=min(640.,(cx+w/2)/gain)
        y0=max(0.,(cy-h/2-top)/gain);y1=min(360.,(cy+h/2-top)/gain)
        if x1>x0 and y1>y0:candidates.append({'bbox_xywh':[x0,y0,x1-x0,y1-y0],
            'score':min(1.,score),'proposal_only':True,'geometry_verified':False})
    candidates.sort(key=lambda row:row['score'],reverse=True)
    result=[]
    for candidate in candidates:
        ax,ay,aw,ah=candidate['bbox_xywh'];suppressed=False
        for old in result:
            bx,by,bw,bh=old['bbox_xywh']
            overlap=max(0.,min(ax+aw,bx+bw)-max(ax,bx))*max(0.,min(ay+ah,by+bh)-max(ay,by))
            if overlap/(aw*ah+bw*bh-overlap)>.45:suppressed=True;break
        if not suppressed:result.append(candidate)
        if len(result)>=16:return {'proposals':result,'status':'CANDIDATE_CAPACITY_REACHED','complete':False}
    return {'proposals':result,'status':'MODEL_PROPOSALS_ONLY','complete':True}


def scalar_timing_proposals(raw,contract):
    """Pointer-free bounded postprocessor, with no temporary ndarray views."""
    from math import isfinite
    from vision.yolo_candidates import input_contract
    if contract!=input_contract(224) or tuple(raw.shape)!=(1,5,1029):raise ValueError('Exact224 output required')
    candidates=[];minimum=1.;maximum=0.;negative=above=0
    top=contract['pad_tblr'][0];gain=contract['gain']
    overflow=False
    for index in range(1029):
        score=float(raw[0,4,index])
        if not isfinite(score) or not -.001<=score<=1.001:raise ValueError('Invalid quantized diagnostic score')
        if score<minimum:minimum=score
        if score>maximum:maximum=score
        if score<0:negative+=1
        if score>1:above+=1
        if score<.25:continue
        if len(candidates)>=64:overflow=True;continue
        cx,cy,w,h=[float(raw[0,k,index]) for k in range(4)]
        if any(not isfinite(v) or abs(v)>896 for v in (cx,cy,w,h)) or w<0 or h<0:
            raise ValueError('Invalid positive proposal coordinates')
        x0=max(0.,(cx-w/2)/gain);x1=min(640.,(cx+w/2)/gain)
        y0=max(0.,(cy-h/2-top)/gain);y1=min(360.,(cy+h/2-top)/gain)
        if x1>x0 and y1>y0:candidates.append({'bbox_xywh':[x0,y0,x1-x0,y1-y0],
            'score':min(1.,score),'proposal_only':True,'geometry_verified':False})
    validity=dict(raw_min_score=minimum,raw_max_score=maximum,raw_probability_range_valid=negative==0 and above==0,
        scores_projected=negative+above,projection_scope='FILTER_SUBTHRESHOLD_ONLY_TIMING',task_accuracy_verified=False)
    if overflow:return {'proposals':[],'status':'CANDIDATE_CAPACITY_REACHED','complete':False},validity
    candidates.sort(key=lambda row:row['score'],reverse=True);result=[]
    for candidate in candidates:
        ax,ay,aw,ah=candidate['bbox_xywh'];suppressed=False
        for old in result:
            bx,by,bw,bh=old['bbox_xywh']
            overlap=max(0.,min(ax+aw,bx+bw)-max(ax,bx))*max(0.,min(ay+ah,by+bh)-max(ay,by))
            if overlap/(aw*ah+bw*bh-overlap)>.45:suppressed=True;break
        if not suppressed:result.append(candidate)
        if len(result)>=16:return {'proposals':result,'status':'CANDIDATE_CAPACITY_REACHED','complete':False},validity
    return {'proposals':result,'status':'MODEL_PROPOSALS_ONLY','complete':True},validity


def release(value,errors,name):
    if value is None:return
    try:
        if hasattr(value,'release'):value.release()
        elif hasattr(value,'__del__'):value.__del__()
    except Exception as error:errors.append(name+': '+str(error))


def probe(root,mode='prepared',iterations=20,image_limit=4):
    if mode not in ('prepared','ai2d') or type(iterations) is not int or not 5<=iterations<=30:
        raise ValueError('Explicit mode and5..30 measured iterations required')
    if type(image_limit) is not int or not 1<=image_limit<=4:raise ValueError('One to four frozen images required')
    import nncase_runtime as nn
    import os
    if hasattr(os,'exitpoint'):os.exitpoint(os.EXITPOINT_ENABLE)
    from ulab import numpy as np
    from k230.tuning.resources import snapshot
    with open(root+'/identity.json') as stream:identity=json.load(stream)
    if identity['training_provenance']['real_optimizer_updates']!=4 or identity['retrained'] is not False:
        raise ValueError('Original four-update identity required')
    if sha_file(root+'/material.kmodel')!=identity['kmodel_sha256']:raise ValueError('Model SHA mismatch')
    report=dict(scope='FROZEN_REAL_TRAIN_IMAGES_TIMING_ONLY',mode=mode,model_sha256=identity['kmodel_sha256'],
        original_optimizer_updates=4,new_optimizer_updates=0,live_camera_included=False,display_included=False,
        task_accuracy_verified=False,rows=[],cleanup_errors=[],status='STARTED')
    def resources():
        state=snapshot();state['kpu_note']='Model inference active in this benchmark; KPU usage percentage unavailable'
        return state
    report['resource_before']=resources()
    kpu=source=destination=builder=ai2d=output=raw_input=data=source_image=values=None
    try:
        print('PIPELINE_STAGE:MODEL_LOAD')
        kpu=nn.kpu();kpu.load_kmodel(root+'/material.kmodel')
        print('PIPELINE_STAGE:MODEL_LOADED')
        if mode=='ai2d':
            destination=kpu.get_input_tensor(0)
            ai2d=nn.ai2d();ai2d.set_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8,np.uint8)
            ai2d.set_pad_param(True,[0,0,0,0,49,49,0,0],0,[114,114,114])
            ai2d.set_resize_param(True,nn.interp_method.tf_bilinear,nn.interp_mode.half_pixel)
            builder=ai2d.build([1,3,360,640],[1,3,224,224])
            print('PIPELINE_STAGE:AI2D_BUILT')
        for row in identity['images'][:image_limit]:
            key='rgb' if mode=='ai2d' else 'input'
            path=root+'/'+row[key+'_file']
            if sha_file(path)!=row[key+'_sha256']:raise ValueError('Frozen input SHA mismatch')
            print('PIPELINE_STAGE:INPUT_READ_'+row['image_id'][:8])
            if mode=='ai2d':
                import image
                source_image=image.Image(640,360,image.RGBP888)
                with open(path,'rb') as stream:
                    if stream.readinto(source_image.bytearray())!=691200:raise ValueError('Truncated planar image')
                data=source_image.to_numpy_ref()
            else:
                with open(path,'rb') as stream:raw_input=stream.read()
                data=np.frombuffer(raw_input,dtype=np.uint8).reshape((1,3,224,224))
            source=nn.from_numpy(data)
            if mode=='prepared':kpu.set_input_tensor(0,source)
            cooperate()
            print('PIPELINE_STAGE:INPUT_READY')
            measured={name:[] for name in ('gc','preprocess','kpu','output','postprocess','compute','iteration')}
            for i in range(iterations+3):
                iteration_start=time.ticks_us()
                a=time.ticks_us()
                if builder is not None:builder.run(source,destination)
                b=time.ticks_us();kpu.run();c=time.ticks_us()
                if i==0:print('PIPELINE_STAGE:KPU_FIRST_DONE')
                # Retain the native output handle while the interpreter reuses
                # its buffer; avoid destroying wrappers between model runs.
                output=kpu.get_output_tensor(0)
                values=output.to_numpy();d=time.ticks_us()
                if i==0:print('PIPELINE_STAGE:OUTPUT_FIRST_READY')
                trace=(lambda stage:print('PIPELINE_STAGE:'+stage)) if i==0 else None
                proposals,validity=scalar_timing_proposals(values,row['input_contract'])
                e=time.ticks_us()
                values=output=None
                if i==0:print('PIPELINE_STAGE:OUTPUT_VALUES_RELEASED_HANDLE_RETAINED')
                cooperate()
                time.sleep_ms(1)
                if i<3:print('PIPELINE_STAGE:WARMUP_DONE_'+str(i))
                f=time.ticks_us()
                if i>=3:
                    if i==3:print('PIPELINE_STAGE:RECORD_BEGIN')
                    for name,start,end in (('gc',iteration_start,a),('preprocess',a,b),('kpu',b,c),('output',c,d),
                            ('postprocess',d,e),('compute',a,e),('iteration',iteration_start,f)):
                        if i==3:print('PIPELINE_STAGE:RECORD_'+name)
                        measured[name].append(time.ticks_diff(end,start)/1000)
            print('PIPELINE_STAGE:MEASUREMENTS_COMPLETE')
            result=dict(image_id=row['image_id'],stage_timings={k:latency(v) for k,v in measured.items()},
                raw_output_validation=validity,proposals=len(proposals['proposals']),
                native_postprocess_status=proposals['status'],
                resources={'collected':False,'reason':'Isolate RTOS telemetry from active KPU timing'})
            report['rows'].append(result);print('PIPELINE_IMAGE:'+json.dumps(result))
            release(source,report['cleanup_errors'],'source');source=None
            data=raw_input=source_image=None;gc.collect()
        report['status']='TIMING_COMPLETE_NOT_TASK_ACCEPTED'
    except Exception as error:
        report.update(status='FAILED',error=str(error))
        import sys
        if hasattr(sys,'print_exception'):sys.print_exception(error)
    finally:
        print('PIPELINE_STAGE:CLEANUP_BEGIN')
        values=None
        release(kpu,report['cleanup_errors'],'kpu');kpu=None
        release(output,report['cleanup_errors'],'output');output=None
        release(source,report['cleanup_errors'],'source');source=None
        release(destination,report['cleanup_errors'],'destination');destination=None
        release(builder,report['cleanup_errors'],'builder');builder=None
        release(ai2d,report['cleanup_errors'],'ai2d');ai2d=None
        data=raw_input=source_image=None;gc.collect()
        try:nn.shrink_memory_pool()
        except Exception as error:report['cleanup_errors'].append('pool: '+str(error))
        gc.collect();report['resource_after']=resources()
    print('PIPELINE_RESULT:'+json.dumps(report))
    return report


import sys,os,time,json,gc,nncase_runtime as nn
from ulab import numpy as np
sys.path.insert(0,'/sdcard/kopt_20261112d')
os.exitpoint(os.EXITPOINT_ENABLE)
with open('/sdcard/kmfps1112/identity.json') as f:ident=json.load(f)
assert sha_file('/sdcard/kmfps1112/material.kmodel')==ident['kmodel_sha256']
model=nn.kpu();tensor=output=data=values=raw_input=None
try:
 model.load_kmodel('/sdcard/kmfps1112/material.kmodel');print('GLOBAL_MODEL_LOADED')
 with open('/sdcard/kmfps1112/input0.bin','rb') as f:raw_input=f.read()
 data=np.frombuffer(raw_input,dtype=np.uint8).reshape((1,3,224,224))
 tensor=nn.from_numpy(data);model.set_input_tensor(0,tensor)
 measurements=[]
 for i in range(13):
  os.exitpoint();t0=time.ticks_us();gc.collect();t1=time.ticks_us()
  model.run();t2=time.ticks_us();output=model.get_output_tensor(0)
  values=output.to_numpy().copy();t3=time.ticks_us()
  proposals,validity=scalar_timing_proposals(values,ident['images'][0]['input_contract']);t4=time.ticks_us()
  values=output=None;time.sleep_ms(1);t5=time.ticks_us()
  if i>=3:measurements.append([time.ticks_diff(t1,t0)/1000,time.ticks_diff(t2,t1)/1000,time.ticks_diff(t3,t2)/1000,time.ticks_diff(t4,t3)/1000,time.ticks_diff(t5,t0)/1000])
  print('GLOBAL_ITER_DONE:'+str(i))
 print('GLOBAL_RESULT:'+json.dumps({'timings_ms':measurements,'proposals':len(proposals['proposals']),'raw_validity':validity,'iterations':10,'source':'frozen train image0','live_camera':False}))
finally:
 model.__del__();model=None;tensor=output=values=data=raw_input=None;gc.collect();nn.shrink_memory_pool();gc.collect();print('GLOBAL_CLEANUP')
