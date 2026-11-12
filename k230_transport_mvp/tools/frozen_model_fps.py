"""Frozen real-image CPU/KPU throughput. No camera, training, or motion IO."""
import json
import time
import gc
import hashlib
import binascii


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        while True:
            block=stream.read(4096)
            if not block:break
            h.update(block)
    return binascii.hexlify(h.digest()).decode()


def metrics(samples):
    ordered=sorted(samples);n=len(ordered)
    if not n or min(ordered)<0:raise ValueError('Measured nonnegative timings required')
    mean=sum(ordered)/n
    return {'iterations':n,'mean_ms':mean,'min_ms':ordered[0],'max_ms':ordered[-1],
            'p95_ms':ordered[min(n-1,(95*n+99)//100-1)],'fps':1000/mean if mean else None}


def release(value,errors,name):
    if value is None:return
    try:
        if hasattr(value,'release'):value.release()
        elif hasattr(value,'__del__'):value.__del__()
    except Exception as error:errors.append(name+': '+str(error))


def run(root,iterations=25,cpu_iterations=4,preprocessing='prepared'):
    if type(iterations) is not int or not 5<=iterations<=50 or type(cpu_iterations) is not int or not 0<=cpu_iterations<=10:
        raise ValueError('Bounded CPU/KPU iteration budgets required')
    if preprocessing not in ('prepared','ai2d'):raise ValueError('Explicit preprocessing mode required')
    from ulab import numpy as np
    import nncase_runtime as nn
    from k230.tuning.resources import snapshot
    from vision.yolo_candidates import decode_material_proposals
    with open(root+'/identity.json') as stream:identity=json.load(stream)
    if identity.get('retrained') is not False or identity['training_provenance']['real_optimizer_updates']!=4:
        raise ValueError('Original four-update pilot identity required')
    model=root+'/material.kmodel'
    if digest(model)!=identity['kmodel_sha256']:raise ValueError('Kmodel identity mismatch')
    for row in identity['images']:
        if row['split']!='train':raise ValueError('Only frozen training-group benchmark inputs admitted')
        for key in ('native','rgb','input'):
            if digest(root+'/'+row[key+'_file'])!=row[key+'_sha256']:raise ValueError('Frozen image bytes changed')
    def hardware_snapshot(phase):
        value=snapshot();value['kpu_note']='KPU model benchmark '+phase+'; hardware utilization percentage is unavailable'
        return value
    report={'scope':'FROZEN_REAL_TRAIN_IMAGES_BOARD_THROUGHPUT','model_sha256':identity['kmodel_sha256'],
        'onnx_sha256':identity['onnx_sha256'],'checkpoint_sha256':identity['training_provenance']['material_trained_checkpoint_sha256'],
        'original_optimizer_updates':4,'new_optimizer_updates':0,'task_accuracy_verified':False,
        'live_camera_fps_verified':False,'jpeg_display_included':False,'rows':[],
        'resource_before':hardware_snapshot('before'),'cleanup_errors':[],'runtime_version':str(nn.version()),
        'preprocessing_mode':preprocessing,'preprocessing_included':preprocessing=='ai2d'}
    print('FROZEN_MODEL_START:'+json.dumps({k:report[k] for k in ('scope','model_sha256','original_optimizer_updates')}))
    kpu=ai2d=builder=destination=source_tensor=output=None
    target=source_image=rgb=raw=values=proposals=None
    try:
        # Original full six-color/paper chain, fresh computation on each call.
        cpu=[]
        if cpu_iterations:
            import image
            from k230.tuning.live_optimized import native_views
            from vision.ellipse_native_optimized import configure_boundary_backend
            from vision.ellipse_native import configure_boundary_backend as configure_original
            configure_boundary_backend(True);configure_original(True)
            with open(root+'/protected-state.json') as stream:state=json.load(stream)
        for row in (identity['images'] if cpu_iterations else []):
            frame=image.Image(640,360,image.RGB565)
            with open(root+'/'+row['native_file'],'rb') as stream:
                if stream.readinto(frame.bytearray())!=640*360*2:raise ValueError('Incomplete source image')
            timings=[]
            for i in range(cpu_iterations+1):
                start=time.ticks_us()
                _,_,geometry=native_views(frame,state['profile'],'RING','PROCESS',1,selected=True,
                    diagnostic_material_colors=[1,2,3,4,5,6],render=False)
                elapsed=time.ticks_diff(time.ticks_us(),start)/1000
                if i:timings.append(elapsed)
            cpu.append({'image_id':row['image_id'],'timing':metrics(timings),
                'materials':len(geometry['material_fits']),'paper_targets':len(geometry['ring_groups']),
                'scene_complete':geometry['material_complete']})
            frame=geometry=None;gc.collect()
            print('FROZEN_CPU_IMAGE:'+json.dumps(cpu[-1]))
        report['cpu_full_geometry']=cpu
        print('FROZEN_STAGE:KPU_LOAD_BEGIN')
        kpu=nn.kpu();start=time.ticks_us();kpu.load_kmodel(model)
        report['load_ms']=time.ticks_diff(time.ticks_us(),start)/1000
        print('FROZEN_STAGE:KPU_LOAD_COMPLETE')
        report['input_desc']=kpu.inputs_desc(0);report['output_desc']=kpu.outputs_desc(0)
        if preprocessing=='ai2d':
            destination=kpu.get_input_tensor(0)
            ai2d=nn.ai2d();ai2d.set_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8,np.uint8)
            ai2d.set_pad_param(True,[0,0,0,0,49,49,0,0],0,[114,114,114])
            ai2d.set_resize_param(True,nn.interp_method.tf_bilinear,nn.interp_mode.half_pixel)
            builder=ai2d.build([1,3,360,640],[1,3,224,224])
            print('FROZEN_STAGE:AI2D_BUILT')
        for row in identity['images']:
            name=row['rgb_file'] if preprocessing=='ai2d' else row['input_file']
            if preprocessing=='ai2d':
                import image
                source_image=image.Image(640,360,image.RGBP888)
                with open(root+'/'+name,'rb') as stream:
                    if stream.readinto(source_image.bytearray())!=691200:raise ValueError('Incomplete planar RGB image')
                rgb=source_image.to_numpy_ref()
            else:
                with open(root+'/'+name,'rb') as stream:raw=stream.read()
                rgb=np.frombuffer(raw,dtype=np.uint8).reshape((1,3,224,224))
            source_tensor=nn.from_numpy(rgb)
            if builder is not None:
                builder.run(source_tensor,destination)
                print('FROZEN_STAGE:AI2D_FIRST_RUN')
            else:kpu.set_input_tensor(0,source_tensor)
            preprocess={'full_tensor_equivalence_verified':False,
                        'cpu_readback':'NOT_REQUESTED' if preprocessing=='ai2d' else 'PREPARED_HOST_INPUT_SHA_VERIFIED'}
            print('FROZEN_STAGE:INPUT_READY_FOR_KPU')
            stages={'ai2d':[],'kpu':[],'output':[],'postprocess':[],'total':[]}
            for i in range(iterations+3):
                t0=time.ticks_us()
                if builder is not None:builder.run(source_tensor,destination)
                t1=time.ticks_us()
                if i==0:print('FROZEN_STAGE:FIRST_KPU_RUN_BEGIN')
                kpu.run();t2=time.ticks_us()
                if i==0:print('FROZEN_STAGE:FIRST_KPU_RUN_END')
                output=kpu.get_output_tensor(0);values=output.to_numpy();t3=time.ticks_us()
                proposals=decode_material_proposals(values,row['input_contract'],confidence=.25,nms=.45,limit=16,np=np)
                t4=time.ticks_us()
                if i>=3:
                    for key,a,b in (('ai2d',t0,t1),('kpu',t1,t2),('output',t2,t3),('postprocess',t3,t4),('total',t0,t4)):
                        stages[key].append(time.ticks_diff(b,a)/1000)
                values=None;release(output,report['cleanup_errors'],'output');output=None
            current={'image_id':row['image_id'],'input_contract':row['input_contract'],
                'stages':{key:metrics(value) for key,value in stages.items()},'ai2d_vs_host_input':preprocess,
                'proposal_count':len(proposals['proposals']),'proposal_status':proposals['status'],
                'sample_proposals':proposals['proposals'][:3],'resource_during':hardware_snapshot('during')}
            report['rows'].append(current);print('FROZEN_KPU_IMAGE:'+json.dumps(current))
            proposals=None
            release(source_tensor,report['cleanup_errors'],'source');source_tensor=None
            rgb=raw=source_image=None;gc.collect()
        report['status']='SPEED_MEASURED_NOT_ACCURACY_ACCEPTED'
    except Exception as error:
        report.update(status='FAILED',error=str(error))
        if hasattr(__import__('sys'),'print_exception'):__import__('sys').print_exception(error)
    finally:
        release(output,report['cleanup_errors'],'output')
        release(source_tensor,report['cleanup_errors'],'source')
        release(builder,report['cleanup_errors'],'builder')
        release(ai2d,report['cleanup_errors'],'ai2d')
        release(destination,report['cleanup_errors'],'destination')
        release(kpu,report['cleanup_errors'],'kpu')
        output=source_tensor=builder=ai2d=destination=kpu=None
        values=proposals=target=source_image=rgb=raw=None
        try:nn.shrink_memory_pool()
        except Exception as error:report['cleanup_errors'].append('pool: '+str(error))
        gc.collect();report['resource_after']=hardware_snapshot('after')
    print('FROZEN_MODEL_RESULT:'+json.dumps(report))
    return report
