"""Compare same captured source/prepared/raw bytes; never infer native FPS."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from cloud_store import atomic_json,sha
from compile_trained_topface import compare_detections
from export_nanodet_candidate import session
from nncase_simulator import KModelRuntime
from task_geometry import MEAN,STD,letterbox_bgr,decode


def compare_native_decoded(raw,captured,identity):
    expected=decode(raw,*identity['model_wh'],score_threshold=identity.get('score_threshold',.25),
                    nms_iou=identity.get('nms_iou',.5))
    differences=[];numeric=[]
    def visit(one,two,path):
        if isinstance(one,dict):
            if not isinstance(two,dict):differences.append(path+': object differs');return
            for key,value in one.items():
                if path=='root' and key=='scope':continue
                if key not in two:differences.append(path+'.'+key+': missing');continue
                visit(value,two[key],path+'.'+key)
        elif isinstance(one,list):
            if not isinstance(two,list) or len(one)!=len(two):differences.append(path+': list length/type differs');return
            for index,(a,b) in enumerate(zip(one,two)):visit(a,b,path+'['+str(index)+']')
        elif isinstance(one,float):
            if not isinstance(two,(int,float)) or isinstance(two,bool):differences.append(path+': number differs');return
            if not np.isfinite(two):differences.append(path+': nonfinite recorded number');return
            error=abs(one-two);numeric.append(error)
            if not np.isclose(one,two,rtol=1e-4,atol=1e-3):differences.append(path+': outside declared diagnostic tolerance')
        elif one!=two or type(one)!=type(two):differences.append(path+': categorical value/type differs')
    visit(expected,captured,'root')
    return {'status':'CAPTURED_NATIVE_DECODER_COMPARED_WITH_SAME_RAW_HOST_DECODER',
        'pass':not differences,'numeric_fields':len(numeric),'maximum_absolute_numeric_error':max(numeric,default=0.),
        'diagnostic_tolerance':{'rtol':1e-4,'atol':1e-3},'difference_count':len(differences),
        'first_differences':differences[:40]}


def load_pair(directory,record,identity,contract_sha256):
    directory=Path(directory)
    if (record['status']!='SAME_FRAME_INPUT_AND_KPU_RAW_SAVED' or record['performance_measurement'] is not False or
            record['model_sha256']!=identity['kmodel_sha256'] or
            record['reference_onnx_sha256']!=identity['onnx_sha256'] or
            record['output_contract_sha256']!=contract_sha256 or not record['input_frozen_before_AI2D'] or
            record['camera_id']!=2):
        raise ValueError('Board pair identity or same-frame input contract differs')
    expected={'source':('source.rgb888p',[1,3,360,640],'uint8','NCHW_RGB'),
              'prepared':('prepared.rgb888p',[1,3,192,320],'uint8','NCHW_RGB'),
              'raw':('kpu-raw.f32le',[1,1275,50],'float32_le','C')}
    values={}
    for key,(name,shape,dtype,layout) in expected.items():
        row=record['files'][key];path=directory/name
        size=int(np.prod(shape))*(4 if key=='raw' else 1)
        if (row['name']!=name or row['shape']!=shape or row['dtype']!=dtype or row['layout']!=layout or
                row['bytes']!=size or path.stat().st_size!=size or sha(path)!=row['sha256']):
            raise ValueError('Board tensor byte count/hash/shape/layout differs from the declared capture')
        values[key]=np.fromfile(path,dtype='<f4' if key=='raw' else np.uint8).reshape(shape)
    if not np.isfinite(values['raw']).all():raise ValueError('Nonfinite native KPU output')
    decoded=record['files']['decoded'];path=directory/'decoded.json'
    if decoded['name']!='decoded.json' or path.stat().st_size!=decoded['bytes'] or sha(path)!=decoded['sha256']:
        raise ValueError('Recorded board decoding changed')
    return values


def evaluate(samples,model_dir,output):
    samples,model_dir,output=map(lambda p:Path(p).resolve(),(samples,model_dir,output))
    if output.exists():raise FileExistsError('Preserve the original native pair comparison')
    state=json.loads((samples/'PAIR_CAPTURE_RESULT.json').read_text(encoding='utf8'))
    identity=json.loads((model_dir/'identity.json').read_text(encoding='utf8'))
    if (state['status']!='PAIRED_CAPTURE_COMPLETE' or not 1<=state['completed']<=3 or
            state['completed']!=state['requested'] or len(state['samples'])!=state['completed'] or
            identity['model_wh']!=[320,192] or identity['output_shape']!=[1,1275,50] or
            sha(model_dir/'topface.onnx')!=identity['onnx_sha256']):
        raise ValueError('Requires complete bounded real capture and matching selected models')
    runtime=session(model_dir/'topface.onnx');rows=[]
    work=output.parent/(output.stem+'-simulator-work')
    with KModelRuntime(model_dir/'topface.kmodel',model_dir/'identity.json',work) as simulator:
        for number,record in enumerate(state['samples'],1):
            if record['sample']!=number:raise ValueError('Native sample sequence differs')
            folder=samples/f'pair-{number:02d}'
            actual=json.loads((folder/'sample.json').read_text(encoding='utf8'))
            if actual!=record:raise ValueError('Native sample file differs from the completed capture index')
            arrays=load_pair(folder,record,identity,sha(model_dir/'output_contract.json'))
            native_decoded=json.loads((folder/'decoded.json').read_text(encoding='utf8'))
            prepared=arrays['prepared'];bgr=prepared[:,::-1,:,:].astype(np.float32)
            normalized=(bgr-MEAN[None,:,None,None])/STD[None,:,None,None]
            floating=runtime.run(None,{runtime.get_inputs()[0].name:normalized})[0][0]
            simulated=simulator.raw_from_rgb_nchw(prepared)
            actual=arrays['raw'][0]
            source_bgr=arrays['source'][0].transpose(1,2,0)[:,:,::-1].copy()
            host_canvas,_=letterbox_bgr(source_bgr,320,192)
            host_rgb=host_canvas[:,:,::-1].transpose(2,0,1)[None]
            pixel_error=np.abs(prepared.astype(np.int16)-host_rgb.astype(np.int16))
            quant_error=np.abs(actual-simulated);float_error=np.abs(actual-floating)
            rows.append({'sample':number,'runtime':record['runtime'],
                'capture_sequence_index':record['capture_sequence_index'],'source_files':record['files'],
                'captured_native_postprocess':compare_native_decoded(actual,native_decoded,identity),
                'actual_ai2d_vs_host_letterbox':{'mean_absolute_uint8':float(pixel_error.mean()),
                    'maximum_absolute_uint8':int(pixel_error.max()),'different_value_count':int((pixel_error>0).sum())},
                'simulator_vs_kpu_raw':{'allclose':bool(np.allclose(actual,simulated,rtol=1e-3,atol=1e-3)),
                    'diagnostic_tolerance':{'rtol':1e-3,'atol':1e-3},
                    'channel_mae':quant_error.mean(0).tolist(),'channel_max':quant_error.max(0).tolist(),
                    'decoded':compare_detections(simulated,actual,320,192)},
                'ort_vs_kpu_raw':{'channel_mae':float_error.mean(0).tolist(),'channel_max':float_error.max(0).tolist(),
                    'decoded':compare_detections(floating,actual,320,192)}})
    result={'status':'NATIVE_SAME_INPUT_KPU_NUMERICS_MEASURED','observed_utc':datetime.now(timezone.utc).isoformat(),
        'identity':identity,'samples':rows,'paired_capture_result_sha256':sha(samples/'PAIR_CAPTURE_RESULT.json'),
        'same_actual_prepared_rgb_input_for_ort_simulator_and_kpu':True,'native_fps':'NOT_MEASURED_IN_DISK_CAPTURE_MODE',
        'interpretation':'Raw/decoded numerical diagnostic only; no GT accuracy or automatic runtime compatibility acceptance.',
        'runtime_compatibility':'Review captured actual runtime against compiled identity before final acceptance.'}
    atomic_json(output,result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('samples','model-dir','output'):parser.add_argument('--'+key,type=Path,required=True)
    print(json.dumps(evaluate(**vars(parser.parse_args())),indent=2))
