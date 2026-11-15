"""Save paired ORT/KModel raw predictions for one frozen dataset split."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort

from cloud_store import atomic_json,sha
from nncase_simulator import KModelRuntime
from task_geometry import preprocess_bgr


def order_sha(rows):
    return hashlib.sha256(json.dumps([row['scene'] for row in rows],separators=(',',':')).encode('utf8')).hexdigest()


def run(dataset,model_dir,output,split='synthetic_test',max_seconds=18000,qualification_count=None,onnx=None):
    dataset,model_dir,output=dataset.resolve(),model_dir.resolve(),output.resolve()
    if ((qualification_count is None and split!='synthetic_test') or
            (qualification_count is not None and (split!='train' or not 1<=qualification_count<=6))):
        raise ValueError('Only the held-out test or a bounded train engineering fixture is admitted')
    if not 120<=max_seconds<=21600:
        raise ValueError('Saved-raw evaluation requires a finite120..21600second budget')
    output.mkdir(parents=True,exist_ok=False)
    index=json.loads((dataset/'AUDIT_V11.json').read_text(encoding='utf8'))
    identity=json.loads((model_dir/'identity.json').read_text(encoding='utf8'))
    onnx=model_dir/'topface.onnx' if onnx is None else onnx.resolve()
    if index['status']!='PASS' or sha(onnx)!=identity['onnx_sha256']:
        raise ValueError('Frozen model or dataset index differs')
    rows=[row for row in index['scenes'] if row['split']==split]
    if qualification_count is not None:
        rows=rows[:qualification_count]
    if not rows:
        raise ValueError('Selected simulation split is empty')
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    reference=ort.InferenceSession(str(onnx),sess_options=options,providers=['CPUExecutionProvider'])
    shape=(len(rows),*identity['output_shape'][1:])
    mapped={name:np.lib.format.open_memmap(output/(name+'-raw.npy'),mode='w+',dtype=np.float32,shape=shape)
            for name in ('ort','simulator')}
    state={'status':'STARTED','scope':'HELD_OUT_SYNTHETIC_TEST' if qualification_count is None else 'ENGINEERING_TRAIN_FIXTURE',
        'split':split,'images':len(rows),'completed_images':0,'scene_order_sha256':order_sha(rows),
        'dataset_index_sha256':sha(dataset/'AUDIT_V11.json'),'identity':identity,'raw_shape':list(shape),
        'optimizer_updates':0,'test_used_for_calibration':False}
    started=time.monotonic();deadline=started+max_seconds
    sum_error=np.zeros(shape[-1],np.float64);max_error=np.zeros(shape[-1],np.float64)
    try:
        with KModelRuntime(model_dir/'topface.kmodel',model_dir/'identity.json',output/'simulator-work') as simulator:
            for number,row in enumerate(rows):
                if time.monotonic()+20>=deadline:
                    raise TimeoutError('Finite saved-raw inference budget reached')
                folder=(dataset/row['scene']).resolve()
                if (not folder.is_relative_to(dataset) or sha(folder/'rgb.png')!=row['rgb_sha256'] or
                        sha(folder/'labels.json')!=row['label_sha256']):
                    raise ValueError('Evaluation image or label differs from its frozen index')
                label=json.loads((folder/'labels.json').read_text(encoding='utf8'))
                if label['split']!=split or label['scene_group_id']!=row['group']:
                    raise ValueError('Evaluation image is outside the selected independent split')
                image=cv2.imread(str(folder/'rgb.png'))
                tensor,_=preprocess_bgr(image,simulator.width,simulator.height)
                floating=reference.run(None,{reference.get_inputs()[0].name:tensor[None]})[0][0]
                quantized=simulator.raw_from_bgr(image)
                if floating.shape!=quantized.shape or not np.isfinite(floating).all():
                    raise ValueError('Paired raw output tensor contract differs')
                mapped['ort'][number]=floating;mapped['simulator'][number]=quantized
                error=np.abs(quantized-floating)
                sum_error+=error.sum(axis=0,dtype=np.float64);max_error=np.maximum(max_error,error.max(axis=0))
                state['completed_images']=number+1
                if (number+1)%50==0:
                    for value in mapped.values():value.flush()
                    atomic_json(output/'PROGRESS.json',state)
                    print('TOPFACE_SIMULATOR_TEST_PROGRESS '+json.dumps({'images':number+1,'total':len(rows),
                        'elapsed_seconds':time.monotonic()-started}),flush=True)
            state.update(status='NNCASE_AND_ORT_TEST_RAW_COMPLETE',simulator_load_seconds=simulator.load_seconds,
                simulator_inference_seconds={'n':len(simulator.inference_seconds),
                    'median':float(np.median(simulator.inference_seconds)),
                    'p95':float(np.quantile(simulator.inference_seconds,.95)),
                    'max':max(simulator.inference_seconds)},
                raw_channel_mae=(sum_error/(len(rows)*shape[1])).tolist(),raw_channel_max=max_error.tolist())
    except Exception as error:
        state.update(status='INFERENCE_FAILED_PRESERVE_RAW',error_type=type(error).__name__)
    finally:
        for value in mapped.values():value.flush()
        mapped.clear()
        state['elapsed_seconds']=time.monotonic()-started
        if state['status']=='NNCASE_AND_ORT_TEST_RAW_COMPLETE':
            state['files']=[{'name':name+'-raw.npy','bytes':(output/(name+'-raw.npy')).stat().st_size,
                             'sha256':sha(output/(name+'-raw.npy'))} for name in ('ort','simulator')]
        atomic_json(output/'RESULT.json',state)
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('dataset','model-dir','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--split',default='synthetic_test',choices=('train','synthetic_test'))
    parser.add_argument('--max-seconds',type=int,default=18000)
    parser.add_argument('--qualification-count',type=int)
    parser.add_argument('--onnx',type=Path)
    result=run(**vars(parser.parse_args()))
    print(json.dumps({k:result.get(k) for k in ('status','scope','images','completed_images','elapsed_seconds','simulator_inference_seconds')},indent=2))
    if result['status']!='NNCASE_AND_ORT_TEST_RAW_COMPLETE':raise SystemExit(1)
