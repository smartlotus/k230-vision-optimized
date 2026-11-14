"""Pretrained loading, original-ONNX equivalence, rectangular/head export.

Engineering feasibility only. No optimizer steps and no task accuracy claim.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from cloud_store import atomic_json, sha
from nanodet_topface import NanoDetTopFace,CHECKPOINT_SHA256,output_contract


def session(path):
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    return ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])


def compare(model,path,height,width,rtol=1e-3,atol=1e-3):
    runtime=session(path);rows=[]
    patterns=[np.zeros((1,3,height,width),dtype=np.float32),
        np.full((1,3,height,width),.5,dtype=np.float32),
        np.random.default_rng(20261114).normal(0,1,(1,3,height,width)).astype(np.float32)]
    for name,data in zip(('zero','mid','seeded_normal'),patterns):
        with torch.inference_mode():reference=model(torch.from_numpy(data)).numpy()
        actual=runtime.run(None,{runtime.get_inputs()[0].name:data})[0]
        row={'pattern':name,'shape':list(actual.shape),'max_abs_error':float(np.max(np.abs(actual-reference))),
             'mean_abs_error':float(np.mean(np.abs(actual-reference))),
             'allclose':bool(np.allclose(actual,reference,rtol=rtol,atol=atol))}
        rows.append(row)
    return {'rtol':rtol,'atol':atol,'pass':all(r['allclose'] for r in rows),'cases':rows}


def main(checkpoint,reference_onnx,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(20261114)
    result={'schema':'topface-nanodet-pretrained-export-probe-v1.1','status':'STARTED',
        'optimizer_updates':0,'task_training_started':False,'board_run':False,
        'source_checkpoint_sha256':sha(checkpoint),'source_checkpoint_bytes':Path(checkpoint).stat().st_size,
        'source_original_onnx_sha256':sha(reference_onnx),'torch':torch.__version__,
        'onnx':onnx.__version__,'onnxruntime':ort.__version__}
    try:
        original=NanoDetTopFace(task=False).eval()
        result['pretrained_loading']=original.initialize_pretrained(checkpoint,CHECKPOINT_SHA256)
        result['published_original_equivalence']=compare(original,reference_onnx,320,320)
        atomic_json(output/'RESULT.json',result)
        if not result['published_original_equivalence']['pass']:
            raise RuntimeError('Original pretrained adapter differs from published official ONNX')
        result['exports']=[]
        task=NanoDetTopFace(task=True).eval()
        result['task_initialization']=task.initialize_pretrained(checkpoint,CHECKPOINT_SHA256)
        for width,height in ((224,128),(320,192),(384,224)):
            for kind,model in (('original',original),('taskhead_untrained',task)):
                path=output/f'nanodet-{kind}-{width}x{height}.onnx'
                started=time.monotonic()
                torch.onnx.export(model,torch.zeros(1,3,height,width),path,input_names=['images'],
                    output_names=['topface_raw' if model.task else 'output'],opset_version=11,
                    do_constant_folding=True,dynamo=False)
                onnx.checker.check_model(onnx.load(path))
                paired=compare(model,path,height,width)
                result['exports'].append({'kind':kind,'width':width,'height':height,
                    'file':path.name,'bytes':path.stat().st_size,'sha256':sha(path),
                    'seconds':time.monotonic()-started,'pytorch_ort':paired})
                atomic_json(output/'RESULT.json',result)
                if not paired['pass']:raise RuntimeError('Exported rectangular PyTorch/ORT comparison failed')
            atomic_json(output/f'output_contract-{width}x{height}.json',output_contract(width,height))
        try:
            with torch.inference_mode():original(torch.zeros(1,3,180,320))
            result['strict_320x180']='FORWARD_EXECUTED_REQUIRES_SEPARATE_EXPORT_VALIDATION'
        except RuntimeError:
            result['strict_320x180']='REJECTED_FPN_SIZE_MISMATCH_USE_ASPECT_PRESERVING_320x192_LETTERBOX'
        result['status']='PRETRAINED_ORIGINAL_AND_RECTANGULAR_PT_ORT_PASS_TASK_HEAD_UNTRAINED'
    except Exception as error:
        result.update(status='FAILED',error_type=type(error).__name__,error_message=str(error)[:250])
        raise
    finally:atomic_json(output/'RESULT.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reference-onnx',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(**vars(p.parse_args()))
