"""Export only a hash-bound completed material pilot checkpoint, never auto-train."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.train_material_pilot import verify_training_run,sha,write_receipt
from tools.material_model_contract import TRAINED,model_metadata,inspect_onnx
from tools.material_preprocess import model_uint8,float_reference
from vision.yolo_candidates import input_contract


def export(root,run,output,size=320,checkpoint='best.pt'):
    root=Path(root).resolve();run=Path(run).resolve();output=Path(output).resolve()
    training,weights,manifest=verify_training_run(root,run,checkpoint)
    input_contract(size)
    receipt_sha=sha(run/'training.json')
    if output.exists():raise ValueError('New immutable ONNX export directory required')
    output.mkdir(parents=True)
    result=dict(schema_version=1,status='EXPORT_STARTED',started_at=datetime.now(timezone.utc).isoformat(),
        training_receipt_sha256=receipt_sha,training_dataset_manifest_sha256=training['dataset_manifest_sha256'],
        trained_checkpoint_sha256=sha(weights),real_optimizer_updates=training['real_optimizer_updates'],
        exported_size=size,checkpoint=checkpoint,task_accuracy_verified=False,native_ai2d_equivalence_verified=False,
        numeric_limits=dict(rtol=1e-4,atol=1e-4),
        numeric_reference=dict(torch_mkldnn=False,ort_graph_optimization='DISABLED',dtype='float32'),
        sample_scope='Synthetic numeric parity only; held-out test images not consumed')
    result_path=output/'export.json';write_receipt(result_path,result)
    old_setting=os.environ.get('YOLO_CONFIG_DIR');os.environ['YOLO_CONFIG_DIR']=str(output/'settings')
    old_mkldnn=None
    try:
        import numpy as np
        import torch,onnx,onnxruntime as ort,ultralytics
        from ultralytics.nn.modules import Detect,C2f
        from ultralytics.nn.tasks import load_checkpoint
        if ultralytics.__version__!='8.3.235':raise ValueError('Pinned YOLO8.3.235 required')
        # Fixed CPU float32 reference avoids backend-fusion rounding differences
        # near zero decoded coordinates. Keep the original elementwise limits;
        # this is a parity check, not an optimized CPU performance benchmark.
        old_mkldnn=torch.backends.mkldnn.enabled;torch.backends.mkldnn.enabled=False
        torch.set_num_threads(2)
        loaded,_=load_checkpoint(str(weights))
        if loaded.names!={0:'visible_top'}:raise ValueError('Checkpoint must be the single material class')
        model=deepcopy(loaded).cpu().float().eval().fuse(verbose=False);model.requires_grad_(False)
        for module in model.modules():
            if isinstance(module,Detect):
                module.export=True;module.dynamic=False;module.format='onnx';module.xyxy=False
            if isinstance(module,C2f):module.forward=module.forward_split
        with torch.inference_mode():
            dummy=torch.zeros(1,3,size,size)
            for _ in range(2):model(dummy)
            path=output/'material.onnx'
            torch.onnx.export(model,dummy,str(path),opset_version=12,input_names=['images'],
                              output_names=['output0'],dynamic_axes=None,dynamo=False,do_constant_folding=True)
            graph=onnx.load(str(path))
            metadata=dict(model_metadata(size,TRAINED),
                material_training_receipt_sha256=receipt_sha,
                material_dataset_manifest_sha256=training['dataset_manifest_sha256'],
                material_trained_checkpoint_sha256=sha(weights))
            for key,value in metadata.items():
                item=graph.metadata_props.add();item.key=key;item.value=value
            onnx.save(graph,str(path))
            inspection=inspect_onnx(path,size)
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            cases={name:np.full((360,640,3),value,np.uint8) for name,value in (('zero',0),('mid',128),('max',255))}
            cases['rgb_planes']=np.broadcast_to(np.array([32,128,224],np.uint8),(360,640,3)).copy()
            cases['noise']=np.random.default_rng(training['seed']).integers(0,256,(360,640,3),dtype=np.uint8)
            comparisons=[]
            for name,rgb in cases.items():
                tensor,_=model_uint8(rgb,size);input_float=float_reference(tensor)
                expected=model(torch.from_numpy(input_float)).detach().cpu().numpy()
                actual=session.run(None,{'images':input_float})[0]
                if not np.all(np.isfinite(actual)):raise ValueError('Nonfinite ONNX output')
                np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4)
                comparisons.append(dict(pattern=name,max_abs_error=float(np.max(np.abs(actual-expected))),passed=True))
        confirmed,confirmed_weight,_=verify_training_run(root,run,checkpoint)
        if sha(run/'training.json')!=receipt_sha or sha(confirmed_weight)!=result['trained_checkpoint_sha256']:
            raise ValueError('Training provenance changed during ONNX export')
        result.update(status='EXPORTED_NUMERIC_PASS_NOT_TASK_ACCEPTED',inspection=inspection,
                      comparisons=comparisons,onnx_sha256=sha(path),finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        result.update(status='FAILED_OR_UNKNOWN_NOT_CONVERTIBLE',error=str(error))
        raise
    finally:
        try:write_receipt(result_path,result)
        finally:
            if old_mkldnn is not None:torch.backends.mkldnn.enabled=old_mkldnn
            if old_setting is None:os.environ.pop('YOLO_CONFIG_DIR',None)
            else:os.environ['YOLO_CONFIG_DIR']=old_setting
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--size',type=int,choices=(224,320),default=320);p.add_argument('--checkpoint',choices=('best.pt','last.pt'),default='best.pt')
    a=p.parse_args();print(json.dumps(export(a.root,a.run,a.output,a.size,a.checkpoint),indent=2))
