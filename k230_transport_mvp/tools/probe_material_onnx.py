"""Seeded random YOLOv8n -> ONNX -> CPU numerical parity, NOT task training."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.material_model_contract import FIXTURE,model_metadata,inspect_onnx
from tools.material_preprocess import model_uint8,float_reference


def probe(output,settings,seed=20261109):
    output=Path(output)
    if output.exists():raise ValueError('Immutable probe directory; choose a new run path')
    output.mkdir(parents=True)
    settings=Path(settings).resolve();settings.mkdir(parents=True,exist_ok=True)
    os.environ['YOLO_CONFIG_DIR']=str(settings)
    import numpy as np
    import torch,onnx,onnxruntime as ort,ultralytics
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules import Detect,C2f
    if ultralytics.__version__!='8.3.235':raise ValueError('Pinned YOLO8.3.235 required')
    torch.set_num_threads(2);torch.manual_seed(seed)
    model=DetectionModel('yolov8n.yaml',nc=1,verbose=False).eval().float().fuse(verbose=False)
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module,Detect):
            module.dynamic=False;module.export=True;module.format='onnx';module.xyxy=False
        if isinstance(module,C2f):module.forward=module.forward_split
    weights=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        weights.update(name.encode());weights.update(str(tuple(tensor.shape)).encode())
        weights.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    rng=np.random.default_rng(seed)
    patterns={name:np.full((360,640,3),value,np.uint8) for name,value in (('zero',0),('mid',128),('max',255))}
    patterns['rgb_planes']=np.broadcast_to(np.array([32,128,224],np.uint8),(360,640,3)).copy()
    patterns['seeded_noise']=rng.integers(0,256,(360,640,3),dtype=np.uint8)
    rows=[]
    with torch.inference_mode():
        for size in (224,320):
            dummy=torch.zeros(1,3,size,size,dtype=torch.float32)
            for _ in range(2):model(dummy)
            path=output/('fixture-visible-top-%d.onnx'%size)
            begin=time.perf_counter()
            torch.onnx.export(model,dummy,str(path),opset_version=12,input_names=['images'],
                              output_names=['output0'],dynamic_axes=None,dynamo=False,
                              do_constant_folding=True)
            exported=onnx.load(str(path))
            for key,value in model_metadata(size,FIXTURE).items():
                entry=exported.metadata_props.add();entry.key=key;entry.value=value
            onnx.save(exported,str(path))
            contract=inspect_onnx(path,size,FIXTURE)
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            comparisons=[]
            for name,rgb in patterns.items():
                tensor,_=model_uint8(rgb,size);reference=float_reference(tensor)
                expected=model(torch.from_numpy(reference)).detach().cpu().numpy()
                actual=session.run(None,{'images':reference})[0]
                if actual.shape!=(1,5,contract['input_contract']['positions']) or not np.all(np.isfinite(actual)):
                    raise ValueError('Invalid decoded ONNX output')
                if np.any(actual[:,2:4,:]<0) or np.any(actual[:,4:,:]<0) or np.any(actual[:,4:,:]>1):
                    raise ValueError('Expected decoded nonnegative sizes and sigmoid scores')
                np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-4)
                comparisons.append(dict(pattern=name,input_uint8_sha256=hashlib.sha256(tensor.tobytes()).hexdigest(),
                    reference_float_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                    max_abs_error=float(np.max(np.abs(actual-expected))),mean_abs_error=float(np.mean(np.abs(actual-expected))),
                    output_shape=list(actual.shape),passed=True))
            rows.append(dict(size=size,onnx_file=path.name,inspection=contract,comparisons=comparisons,
                             export_and_check_seconds=time.perf_counter()-begin))
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),status='STRUCTURAL_EXPORT_NUMERIC_PASS_ONLY',
                seed=seed,artifact_kind=FIXTURE,weights_sha256=weights.hexdigest(),
                versions=dict(torch=torch.__version__,ultralytics=ultralytics.__version__,onnx=onnx.__version__,
                              onnxruntime=ort.__version__,numpy=np.__version__),models=rows,
                real_optimizer_updates=0,human_confirmed_gt_images=0,pretrained_downloaded=False,
                task_accuracy_verified=False,native_board_run=False,nncase_compile_executed=False,
                acceptance_limits=dict(rtol=1e-4,atol=1e-4),
                source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in (
                    'tools/probe_material_onnx.py','tools/material_model_contract.py','tools/material_preprocess.py',
                    'vision/yolo_candidates.py')})
    (output/'onnx-probe.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'models':[r['inspection']['output_shape'] for r in rows],
                      'real_optimizer_updates':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--settings',type=Path,required=True)
    args=parser.parse_args();probe(args.output,args.settings)
