"""Train-only A16 PTQ of a saved CameraXY temporal model with uint8 input.

Normalization is inside the exported graph, so K230 never has to build a
float32 RGB history on the CPU. Native camera/AI2D equivalence is separate.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from dataset import audit,digest
from camera_preprocess import rgb_uint8
from map_models import MapCameraXYNet


class Uint8Input(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,rgb_window_uint8):return self.model(rgb_window_uint8.float()/255.)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--parent',type=Path,required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--limit',type=int)
    ap.add_argument('--split',choices=('val','test'),default='test')
    ap.add_argument('--activation',choices=('int16','uint8'),default='int16')
    args=ap.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source=args.model.resolve();data=args.data.resolve()
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    saved=torch.load(source/'best.pt',map_location='cpu',weights_only=True)
    if saved['parent_sha256']!=digest(args.parent):raise ValueError('Parent lineage differs')
    base=MapCameraXYNet(parent,saved['route'],saved['window'],saved.get('single_frame_adjustment',False)).eval()
    base.load_state_dict(saved['state'])
    model=Uint8Input(base).eval();torch.set_num_threads(2)
    rows=[json.loads(x) for x in (data/'manifest.jsonl').read_text().splitlines()];audit(rows,data)
    if digest(data/'manifest.jsonl')!=saved['dataset_manifest_sha256']:raise ValueError('Dataset identity differs')
    groups={}
    for r in rows:groups.setdefault(r['trajectory_id'],[]).append(r)
    windows={}
    for frames in groups.values():
        for i,r in enumerate(frames):
            clip=frames[max(0,i-saved['window']+1):i+1]
            windows[r['sample_id']]=[clip[0]]*(saved['window']-len(clip))+clip
    def tensor(row):
        values=[rgb_uint8(data/r['image_path']) for r in windows[row['sample_id']]]
        return np.concatenate(values,axis=0)[None].copy()
    shape=[1,3*saved['window'],96,160];onnx=out/'camera_xy_uint8.onnx'
    torch.onnx.export(model,torch.zeros(shape,dtype=torch.uint8),str(onnx),input_names=['rgb_window_uint8'],
                      output_names=['location_logits'],opset_version=13,dynamo=False)
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    runtime=ort.InferenceSession(str(onnx),sess_options=options,providers=['CPUExecutionProvider'])
    original=ort.InferenceSession(str(source/'camera_xy.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    train=[r for r in rows if r['split']=='train'];calibration=train[::max(1,len(train)//100)][:100]
    for r in calibration[:3]:
        x=tensor(r);a=runtime.run(None,{'rgb_window_uint8':x})[0]
        b=original.run(None,{'rgb_window':x.astype(np.float32)/255})[0]
        np.testing.assert_allclose(a,b,atol=1e-4,rtol=1e-3)
    plugin=str(Path(importlib.util.find_spec('_nncase').origin).parent)
    os.environ['NNCASE_PLUGIN_PATH']=plugin;os.environ['PATH']=plugin+os.pathsep+os.environ.get('PATH','')
    dll=os.add_dll_directory(plugin) if os.name=='nt' else None
    import nncase
    versions={p:importlib.metadata.version(p) for p in ('nncase','nncase-kpu')}
    if set(versions.values())!={'2.11.0'}:raise ValueError('Compiler pin differs')
    os.chdir(out);opts=nncase.CompileOptions();opts.target='k230';opts.preprocess=False
    opts.input_type='uint8';opts.input_shape=shape;opts.shape_bucket_enable=False;opts.dump_ir=False;opts.dump_asm=False
    opts.dump_dir=str(out/'dump');compiler=nncase.Compiler(opts)
    compiler.import_onnx(onnx.read_bytes(),nncase.ImportOptions())
    ptq=nncase.PTQTensorOptions();ptq.samples_count=len(calibration);ptq.calibrate_method='NoClip'
    ptq.quant_type=args.activation;ptq.w_quant_type='uint8';ptq.set_tensor_data([[tensor(r)] for r in calibration])
    compiler.use_ptq(ptq);compiler.compile();binary=compiler.gencode_tobytes();(out/'camera_xy.kmodel').write_bytes(binary)
    sim=nncase.Simulator();sim.load_model(binary);centers=np.asarray(parent['centers_mm'])
    test=[r for r in rows if r['split']==args.split];test=test if args.limit is None else test[:args.limit]
    comparisons=[];all_scores=[]
    for i,r in enumerate(test):
        x=tensor(r);ref=runtime.run(None,{'rgb_window_uint8':x})[0]
        sim.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(x));start=time.perf_counter();sim.run();elapsed=time.perf_counter()-start
        actual=sim.get_output_tensor(0).to_numpy().copy()
        if actual.shape!=ref.shape or not np.isfinite(actual).all():raise ValueError('Invalid Simulator output')
        gt=np.array([r['camera_xy_target'][k] for k in ('camera_x_mm','camera_y_mm')])
        record={'sample_id':r['sample_id'],'anchor_same':int(ref.argmax())==int(actual.argmax()),
            'onnx_error_mm':float(np.linalg.norm(centers[int(ref.argmax())]-gt)),
            'kmodel_error_mm':float(np.linalg.norm(centers[int(actual.argmax())]-gt)),
            'mae':float(abs(actual-ref).mean()),'max_abs':float(abs(actual-ref).max()),'host_simulator_seconds':elapsed}
        comparisons.append(record);all_scores.append(actual[0])
        if (i+1)%25==0:print(json.dumps({'evaluated':i+1,'planned':len(test)}),flush=True)
    np.savez_compressed(out/'SIMULATOR_SCORES.npz',sample_ids=np.array([r['sample_id'] for r in test]),logits=np.stack(all_scores))
    report={'status':'COMPILE_AND_SIMULATOR_PASS','qualification':'EXPERIMENTAL_NOT_NATIVE_ACCEPTANCE',
        'route':saved['route'],'window':saved['window'],'checkpoint_sha256':digest(source/'best.pt'),
        'parent_sha256':saved['parent_sha256'],'float_onnx_sha256':digest(source/'camera_xy.onnx'),
        'uint8_onnx_sha256':digest(onnx),'kmodel_sha256':digest(out/'camera_xy.kmodel'),'kmodel_bytes':len(binary),
        'compiler':versions,'input_shape':shape,'input_type':'uint8','input_order':'past-to-current RGB NCHW packed channels',
        'normalization':'Cast + divide255 ONCE inside graph; compiler preprocess=False',
        'float_uint8_export_parity':'PASS on3 train windows','ptq':{'calibration_split':'train ONLY','samples':len(calibration),
            'activation':args.activation,'weights':'uint8','method':'NoClip','ids':[r['sample_id'] for r in calibration]},
        'dataset_manifest_sha256':digest(data/'manifest.jsonl'),'evaluation_split':args.split,
        'evaluated_frames':len(test),'full_evaluation_split_frames':sum(r['split']==args.split for r in rows),
        'anchor_agreement':sum(r['anchor_same'] for r in comparisons)/len(comparisons),'pairs':comparisons,
        'native_k230':'NOT RUN','native_ai2d_pixel_equivalence':'NOT RUN','real_confidence':'UNKNOWN'}
    (out/'KMODEL_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','ptq')}),flush=True)


if __name__=='__main__':main()
