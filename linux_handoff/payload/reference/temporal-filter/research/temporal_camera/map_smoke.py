"""Exercise actual pretrained RGB/map graph and nonzero temporal adapters."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import onnx
import onnxruntime as ort

from dataset import digest
from map_models import MapCameraXYNet


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--competition',action='store_true')
    args=ap.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    parent=torch.load(args.parent,map_location='cpu',weights_only=True);torch.set_num_threads(2)
    reports=[]
    for route,t in [('cnn',1),('gru',4),('tcn',4),('gru',8),('tcn',8)]:
        torch.manual_seed(117);name=f'{route}{t}';model=MapCameraXYNet(parent,route,t,args.competition).eval()
        if route!='cnn' or args.competition:
            # Zero residuals would allow a compiler to erase the temporal branch.
            torch.nn.init.normal_(model.correction.weight,std=.01)
        x=torch.rand(1,3*t,96,160);path=out/(name+'.onnx')
        torch.onnx.export(model,x,str(path),input_names=['rgb_window'],output_names=['location_logits'],opset_version=13,dynamo=False)
        onnx.checker.check_model(onnx.load(path))
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        with torch.inference_mode():expected=model(x).numpy()
        actual=session.run(None,{'rgb_window':x.numpy()})[0]
        np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-3)
        report={'scope':'PRETRAINED_MAP_WITH_NONZERO_TEMPORAL_OPERATOR_PROBE','status':'ONNX_ORT_PASS',
            'parent_sha256':digest(args.parent),'input_shape':list(x.shape),'output_shape':list(actual.shape),
            'onnx_sha256':digest(path),'torch_ort_max_abs':float(abs(actual-expected).max()),
            'parameters':sum(p.numel() for p in model.parameters()),'map_reference_bytes':model.reference.numel()*4,
            'opset':13,'route':route,'window':t,'operator_set':sorted({n.op_type for n in onnx.load(path).graph.node})}
        (out/(name+'-onnx.json')).write_text(json.dumps(report,indent=2)+'\n')
        with (out/(name+'-compile.log')).open('w') as log:
            try:
                result=subprocess.run([sys.executable,str(Path(__file__).with_name('smoke.py')),'--out',str(out),
                    '--compile-one',name],stdout=log,stderr=subprocess.STDOUT,timeout=300)
                report['compile_exit_code']=result.returncode
            except subprocess.TimeoutExpired:report['compile_exit_code']='TIMEOUT_300s'
        reports.append(report);(out/'SUMMARY.json').write_text(json.dumps(reports,indent=2)+'\n')
        print(json.dumps({'route':name,'onnx':'PASS','compile_exit_code':report['compile_exit_code']}),flush=True)


if __name__=='__main__':main()
