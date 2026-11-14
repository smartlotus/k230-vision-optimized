"""Bounded original-COCO graph compatibility probe; never task training."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np


def sha_bytes(value):return hashlib.sha256(value).hexdigest()


def write_json(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8');temporary.replace(path)


def fixtures(shape):
    size=int(np.prod(shape));rng=np.random.default_rng(20261114)
    yield 'zero',np.zeros(shape,np.float32)
    yield 'mid',np.full(shape,.5,np.float32)
    yield 'ramp',np.linspace(0,1,size,dtype=np.float32).reshape(shape)
    indices=np.indices(shape[-2:]).sum(axis=0)%2
    yield 'checkerboard',np.broadcast_to(indices,shape).astype(np.float32).copy()
    yield 'seeded_noise',rng.uniform(0,1,shape).astype(np.float32)


def representative_train_images(root,shape,count=20):
    import cv2
    rows=[];data=[];root=Path(root).resolve()
    for path in sorted(root.glob('scene-*/labels.json')):
        label=json.loads(path.read_text(encoding='utf8'))
        if label['split']!='train':continue
        image_path=path.parent/'rgb.png';image=cv2.imread(str(image_path))
        if image is None or image.shape!=(360,640,3):raise ValueError('Expected qualified640x360 train image')
        height,width=shape[2:];gain=min(width/640,height/360)
        resized=(round(640*gain),round(360*gain));top=(height-resized[1])//2;left=(width-resized[0])//2
        canvas=np.full((height,width,3),114,np.uint8)
        canvas[top:top+resized[1],left:left+resized[0]]=cv2.resize(image,resized,interpolation=cv2.INTER_LINEAR)
        tensor=((canvas.astype(np.float32)-np.array([103.53,116.28,123.675],np.float32))/
                 np.array([57.375,57.12,58.395],np.float32)).transpose(2,0,1)[None].copy()
        data.append((path.parent.name,tensor))
        rows.append({'scene':path.parent.name,'split':'train','camera_group':label['camera_group_id'],
            'rgb_sha256':sha_bytes(image_path.read_bytes()),'label_sha256':sha_bytes(path.read_bytes())})
        if len(data)==count:break
    if len(data)<count:raise ValueError('Insufficient independently qualified train images for engineering PTQ')
    return data,rows


def run(model_path,output,compile_model=False,scope='original-coco',calibration_root=None):
    import onnx
    import onnxruntime as ort
    model_path=Path(model_path).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('Immutable probe output required')
    output.mkdir(parents=True)
    model_bytes=model_path.read_bytes();model=onnx.load_model_from_string(model_bytes)
    onnx.checker.check_model(model)
    report={'schema':'topface-g3-original-model-probe-v1','observed_utc':datetime.now(timezone.utc).isoformat(),
        'source_model':str(model_path),'source_bytes':len(model_bytes),'source_sha256':sha_bytes(model_bytes),
        'status':'ONNX_CHECKER_PASS','stage':'ORT','opset':[{'domain':x.domain,'version':x.version} for x in model.opset_import],
        'operator_counts':dict(Counter(x.op_type for x in model.graph.node)),
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','onnx','onnxruntime','nncase','nncase-kpu']},
        'optimizer_steps':0,'task_model':scope!='original-coco','board_run':False,'task_accuracy_verified':False,
        'scope':scope,'final_model_ptq':False}
    write_json(output/'RESULT.json',report)
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=2
    session=ort.InferenceSession(model_bytes,sess_options=options,providers=['CPUExecutionProvider'])
    inputs=session.get_inputs()
    if len(inputs)!=1 or inputs[0].type!='tensor(float)':raise ValueError('Expected one float32 image input')
    shape=list(inputs[0].shape)
    if any(not isinstance(x,int) or x<=0 for x in shape):raise ValueError('Static engineering probe input required')
    report['inputs']=[{'name':x.name,'shape':x.shape,'dtype':x.type} for x in inputs]
    report['outputs']=[{'name':x.name,'shape':x.shape,'dtype':x.type} for x in session.get_outputs()]
    cases=[];references=[];data=list(fixtures(shape));calibration=data
    if scope=='taskhead-untrained':
        if calibration_root is None:raise ValueError('Task-head engineering probe requires train-only source')
        representative,rows=representative_train_images(calibration_root,shape)
        calibration=representative[:16];data=representative[16:20]
        report['calibration_members']=rows[:16];report['paired_members']=rows[16:20]
    for name,tensor in data:
        started=time.perf_counter();values=session.run(None,{inputs[0].name:tensor})
        if not all(np.isfinite(value).all() for value in values):raise ValueError('ORT produced nonfinite output')
        references.append(values)
        cases.append({'pattern':name,'input_sha256':sha_bytes(tensor.tobytes()),
            'ort_seconds':time.perf_counter()-started,'output_bytes':sum(v.nbytes for v in values),
            'outputs':[{'shape':list(v.shape),'dtype':str(v.dtype),'min':float(v.min()),'max':float(v.max()),
                        'sha256':sha_bytes(v.tobytes())} for v in values]})
    report.update(status='ORT_CPU_PASS_ORIGINAL_COCO_GRAPH_ONLY',cases=cases,
        calibration_source=('SIXTEEN_QUALIFIED_SYNTHETIC_TRAIN_IMAGES_ENGINEERING_ONLY_NOT_FINAL_PTQ' if scope=='taskhead-untrained'
                            else 'FIVE_SYNTHETIC_ENGINEERING_FIXTURES_NOT_FINAL_MODEL_PTQ_DATA'))
    write_json(output/'RESULT.json',report)
    print(json.dumps({'model':model_path.name,'status':report['status'],'outputs':report['outputs']}),flush=True)
    if not compile_model:return report
    spec=importlib.util.find_spec('_nncase')
    if spec is None:raise ValueError('Missing existing nncase extension')
    plugin=str(Path(spec.origin).parent)
    os.environ['NNCASE_PLUGIN_PATH']=plugin;os.environ['PATH']=plugin+os.pathsep+os.environ.get('PATH','')
    dll=os.add_dll_directory(plugin) if os.name=='nt' else None
    try:
        import nncase
        os.chdir(output)
        settings=nncase.CompileOptions();settings.target='k230';settings.preprocess=False
        settings.dump_ir=True;settings.dump_asm=True;settings.dump_dir=str(output/'compiler-dump')
        settings.shape_bucket_enable=False
        compiler=nncase.Compiler(settings)
        report.update(stage='IMPORT_ONNX',status='COMPILER_STARTED')
        write_json(output/'RESULT.json',report)
        compiler.import_onnx(model_bytes,nncase.ImportOptions())
        ptq=nncase.PTQTensorOptions();ptq.samples_count=len(calibration)
        ptq.calibrate_method='NoClip';ptq.quant_type='uint8';ptq.w_quant_type='uint8'
        ptq.set_tensor_data([[tensor] for _,tensor in calibration]);compiler.use_ptq(ptq)
        report['stage']='COMPILE';write_json(output/'RESULT.json',report)
        started=time.perf_counter();compiler.compile();result=compiler.gencode_tobytes()
        (output/('original-coco.kmodel' if scope=='original-coco' else 'untrained-taskhead.kmodel')).write_bytes(result)
        report.update(status='K230_COMPILED_ORIGINAL_COCO_ONLY',stage='SIMULATOR',
            compile_seconds=time.perf_counter()-started,kmodel_bytes=len(result),kmodel_sha256=sha_bytes(result))
        write_json(output/'RESULT.json',report)
        simulator=nncase.Simulator();simulator.load_model(result)
        if simulator.inputs_size!=1:raise ValueError('Unexpected simulator input count')
        report['simulator_inputs']=[list(simulator.get_input_shape(i)) for i in range(simulator.inputs_size)]
        comparisons=[]
        for (name,tensor),reference in zip(data[:2],references[:2]):
            started=time.perf_counter();simulator.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(tensor))
            simulator.run();values=[simulator.get_output_tensor(i).to_numpy().copy() for i in range(simulator.outputs_size)]
            if len(values)!=len(reference):raise ValueError('Simulator output count differs from ORT')
            metrics=[]
            for actual,expected in zip(values,reference):
                if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError('Invalid simulator tensor')
                error=np.abs(actual.astype(np.float32)-expected)
                metrics.append({'shape':list(actual.shape),'max_abs_error':float(error.max()),
                    'mean_abs_error':float(error.mean()),'relative_l2_error':float(np.linalg.norm(error)/max(np.linalg.norm(expected),1e-12))})
            comparisons.append({'pattern':name,'seconds':time.perf_counter()-started,'outputs':metrics})
        report.update(status=('COMPILE_AND_SIMULATOR_EXECUTED_ORIGINAL_COCO_ONLY' if scope=='original-coco'
                              else 'COMPILE_AND_SIMULATOR_EXECUTED_UNTRAINED_TASK_HEAD'),stage='DONE',simulator_comparisons=comparisons,
            numerical_acceptance='MEASURED_ONLY_NO_NEW_TASK_TOLERANCE_REGISTERED',kpu_partition_acceptance=False)
    except Exception as error:
        report.update(status='COMPILE_OR_SIMULATOR_FAILED',error_type=type(error).__name__,error=str(error)[:1500])
    finally:
        write_json(output/'RESULT.json',report)
        if dll:dll.close()
    print(json.dumps({k:report.get(k) for k in ['status','stage','kmodel_bytes','compile_seconds','error']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--compile',action='store_true')
    parser.add_argument('--scope',choices=['original-coco','taskhead-untrained'],default='original-coco')
    parser.add_argument('--calibration-root',type=Path)
    args=parser.parse_args();run(args.model,args.output,args.compile,args.scope,args.calibration_root)
