"""Host-only ONNX / compiler input contracts; no capture or training side effects."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from tools.material_dataset import audit,verify_calibration,source_file,canonical
from tools.file_integrity import sha256_file
from vision.yolo_candidates import input_contract

FIXTURE = 'RANDOM_WEIGHTS_STRUCTURAL_FIXTURE'
TRAINED = 'TRAINED_MATERIAL_VISIBLE_TOP'
ENCODING = 'DECODED_CXCYWH_AND_SIGMOID'


def model_metadata(size,artifact_kind):
    if artifact_kind not in (FIXTURE,TRAINED):raise ValueError('Explicit model provenance required')
    return {'material_artifact_kind':artifact_kind,'material_classes':json.dumps(['visible_top']),
            'material_output_encoding':ENCODING,'material_onnx_normalization':'EXTERNAL_FLOAT32_DIV255_ONCE',
            'material_input_contract':json.dumps(input_contract(size),sort_keys=True,separators=(',',':'))}


def inspect_onnx(path,size,expected_kind=TRAINED):
    import onnx
    path=Path(path)
    if not path.is_file() or not 0<path.stat().st_size<=64*1024*1024:
        raise ValueError('Bounded ONNX file required')
    if expected_kind not in (FIXTURE,TRAINED):raise ValueError('Explicit expected provenance required')
    model=onnx.load_model(str(path),load_external_data=False)
    expected=model_metadata(size,expected_kind)
    metadata={item.key:item.value for item in model.metadata_props}
    if any(metadata.get(k)!=v for k,v in expected.items()):raise ValueError('Material ONNX provenance/contract metadata mismatch')
    if len(metadata)!=len(model.metadata_props):raise ValueError('Duplicate ONNX metadata key')
    opsets={item.domain:item.version for item in model.opset_import}
    if opsets!={'':12} or len(model.opset_import)!=1:raise ValueError('Fixed standard opset12 required')

    def walk(graph):
        def tensor_storage(tensor):
            if tensor.data_location==onnx.TensorProto.EXTERNAL or tensor.external_data:
                raise ValueError('External ONNX tensor storage is not portable')
            if tensor.data_type in (onnx.TensorProto.FLOAT,onnx.TensorProto.DOUBLE,onnx.TensorProto.FLOAT16):
                if not np.all(np.isfinite(onnx.numpy_helper.to_array(tensor))):
                    raise ValueError('Nonfinite ONNX weights/constants')
        for tensor in graph.initializer:tensor_storage(tensor)
        for tensor in graph.sparse_initializer:
            tensor_storage(tensor.values);tensor_storage(tensor.indices)
        for node in graph.node:
            if node.domain not in ('','ai.onnx') or node.op_type in ('NonMaxSuppression','If','Loop','Scan'):
                raise ValueError('No custom, NMS or dynamic control-flow export')
            for attribute in node.attribute:
                if attribute.type==onnx.AttributeProto.GRAPH:walk(attribute.g)
                elif attribute.type==onnx.AttributeProto.GRAPHS:
                    for child in attribute.graphs:walk(child)
                elif attribute.type==onnx.AttributeProto.TENSOR:tensor_storage(attribute.t)
                elif attribute.type==onnx.AttributeProto.TENSORS:
                    for tensor in attribute.tensors:tensor_storage(tensor)
    walk(model.graph)
    try:onnx.checker.check_model(model,full_check=True)
    except (onnx.checker.ValidationError,onnx.shape_inference.InferenceError) as error:
        raise ValueError('Invalid ONNX graph or inferred shapes') from error
    if len(model.graph.input)!=1 or len(model.graph.output)!=1:
        raise ValueError('Exactly one input/output required')

    def describe(value):
        tensor=value.type.tensor_type
        if tensor.elem_type!=onnx.TensorProto.FLOAT:raise ValueError('ONNX interface must be float32')
        dims=[]
        for dim in tensor.shape.dim:
            if dim.HasField('dim_param') or not dim.HasField('dim_value') or dim.dim_value<=0:
                raise ValueError('Static positive dimensions required')
            dims.append(dim.dim_value)
        return dims
    contract=input_contract(size)
    if describe(model.graph.input[0])!=[1,3,size,size] or describe(model.graph.output[0])!=[1,5,contract['positions']]:
        raise ValueError('Material source/output dimensions mismatch')
    return dict(model_sha256=sha256_file(path),bytes=path.stat().st_size,
                artifact_kind=expected_kind,input_name=model.graph.input[0].name,
                output_name=model.graph.output[0].name,input_shape=[1,3,size,size],
                output_shape=[1,5,contract['positions']],opset=12,input_contract=contract,
                metadata=metadata,node_count=len(model.graph.node),
                task_accuracy_verified=False,native_ai2d_equivalence_verified=False)


def compiler_settings(size):
    input_contract(size)
    return dict(target='k230',preprocess=True,swapRB=False,input_shape=[1,3,size,size],
                input_type='uint8',input_range=[0,1],mean=[0,0,0],std=[1,1,1],input_layout='NCHW',
                output_layout='',dump_asm=False,dump_ir=False,shape_bucket_enable=False)


def require_nncase_version():
    versions={key:importlib.metadata.version(key) for key in ('nncase','nncase-kpu')}
    if set(versions.values())!={'2.11.0'}:raise ValueError('Compiler and KPU plugin must both match2.11.0')
    return versions


def load_reviewed_calibration(root,manifest,prepared,size):
    """Load immutable prepared uint8 tensors; never infer truth from predictions."""
    report=audit(root,manifest)
    if not report['export_ready']:raise ValueError('Reviewed grouped train/val/test data required')
    verify_calibration(root,manifest,prepared)
    prepared=Path(prepared)
    receipt=json.loads((prepared/'calibration-manifest.json').read_text(encoding='utf-8'))
    if receipt['input_contract']!=input_contract(size):raise ValueError('ONNX and calibration sizes/padding differ')
    arrays=[]
    for row in receipt['samples']:
        raw=source_file(prepared,row['file']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('Calibration bytes changed after verification')
        shape=row['shape']
        if row['dtype']!='uint8' or shape!=[1,3,size,size] or len(raw)!=3*size*size:
            raise ValueError('Exact uint8 NCHW calibration tensor required')
        arrays.append([np.frombuffer(raw,dtype=np.uint8).copy().reshape(shape)])
    return arrays,receipt


def verify_training_provenance(root,manifest,model,training_run):
    """Verify local training/export evidence, not self-declared ONNX labels alone."""
    from tools.train_material_pilot import verify_training_run,sha
    if training_run is None:raise ValueError('Completed training run required for task compilation')
    run=Path(training_run).resolve()
    receipt_path=Path(model).parent/'export.json'
    exported=json.loads(receipt_path.read_text(encoding='utf-8'))
    if exported.get('status')!='EXPORTED_NUMERIC_PASS_NOT_TASK_ACCEPTED' or exported.get('onnx_sha256')!=sha(model):
        raise ValueError('Successful hash-bound task export receipt required')
    comparisons=exported.get('comparisons')
    if (not isinstance(comparisons,list) or len(comparisons)!=5 or
            any(not isinstance(row,dict) or row.get('passed') is not True for row in comparisons) or
            exported.get('numeric_limits')!={'rtol':1e-4,'atol':1e-4}):
        raise ValueError('Missing task export numerical comparison evidence')
    if {row.get('pattern') for row in comparisons}!={'zero','mid','max','rgb_planes','noise'}:
        raise ValueError('Task export numerical cases missing or duplicated')
    training,weights,_=verify_training_run(root,run,exported.get('checkpoint'))
    expected_dataset=hashlib.sha256(canonical(manifest)).hexdigest()
    if training['dataset_manifest_sha256']!=expected_dataset:raise ValueError('Training and conversion dataset identity mismatch')
    expected=dict(material_training_receipt_sha256=sha(run/'training.json'),
                  material_dataset_manifest_sha256=expected_dataset,
                  material_trained_checkpoint_sha256=sha(weights))
    import onnx
    graph=onnx.load(str(model),load_external_data=False)
    metadata={p.key:p.value for p in graph.metadata_props}
    if any(metadata.get(k)!=v for k,v in expected.items()):raise ValueError('ONNX training provenance metadata mismatch')
    if (exported.get('training_receipt_sha256')!=expected['material_training_receipt_sha256'] or
            exported.get('trained_checkpoint_sha256')!=expected['material_trained_checkpoint_sha256'] or
            exported.get('training_dataset_manifest_sha256')!=expected_dataset or
            exported.get('real_optimizer_updates')!=training['real_optimizer_updates']):
        raise ValueError('Task export/training receipt mismatch')
    return dict(expected,export_receipt_sha256=sha(receipt_path),real_optimizer_updates=training['real_optimizer_updates'])
