"""Compile the trained 50-channel TopFaceNet with audited train-only PTQ."""
import argparse
import importlib.metadata
import importlib.util
import json
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from cloud_store import atomic_json, sha
from task_geometry import MEAN, STD, box_iou, decode, letterbox_bgr


def load_samples(dataset, width, height, calibration_count, pair_count,backgrounds=None):
    dataset = dataset.resolve()
    index = json.loads((dataset/'AUDIT_V11.json').read_text(encoding='utf8'))
    if index['status'] != 'PASS':
        raise ValueError('PTQ requires an audited training dataset')
    train = sorted((row for row in index['scenes'] if row['split'] == 'train'),
                   key=lambda row: (row['camera_group'], row['scene']))
    count = calibration_count + pair_count
    if len(train) < count:
        raise ValueError('Insufficient audited train scenes for calibration and paired verification')
    indices = np.linspace(0, len(train)-1, count, dtype=int)
    paired_indices = {int(index) for index in np.linspace(0, count-1, pair_count, dtype=int)}
    specimens = []
    for specimen_index,index in enumerate(indices):
        row = train[index]
        folder = (dataset/row['scene']).resolve()
        if not folder.is_relative_to(dataset) or sha(folder/'rgb.png') != row['rgb_sha256']:
            raise ValueError('Train RGB differs from the audit')
        if sha(folder/'labels.json') != row['label_sha256']:
            raise ValueError('Train labels differ from the audit')
        label = json.loads((folder/'labels.json').read_text(encoding='utf8'))
        if label['split'] != 'train' or label['scene_group_id'] != row['group']:
            raise ValueError('PTQ scene identity differs')
        image=cv2.imread(str(folder/'rgb.png'))
        operations=[]
        if backgrounds is not None and specimen_index not in paired_indices:
            image,operations=backgrounds.apply(image,label['instances'],
                np.random.default_rng(np.random.SeedSequence([20261114,int(index),0xCA1])))
        canvas, _ = letterbox_bgr(image, width, height)
        bgr = ((canvas.astype(np.float32)-MEAN)/STD).transpose(2,0,1)[None].copy()
        rgb = canvas[:,:,::-1].transpose(2,0,1)[None].copy()
        traced={**row,'compiler_rgb_tensor_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}
        if backgrounds is not None and specimen_index not in paired_indices:
            traced['training_background_augmentation']={'bank_manifest_sha256':backgrounds.identity['manifest_sha256'],
                'seed_components':[20261114,int(index),0xCA1],'paste_boxes_xyxy':operations}
        specimens.append((traced, rgb, bgr))
    return ([value for i,value in enumerate(specimens) if i not in paired_indices],
            [value for i,value in enumerate(specimens) if i in paired_indices])


def compare_detections(reference, actual, width, height):
    reference_result = decode(reference,width,height)
    simulator_result = decode(actual,width,height)
    expected = reference_result['detections']
    observed = simulator_result['detections']
    remaining = set(range(len(observed)))
    matches = []
    matched_reference = set()
    for expected_index, target in enumerate(expected):
        indices = sorted(remaining)
        if not indices:
            break
        overlaps = box_iou(np.asarray(target['bbox_xyxy_visible']),
                           np.asarray([observed[i]['bbox_xyxy_visible'] for i in indices]))
        position = int(np.argmax(overlaps))
        if overlaps[position] < .5:
            continue
        index = indices[position]
        remaining.remove(index)
        matched_reference.add(expected_index)
        pair = observed[index]
        ellipse_a, ellipse_b = target['ellipse'], pair['ellipse']
        row = {'ort_index':expected_index, 'simulator_index':index,
               'bbox_iou':float(overlaps[position]), 'same_color':target['color_id']==pair['color_id'],
               'score_abs':abs(target['score']-pair['score']),
               'ort_ellipse_finite':ellipse_a is not None, 'simulator_ellipse_finite':ellipse_b is not None,
               'ort_angle_valid':ellipse_a is not None and ellipse_a['orientation_valid'],
               'simulator_angle_valid':ellipse_b is not None and ellipse_b['orientation_valid'],
               'physical_center_px':float(np.linalg.norm(np.asarray(target['center_xy_projected_3d'])-
                                                            pair['center_xy_projected_3d']))}
        if ellipse_a is not None and ellipse_b is not None:
            row['conic_center_px'] = float(np.linalg.norm(np.asarray(ellipse_a['conic_center_xy'])-
                                                        ellipse_b['conic_center_xy']))
            row['axis_relative'] = [abs(ellipse_b[key]-ellipse_a[key])/ellipse_a[key]
                                    for key in ('semi_major','semi_minor')]
            if ellipse_a['orientation_valid'] and ellipse_b['orientation_valid']:
                row['angle_error_deg'] = abs((ellipse_a['orientation_deg']-ellipse_b['orientation_deg']+90)%180-90)
        matches.append(row)
    return {'ort_count':len(expected),'simulator_count':len(observed),'matches':matches,
            'matched_count':len(matches), 'unmatched_ort_indices':sorted(set(range(len(expected)))-matched_reference),
            'unmatched_simulator_indices':sorted(remaining),
            'geometry_transitions':{
                'finite_to_missing':sum(row['ort_ellipse_finite'] and not row['simulator_ellipse_finite'] for row in matches),
                'missing_to_finite':sum(not row['ort_ellipse_finite'] and row['simulator_ellipse_finite'] for row in matches),
                'angle_valid_to_invalid':sum(row['ort_angle_valid'] and not row['simulator_angle_valid'] for row in matches),
                'angle_invalid_to_valid':sum(not row['ort_angle_valid'] and row['simulator_angle_valid'] for row in matches)},
            'ort_capacity':reference_result['capacity_state'],
            'simulator_capacity':simulator_result['capacity_state']}


def compile_model(export_dir, dataset, output, calibration_count=64, pair_count=12,background_bank=None):
    export_dir = export_dir.resolve()
    dataset = dataset.resolve()
    export = json.loads((export_dir/'RESULT.json').read_text(encoding='utf8'))
    model = export_dir/'topface.onnx'
    if (export['status'] != 'PYTORCH_ONNX_PARITY_PASS' or sha(model) != export['onnx_sha256'] or
            sha(dataset/'AUDIT_V11.json') != export['dataset_audit_sha256']):
        raise ValueError('Trained export and PTQ dataset identities must match')
    if not 16 <= calibration_count <= 256 or not 2 <= pair_count <= 32:
        raise ValueError('Calibration and verification sizes are outside the supported range')
    width,height = export['width'],export['height']
    backgrounds=None
    if background_bank is not None:
        from background_augmentation import BackgroundBank
        backgrounds=BackgroundBank(background_bank)
    if export['experiment']=='formal_full' and export.get('real_background_augmentation')!=(
            backgrounds.identity if backgrounds is not None else None):
        raise ValueError('Formal PTQ backgrounds differ from the trained augmentation policy')
    calibration,paired = load_samples(dataset,width,height,calibration_count,pair_count,backgrounds)
    output = output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    plugin_spec = importlib.util.find_spec('_nncase')
    if plugin_spec is None:
        raise ValueError('The pinned nncase compiler is unavailable')
    plugin_path = str(Path(plugin_spec.origin).parent)
    previous_environment = {key:os.environ.get(key) for key in ('NNCASE_PLUGIN_PATH','PATH')}
    os.environ['NNCASE_PLUGIN_PATH'] = plugin_path
    os.environ['PATH'] = plugin_path+os.pathsep+os.environ.get('PATH','')
    dll = os.add_dll_directory(plugin_path) if os.name=='nt' else None
    current = Path.cwd()
    try:
        import nncase
        versions = {name:importlib.metadata.version(name) for name in ('nncase','nncase-kpu')}
        if set(versions.values()) != {'2.11.0'}:
            raise ValueError('Use the pinned compiler and K230 plugin versions')
        os.chdir(output)
        options = nncase.CompileOptions()
        settings = {'target':'k230','preprocess':True,'input_shape':[1,3,height,width],
                    'input_type':'uint8','input_range':[0,255],'swapRB':True,
                    'mean':MEAN.tolist(),'std':STD.tolist(),'input_layout':'NCHW',
                    'shape_bucket_enable':False,'dump_ir':True,'dump_asm':True,
                    'dump_dir':str(output/'compiler-dump')}
        for key,value in settings.items():
            setattr(options,key,value)
        compiler = nncase.Compiler(options)
        compiler.import_onnx(model.read_bytes(),nncase.ImportOptions())
        ptq = nncase.PTQTensorOptions()
        ptq.samples_count = len(calibration)
        ptq.calibrate_method = 'NoClip'
        ptq.quant_type = ptq.w_quant_type = 'uint8'
        ptq.set_tensor_data([[rgb] for _,rgb,_ in calibration])
        compiler.use_ptq(ptq)
        compiler.compile()
        binary = compiler.gencode_tobytes()
        (output/'topface.kmodel').write_bytes(binary)
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        runtime = ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        simulator = nncase.Simulator()
        simulator.load_model(binary)
        comparisons = []
        for sample,rgb,bgr in paired:
            reference = runtime.run(None,{runtime.get_inputs()[0].name:bgr})[0][0]
            simulator.set_input_tensor(0,nncase.RuntimeTensor.from_numpy(rgb))
            simulator.run()
            actual = simulator.get_output_tensor(0).to_numpy().copy()[0]
            if actual.shape!=reference.shape or not np.isfinite(actual).all():
                raise ValueError('Simulator output shape or finite contract differs')
            error = np.abs(actual-reference)
            comparisons.append({'scene':sample['scene'],'raw_channel_mae':error.mean(axis=0).tolist(),
                                'raw_channel_max':error.max(axis=0).tolist(),
                                'decoded':compare_detections(reference,actual,width,height)})
        identity={'kmodel_sha256':sha(output/'topface.kmodel'),'onnx_sha256':export['onnx_sha256'],
                  'checkpoint_sha256':export['checkpoint_sha256'],'model_wh':[width,height],
                  'output_shape':[1,*reference.shape],'input':'NCHW_RGB_UINT8',
                  'score_threshold':.25,'nms_iou':.5,'compiler':versions,'board_status':'NOT_RUN'}
        atomic_json(output/'identity.json',identity)
        contract = json.loads((export_dir/'output_contract.json').read_text(encoding='utf8'))
        contract.update(training_status='TRAINED',ptq_status='TRAIN_ONLY_PTQ_PAIRED',board_status='NOT_RUN')
        contract['compiled_input']={'shape':[1,3,height,width],'color':'RGB','dtype':'uint8','layout':'NCHW',
                                    'normalization':'nncase swaps RGB to BGR and applies the model mean and std'}
        atomic_json(output/'output_contract.json',contract)
        result={'status':'TRAINED_MODEL_COMPILED_AND_PAIRED','identity':identity,'compiler_settings':settings,
                'calibration_background_bank':backgrounds.identity if backgrounds is not None else None,
                'calibration_members':[row for row,_,_ in calibration],
                'paired_members':[row for row,_,_ in paired],'paired_measurements':comparisons}
        atomic_json(output/'RESULT.json',result)
    finally:
        os.chdir(current)
        if dll is not None:
            dll.close()
        for key,value in previous_environment.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key]=value
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('export-dir','dataset','output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--calibration-count', type=int, default=64)
    parser.add_argument('--pair-count', type=int, default=12)
    parser.add_argument('--background-bank',type=Path)
    result = compile_model(**vars(parser.parse_args()))
    print(json.dumps(result['identity'],indent=2))
