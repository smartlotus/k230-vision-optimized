"""Export a byte-pinned trained checkpoint and verify its ONNX forward pass."""
import argparse
import json
from pathlib import Path

import onnx
import torch
import numpy as np

from cloud_store import atomic_json, sha
from export_nanodet_candidate import compare
from nanodet_topface import NanoDetTopFace, output_contract


def export(checkpoint, checkpoint_sha256, output, dataset=None):
    if sha(checkpoint) != checkpoint_sha256:
        raise ValueError('Trained checkpoint differs from its accepted digest')
    value = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if value['schema'] != 'topface-training-checkpoint-v1.1':
        raise ValueError('Expected a TopFaceNet training checkpoint')
    state = value['state']
    if state['optimizer_updates'] < 1 or state['kind'] not in (
        'pilot_pretrained', 'pilot_random_control', 'pilot_geometry_refine', 'formal_full'
    ):
        raise ValueError('Expected a learned Pilot or formal model')
    torch.set_num_threads(2)
    width, height = state['width'], state['height']
    model = NanoDetTopFace().eval()
    model.load_state_dict(value['model'], strict=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model_path = output/'topface.onnx'
    torch.onnx.export(model, torch.zeros(1, 3, height, width), model_path, input_names=['images'],
                      output_names=['topface_raw'], opset_version=11, do_constant_folding=True, dynamo=False)
    onnx.checker.check_model(onnx.load(model_path))
    receipt = {'schema': 'topface-trained-export-v1.1', 'checkpoint_sha256': checkpoint_sha256,
               'optimizer_updates': state['optimizer_updates'], 'epoch': state['epoch'],
               'experiment': state['kind'], 'dataset_audit_sha256': state['dataset_audit_sha256'],
               'real_background_augmentation':state.get('real_background_augmentation'),
               'width': width, 'height': height, 'onnx_sha256': sha(model_path),
               'onnx_bytes': model_path.stat().st_size, 'pytorch_ort': compare(model, model_path, height, width)}
    if dataset is not None:
        from compile_trained_topface import load_samples
        from export_nanodet_candidate import session
        if sha(dataset/'AUDIT_V11.json') != state['dataset_audit_sha256']:
            raise ValueError('Actual-image parity dataset differs from the checkpoint')
        _,samples=load_samples(dataset,width,height,64,12)
        runtime=session(model_path)
        rows=[]
        for member,_,tensor in samples:
            with torch.inference_mode():
                reference=model(torch.from_numpy(tensor)).numpy()[0]
            actual=runtime.run(None,{runtime.get_inputs()[0].name:tensor})[0][0]
            error=np.abs(actual-reference)
            rows.append({'scene':member['scene'],'rgb_sha256':member['rgb_sha256'],
                         'allclose':bool(np.allclose(actual,reference,rtol=1e-3,atol=1e-3)),
                         'raw_channel_mae':error.mean(axis=0).tolist(),
                         'raw_channel_max':error.max(axis=0).tolist()})
        receipt['actual_train_image_pytorch_ort']={'images':len(rows),'rtol':1e-3,'atol':1e-3,
            'pass':all(row['allclose'] for row in rows),'measurements':rows}
        receipt['pytorch_ort']['pass'] &= receipt['actual_train_image_pytorch_ort']['pass']
    receipt['status'] = 'PYTORCH_ONNX_PARITY_PASS' if receipt['pytorch_ort']['pass'] else 'PYTORCH_ONNX_PARITY_FAILED'
    atomic_json(output/'RESULT.json', receipt)
    contract=output_contract(width,height)
    contract.update(training_status='TRAINED',candidate_not_final=state['kind']!='formal_full',
                    source_checkpoint_sha256=checkpoint_sha256,optimizer_updates=state['optimizer_updates'])
    atomic_json(output/'output_contract.json',contract)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path)
    result = export(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2))
    if result['status'] != 'PYTORCH_ONNX_PARITY_PASS':
        raise SystemExit(1)
