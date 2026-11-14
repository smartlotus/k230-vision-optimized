"""Evaluate one frozen TopFaceNet model on the independent synthetic test."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cloud_store import atomic_json, sha
from evaluate_task import evaluate
from nanodet_topface import NanoDetTopFace
from task_training import TopFaceDataset, collate


def evaluate_final(dataset, checkpoint, checkpoint_sha256, output, device='cuda:0'):
    if output.exists():
        raise FileExistsError('Preserve the existing final test result')
    if sha(checkpoint) != checkpoint_sha256:
        raise ValueError('Frozen final checkpoint digest differs')
    value = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = value['state']
    if state['kind'] != 'formal_full' or state['optimizer_updates'] < 1:
        raise ValueError('Independent test requires a trained formal model')
    test = TopFaceDataset(dataset, 'synthetic_test', state['width'], state['height'])
    if test.audit_sha256 != state['dataset_audit_sha256']:
        raise ValueError('Synthetic test inventory differs from the frozen training manifest')
    loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate)
    device = torch.device(device)
    model = NanoDetTopFace().to(device)
    model.load_state_dict(value['model'], strict=True)
    result = evaluate(model, loader, device, score_threshold=.25, include_slices=True,angle_diagnostics=True)
    result.update(scope='HELD_OUT_SYNTHETIC_TEST', checkpoint_sha256=checkpoint_sha256,
                  dataset_audit_sha256=test.audit_sha256, model_epoch=state['epoch'],
                  optimizer_updates=state['optimizer_updates'], physical_mm_error='UNAVAILABLE_WITHOUT_CALIBRATION')
    atomic_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    result = evaluate_final(**vars(parser.parse_args()))
    print(json.dumps({key:result[key] for key in ('scope','images','task_mAP50',
          'task_mAP50_95','class_agnostic_recall50','matched_color_accuracy')}, indent=2))
