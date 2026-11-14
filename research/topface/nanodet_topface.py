"""NanoDet-Plus-m core with an explicit six-color/center/ellipse head.

Feature-extractor modules are unmodified upstream Apache-2.0 code (SOURCE.json).
This adapter avoids Lightning-only training/serialization dependencies. It is
not the permitted fallback custom detector: the NanoDet backbone, GhostPAN,
four DWConv towers and DFL box representation are retained.
"""
import hashlib
import math
from pathlib import Path
import pickle
import types

import torch
from torch import nn

from vendor.nanodet_core.model.backbone.shufflenetv2 import ShuffleNetV2
from vendor.nanodet_core.model.fpn.ghost_pan import GhostPAN
from vendor.nanodet_core.model.module.conv import DepthwiseConvModule

CHECKPOINT_SHA256 = 'f4c6080f3ef35a64c3030d559438b436462f227ce29700bca71327203e42dea2'
STRIDES=(8,16,32,64)


class IgnoredLightningCheckpointKey:
    """Opaque dictionary key for legacy callback metadata; never executed."""


class TensorCheckpointUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name)==('pytorch_lightning.callbacks.model_checkpoint','ModelCheckpoint'):
            return IgnoredLightningCheckpointKey
        allowed={('collections','OrderedDict'),('torch','FloatStorage'),
                 ('torch','LongStorage'),('torch._utils','_rebuild_tensor_v2')}
        if (module,name) not in allowed:
            raise pickle.UnpicklingError('Unreviewed checkpoint global')
        return super().find_class(module,name)


def checkpoint_state(path,expected_sha256):
    path=Path(path)
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if not expected_sha256 or actual!=expected_sha256:
        raise ValueError('Pretrained checkpoint must match reviewed original bytes')
    reader=types.SimpleNamespace(Unpickler=TensorCheckpointUnpickler,__name__='topface_tensor_checkpoint')
    checkpoint=torch.load(path,map_location='cpu',pickle_module=reader,weights_only=False)
    state=checkpoint.get('state_dict',checkpoint)
    # Official release stores inference weights under model.*, training-only
    # auxiliary modules are ignored explicitly after exact shape checks.
    prefix='avg_model.' if any(k.startswith('avg_model.') for k in state) else 'model.'
    state={key.removeprefix(prefix):value for key,value in state.items() if key.startswith(prefix)}
    if not state:raise ValueError('Original release model.* tensor state not found')
    return state


class PredictionHead(nn.Module):
    def __init__(self,task):
        super().__init__();self.task=task
        self.cls_convs=nn.ModuleList([nn.ModuleList([
            DepthwiseConvModule(96,96,5,stride=1,padding=2,norm_cfg=dict(type='BN'),activation='LeakyReLU')
            for _ in range(2)]) for _ in STRIDES])
        self.gfl_cls=nn.ModuleList([nn.Conv2d(96,50 if task else 112,1) for _ in STRIDES])
        if task:
            for layer in self.gfl_cls:
                nn.init.normal_(layer.weight,std=.01);nn.init.zeros_(layer.bias)
                nn.init.constant_(layer.bias[:6],-4.595)

    def forward(self,features):
        outputs=[]
        for feat,tower,predict in zip(features,self.cls_convs,self.gfl_cls):
            for conv in tower:feat=conv(feat)
            pred=predict(feat)
            if not self.task:
                # Match the ORIGINAL published ONNX branch exactly. Task export
                # deliberately returns logits and decodes once outside the graph.
                pred=torch.cat((pred[:,:80].sigmoid(),pred[:,80:]),dim=1)
            outputs.append(pred.flatten(start_dim=2))
        return torch.cat(outputs,dim=2).permute(0,2,1)


class NanoDetTopFace(nn.Module):
    def __init__(self,task=True):
        super().__init__();self.task=task
        self.backbone=ShuffleNetV2(model_size='1.0x',out_stages=(2,3,4),
                                   activation='LeakyReLU',pretrain=False)
        self.fpn=GhostPAN(in_channels=[116,232,464],out_channels=96,kernel_size=5,
                         num_extra_level=1,use_depthwise=True,activation='LeakyReLU')
        self.head=PredictionHead(task)

    def forward(self,x):return self.head(self.fpn(self.backbone(x)))

    def initialize_pretrained(self,path,expected_sha256):
        source=checkpoint_state(path,expected_sha256);target=self.state_dict()
        loaded={};skipped=[]
        for key,value in source.items():
            if key in target and target[key].shape==value.shape:
                loaded[key]=value
            elif self.task and key.startswith('head.gfl_cls.'):
                # Keep original 32 DFL regression channels; replace 80 COCO
                # categories with six colors and twelve project geometry channels.
                new=target[key].clone();new[6:38]=value[80:112];loaded[key]=new
            else:skipped.append(key)
        missing=sorted(set(target)-set(loaded))
        if missing:raise ValueError('Pretrained feature/tower state is incomplete: '+','.join(missing[:8]))
        self.load_state_dict(loaded,strict=True)
        return {'loaded_tensors':len(loaded),'ignored_training_only_tensors':len(skipped),
            'missing_tensors':missing,'checkpoint_sha256':expected_sha256,
            'classification_initialization':'random six-color' if self.task else 'original COCO',
            'geometry_initialization':'random twelve-channel' if self.task else 'absent',
            'box_regression_initialization':'original COCO DFL channels',
            'optimizer_updates':0}


def output_contract(width,height):
    if width%32 or height%32:raise ValueError('Use aligned dimensions; strict320x180 must letterbox/pad to320x192')
    locations=sum(math.ceil(width/s)*math.ceil(height/s) for s in STRIDES)
    return {'schema':'topface-nanodet-candidate-output-v1.1','candidate_not_final':True,
        'input':{'shape':[1,3,height,width],'dtype':'float32','layout':'NCHW','color':'BGR',
            'normalization':'(uint8-[103.53,116.28,123.675])/[57.375,57.12,58.395] exactly once',
            'source_wh':[640,360],'resize':'aspect-preserving bilinear; centered letterbox pad114'},
        'output':{'name':'topface_raw','shape':[1,locations,50],'dtype':'float32',
            'channels':{'color_quality_logits':[0,6],'box_ltrb_dfl_logits':[6,38],
                'physical_center_xy_stride_offsets':[38,40],'center_valid_visible_logits':[40,42],
                'ellipse_log_semiaxes_stride':[42,44],'orientation_sin2_cos2':[44,46],
                'geometry_confidence_logit':46,'visibility_quality_logit':47,
                'ellipse_conic_center_xy_stride_offsets':[48,50]},
            'grid':'NanoDet x/y integer grid at strides8,16,32,64; row-major within each level',
            'bbox_decode':'softmax each8-bin DFL then expectation0..7 times stride; ltrb around grid',
            'center_decode':'grid + raw offset * stride; physical CAD center, not conic center',
            'ellipse_decode':'conic center=grid+channels48:50*stride; exp(log semiaxes)*stride; orientation atan2(sin2,cos2)/2',
            'nms':'host after declared sigmoid of class logits; class-agnostic top-face box NMS',
            'maximum_returned_instances':5,'capacity':'0 EMPTY,1..3 OK,4..5 OVER_CAPACITY; uncertainty separate'},
        'training_status':'NOT_TRAINED','ptq_status':'NOT_FINAL_CALIBRATION','board_status':'NOT_RUN'}
