"""Regenerate frozen reference embeddings from corrected semantic labels."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from dataset import digest


def reference_bank(field, centers, views):
    angles = np.arange(views) * math.tau / views
    focal = math.hypot(32,18) / (2*math.tan(math.radians(96)/2))
    u,v = np.meshgrid((np.arange(32)-15.5)*400/focal,(8.5-np.arange(18))*400/focal)
    cosine,sine = np.cos(angles)[:,None],np.sin(angles)[:,None]
    x = centers[:,None,None,0] + cosine*u.ravel() - sine*v.ravel()
    y = centers[:,None,None,1] + sine*u.ravel() + cosine*v.ravel()
    h,w = field.shape
    cols = np.rint((x/3000+.5)*w-.5).astype(int)
    rows = np.rint((.5-y/3000)*h-.5).astype(int)
    bank = field[rows.clip(0,h-1),cols.clip(0,w-1)]
    bank[(rows<0)|(rows>=h)|(cols<0)|(cols>=w)] = 255
    return bank


def main():
    parser = argparse.ArgumentParser()
    for name in ('mask','map-best','parent','matcher','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    sys.path.insert(0,str(args.matcher.resolve().parent))
    spec = importlib.util.spec_from_file_location('camera_xy_v2_matcher',args.matcher)
    matcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher)
    state = torch.load(args.map_best,map_location='cpu',weights_only=True)
    parent = torch.load(args.parent,map_location='cpu',weights_only=True)
    expected = ('camera_x_mm','camera_y_mm')
    if tuple(state['targets'])!=expected or tuple(parent['targets'])!=expected:
        raise ValueError('Expected an XY-only parent')
    if state['dataset_manifest_sha256']!=parent['dataset_manifest_sha256']:
        raise ValueError('Parent training lineage differs')
    net = matcher.MapMatcher().eval()
    net.load_state_dict(state['weights'],strict=True)
    with Image.open(args.mask) as image:
        field = np.asarray(image).copy()
    if field.shape!=(800,800):
        raise ValueError('Expected corrected 800x800 mm-scale overview mask')
    old_centers = np.asarray(parent['centers_mm'])
    old_reference = parent['state']['reference']
    for name,value in net.rgb.state_dict().items():
        if not torch.equal(value,parent['state']['rgb.'+name]):
            raise ValueError('Map-best RGB encoder differs from accepted parent')
    xs,ys = np.meshgrid(np.arange(-1175,1200,50),np.arange(-1175,1200,50))
    centers = np.column_stack((xs.ravel(),ys.ravel()))
    cols = ((centers[:,0]/3000+.5)*800).astype(int)
    rows = ((.5-centers[:,1]/3000)*800).astype(int)
    # Keep every optical center in field, including above the forbidden zones.
    # The mask class describes image appearance, rather than hard reachability.
    centers = centers[field[rows,cols]!=255]
    views = 12
    descriptors = matcher.reference_descriptors(reference_bank(field,centers,views))
    with torch.inference_mode():
        embedding = net.map_embedding(torch.from_numpy(descriptors))
        frozen = matcher.FrozenMapHead(net,embedding)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({**parent,'state':frozen.state_dict(),'centers_mm':centers.tolist(),
                'reference_views':views,'reference_mask_sha256':digest(args.mask)},args.out)
    report = {'status':'CORRECTED_REFERENCE_READY','parent_sha256':digest(args.parent),
        'checkpoint_sha256':digest(args.out),'source_map_best_sha256':digest(args.map_best),
        'field_mask_sha256':digest(args.mask),'targets':list(expected),'rgb_encoder_equal':True,
        'sites':len(centers),'old_sites':len(old_centers),'views_per_site':views,
        'reference_bytes':frozen.reference.numel()*4,'old_reference_bytes':old_reference.numel()*4,
        'method':'50 mm full-field anchors, existing frozen descriptor model, 12 internal orientation views'}
    (args.out.parent/'REFERENCE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
