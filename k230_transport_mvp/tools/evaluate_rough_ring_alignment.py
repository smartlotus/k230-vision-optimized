"""Host-only replay of a frozen native RGB565 + native LAB grid sample.

An explicitly chosen material-interior ROI proposes a threshold. The paper
detector separately consumes only the grayscale paper ROI and dark strokes.
No frame sequence, plane calibration, target number or placement score is inferred.
"""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from k230.camera_roles import OBJECT_CAMERA_ID
from vision.ellipse_native import contour_fit
from vision.ring_native import detect_paper_targets,associate_material_target
from vision.alignment import measure_alignment


def evaluate(sample, native_bytes, roi, material_interior_roi):
    if (sample.get('source')!='REAL_CAMERA_NATIVE_LAB' or sample.get('is_simulated') is not False
            or sample.get('camera_id')!=OBJECT_CAMERA_ID or sample.get('mode')!='RING'):
        raise ValueError('explicit native object-camera RING sample required')
    width,height=sample.get('width'),sample.get('height')
    stride,x0,y0=sample.get('stride'),sample.get('x0'),sample.get('y0')
    if (any(type(v) is not int or v<1 for v in (width,height,stride))
            or any(type(v) is not int for v in (x0,y0))
            or len(native_bytes)!=width*height*2 or not 0<=x0<stride or not 0<=y0<stride):
        raise ValueError('native shape, byte count and grid origin required')
    if (type(sample.get('frame_id')) is not int or sample['frame_id']<1
            or type(sample.get('timestamp_ms')) is not int or sample['timestamp_ms']<0
            or not isinstance(sample.get('image_profile'),dict)
            or sample['image_profile'].get('width')!=width
            or sample['image_profile'].get('height')!=height):
        raise ValueError('frame identity and matching image profile required')
    for rect in (roi,material_interior_roi):
        if (len(rect)!=4 or any(type(v) is not int for v in rect)
                or rect[0]<0 or rect[1]<0 or min(rect[2:])<8
                or rect[0]+rect[2]>width or rect[1]+rect[3]>height):
            raise ValueError('explicit in-image observation/interior ROI required')
    if (material_interior_roi[0]<roi[0] or material_interior_roi[1]<roi[1]
            or material_interior_roi[0]+material_interior_roi[2]>roi[0]+roi[2]
            or material_interior_roi[1]+material_interior_roi[3]>roi[1]+roi[3]):
        raise ValueError('material interior must lie inside observation ROI')
    grid=np.asarray(sample['lab_grid'])
    expected=(len(range(y0,height,stride)),len(range(x0,width,stride)),3)
    if grid.shape!=expected or not np.isfinite(grid).all():raise ValueError('complete finite LAB grid required')
    for axis,(low,high) in enumerate(((0,100),(-128,127),(-128,127))):
        if np.any(grid[:,:,axis]<low) or np.any(grid[:,:,axis]>high):raise ValueError('LAB grid outside native range')
    yy,xx=np.indices(grid.shape[:2]);px,py=x0+xx*stride,y0+yy*stride
    x,y,w,h=material_interior_roi
    selected=(px>=x)&(px<x+w)&(py>=y)&(py<y+h)
    if np.count_nonzero(selected)<16:raise ValueError('at least sixteen interior native cells required')
    threshold=[]
    for axis,(low,high) in enumerate(((0,100),(-128,127),(-128,127))):
        a,b=np.percentile(grid[selected][:,axis],[2,98])
        threshold.extend((max(low,int(np.floor(a))-8),min(high,int(np.ceil(b))+8)))
    x,y,w,h=roi
    mask=np.where((px>=x)&(px<x+w)&(py>=y)&(py<y+h),255,0).astype(np.uint8)
    for axis in range(3):mask[(grid[:,:,axis]<threshold[2*axis])|(grid[:,:,axis]>threshold[2*axis+1])]=0
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    materials=[]
    for c in contours:
        area=cv2.contourArea(c)*stride*stride
        if not 500<=area<=6000 or len(c)<16:continue
        center,axes,angle=cv2.fitEllipse(c)
        native_fit=((center[0]*stride+x0,center[1]*stride+y0),tuple(a*stride for a in axes),angle)
        points=[[v[0]*stride+x0,v[1]*stride+y0] for v in c.reshape(-1,2).tolist()]
        fit=contour_fit(points,roi=roi,pixel_area=area,native_fit=native_fit)
        if fit['accepted']:materials.append(fit)
    report={'scope':'HOST_REPLAY_OF_ONE_NATIVE_FRAME','frame_id':sample.get('frame_id'),
        'threshold_candidate':threshold,'interior_native_cells':int(np.count_nonzero(selected)),
        'material_candidates':materials,'paper_candidates':[],
        'alignment':{'status':'UNKNOWN','reason':'UNIQUE_MATERIAL_REQUIRED','quality':None},
        'official_points':None,'motion_authorized':False,'plane_projection_verified':False,
        'native_sha256':hashlib.sha256(native_bytes).hexdigest(),
        'limits':['One frozen frame cannot prove stability.',
                  'Material top and paper can have height parallax; pixel offsets are not robot-plane errors.',
                  'Manually selected interior ROI and proposed LAB threshold are unverified.',
                  'Input files are supplied as a same-capture pair; recorded hashes bind this replay, not independent capture authenticity.']}
    image=np.frombuffer(native_bytes,np.uint8).reshape(height,width,2)
    gray=cv2.cvtColor(cv2.cvtColor(image,cv2.COLOR_BGR5652BGR),cv2.COLOR_BGR2GRAY)
    paper=detect_paper_targets(gray[y:y+h,x:x+w],roi,pixel_scale=1)
    report['paper_candidates']=paper['groups']
    report['detector_relationship']='INDEPENDENT_MATERIAL_AND_PAPER_THEN_ASSOCIATION'
    if not paper['complete']:
        report['alignment']['reason']=paper['reason'];return report
    pair=associate_material_target(materials,paper['groups'])
    if not pair['target']:
        report['alignment']['reason']=pair['reason'];return report
    material=pair['material']
    report['associated_paper_candidate']=pair['target']
    metadata={k:sample[k] for k in ('camera_id','frame_id','timestamp_ms','is_simulated','image_profile')}
    a=dict(metadata,status='GEOMETRY_ONLY',u_px=material['center'][0],v_px=material['center'][1],stable=False)
    b=dict(metadata,status='GEOMETRY_ONLY',u_px=pair['target']['center'][0],v_px=pair['target']['center'][1],stable=False)
    report['alignment']=measure_alignment(a,b,now_ms=sample['timestamp_ms'])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--roi',nargs=4,type=int,required=True)
    p.add_argument('--material-interior-roi',nargs=4,type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Existing evidence is immutable; choose a new output')
    sample_bytes=a.sample.read_bytes()
    result=evaluate(json.loads(sample_bytes),a.native.read_bytes(),a.roi,a.material_interior_roi)
    result.update(sample_sha256=hashlib.sha256(sample_bytes).hexdigest(),roi=a.roi,
                  material_interior_roi=a.material_interior_roi)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({'evidence':str(a.output),'material_candidates':len(result['material_candidates']),
                     'paper_candidates':len(result['paper_candidates']),'alignment':result['alignment']}))


if __name__=='__main__':main()
