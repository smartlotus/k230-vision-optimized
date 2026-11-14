"""Frozen model-pixel coordinates, BGR preprocessing and 50-channel decoding.

Geometry uses continuous image-edge coordinates: top-left=(0,0), bottom-right
=(W,H). The simulated K uses cx=W/2,cy=H/2. NanoDet priors are (col*s,row*s),
with NO half-stride offset. Image resampling uses OpenCV bilinear/half-pixel;
continuous edge coordinates scale by gain and add integer centered padding.
"""
import math
import cv2
import numpy as np

STRIDES=(8,16,32,64)
MEAN=np.array([103.53,116.28,123.675],np.float32)
STD=np.array([57.375,57.12,58.395],np.float32)


def letterbox_parameters(width,height,source_width=640,source_height=360):
    if (width,height) not in ((224,128),(320,192),(384,224)):
        raise ValueError('Unregistered input;320x180 must pad to320x192')
    gain=min(width/source_width,height/source_height)
    rw,rh=round(source_width*gain),round(source_height*gain)
    top,left=(height-rh)//2,(width-rw)//2
    return {'source_wh':[source_width,source_height],'model_wh':[width,height],
        'gain':gain,'resized_wh':[rw,rh],'padding_tblr':[top,height-rh-top,left,width-rw-left],
        'pad_value':114,'coordinate_origin':'continuous top-left image edge',
        'point_forward':'source_xy*gain+[left,top]','point_inverse':'(model_xy-[left,top])/gain',
        'axis_forward':'source_length*gain','axis_inverse':'model_length/gain',
        'orientation':'theta in radians encoded sin(2theta),cos(2theta); unchanged by isotropic resize'}


def preprocess_bgr(image,width=320,height=192):
    if image.dtype!=np.uint8 or image.shape!=(360,640,3):raise ValueError('Expected uint8 BGR640x360')
    info=letterbox_parameters(width,height)
    canvas=np.full((height,width,3),114,np.uint8)
    top,_,left,_=info['padding_tblr'];rw,rh=info['resized_wh']
    canvas[top:top+rh,left:left+rw]=cv2.resize(image,(rw,rh),interpolation=cv2.INTER_LINEAR)
    normalized=((canvas.astype(np.float32)-MEAN)/STD).transpose(2,0,1).copy()
    return normalized,info


def priors(width,height):
    points=[];scales=[]
    for stride in STRIDES:
        yy,xx=np.meshgrid(np.arange(math.ceil(height/stride)),np.arange(math.ceil(width/stride)),indexing='ij')
        points.append(np.stack([xx.ravel(),yy.ravel()],axis=1)*stride)
        scales.append(np.full(xx.size,stride,np.float32))
    return np.concatenate(points).astype(np.float32),np.concatenate(scales)


def sigmoid(logits):
    logits=np.asarray(logits,dtype=np.float32);out=np.empty_like(logits)
    positive=logits>=0;out[positive]=1/(1+np.exp(-logits[positive]))
    e=np.exp(logits[~positive]);out[~positive]=e/(1+e)
    return out


def decode_boxes(raw,width,height):
    points,scale=priors(width,height)
    raw=np.asarray(raw,dtype=np.float32)
    if raw.shape!=(len(points),50) or not np.isfinite(raw).all():
        raise ValueError('Wrong/nonfinite declared50-channel raw tensor; do not reinterpret as probabilities')
    logits=raw[:,6:38].reshape(-1,4,8);ex=np.exp(logits-logits.max(axis=2,keepdims=True))
    distances=(ex/ex.sum(axis=2,keepdims=True)*np.arange(8,dtype=np.float32)).sum(axis=2)*scale[:,None]
    return np.column_stack([points[:,0]-distances[:,0],points[:,1]-distances[:,1],
                            points[:,0]+distances[:,2],points[:,1]+distances[:,3]])


def box_iou(one,many):
    inter=np.maximum(0,np.minimum(one[2:],many[:,2:])-np.maximum(one[:2],many[:,:2])).prod(axis=1)
    a=np.maximum(0,one[2:]-one[:2]).prod();b=np.maximum(0,many[:,2:]-many[:,:2]).prod(axis=1)
    return inter/np.maximum(a+b-inter,1e-9)


def decode(raw,width=320,height=192,score_threshold=.25,nms_iou=.5,max_instances=5):
    if not 0<score_threshold<1 or max_instances!=5:raise ValueError('Declared threshold and five-instance contract required')
    raw=np.asarray(raw,dtype=np.float32)
    if raw.ndim==3 and raw.shape[0]==1:raw=raw[0]
    boxes=decode_boxes(raw,width,height);points,scale=priors(width,height)
    color_prob=sigmoid(raw[:,:6]);scores=color_prob.max(axis=1)
    info=letterbox_parameters(width,height);top,_,left,_=info['padding_tblr'];rw,rh=info['resized_wh']
    intersects=(np.minimum(boxes[:,2:],[left+rw,top+rh])-np.maximum(boxes[:,:2],[left,top])>0).all(axis=1)
    nms_boxes=np.minimum(np.maximum(boxes,[left,top,left,top]),[left+rw,top+rh,left+rw,top+rh])
    candidates=np.flatnonzero((scores>=score_threshold)&intersects)
    order=candidates[np.argsort(-scores[candidates],kind='stable')]
    selected=[]
    while len(order):
        index=int(order[0]);selected.append(index)
        rest=order[1:];order=rest[box_iou(nms_boxes[index],nms_boxes[rest])<=nms_iou]
        if len(selected)>=200:break  # finite pre-NMS evaluation cap, never a3-object cap
    offset=np.array([left,top],np.float32)
    gain=info['gain'];detections=[]
    for index in selected[:max_instances]:
        source_box=(boxes[index]-np.tile(offset,2))/gain
        visible_box=np.minimum(np.maximum(source_box,[0,0,0,0]),[640,360,640,360])
        if min(visible_box[2:]-visible_box[:2])<=0:continue
        physical=(points[index]+raw[index,38:40]*scale[index]-offset)/gain
        conic=(points[index]+raw[index,48:50]*scale[index]-offset)/gain
        valid_visible=sigmoid(raw[index,40:42]);quality=sigmoid(raw[index,46:48])
        orientation=raw[index,44:46];norm=float(np.linalg.norm(orientation))
        theta=.5*math.atan2(float(orientation[0]),float(orientation[1])) if norm>1e-6 else None
        ellipse=None
        # Reject unrepresentable geometry; never clip logits/probabilities to
        # conceal an output-contract failure. Detection remains separately valid.
        if np.max(np.abs(raw[index,42:44]))<20 and theta is not None:
            axes=np.exp(raw[index,42:44].astype(np.float64))*scale[index]/gain
            if axes[1]>axes[0]:axes=axes[::-1];theta+=math.pi/2
            theta=(theta+math.pi/2)%math.pi-math.pi/2
            ellipse={'conic_center_xy':conic.tolist(),'semi_major':float(axes[0]),'semi_minor':float(axes[1]),
                'orientation_deg':math.degrees(theta),'orientation_valid':bool(axes[1]/axes[0]<.98),
                'geometry_confidence':float(quality[0])}
        detections.append({'score':float(scores[index]),'color_id':int(np.argmax(color_prob[index]))+1,
            'color_probabilities':color_prob[index].tolist(),'bbox_xyxy_visible':visible_box.tolist(),
            'bbox_xyxy_unclipped':source_box.tolist(),'center_xy_projected_3d':physical.tolist(),
            'center_valid_probability':float(valid_visible[0]),'center_visible_probability':float(valid_visible[1]),
            'ellipse':ellipse,'visibility_quality':float(quality[1]),
            'boundary_candidate':bool(np.any(np.abs(visible_box-source_box)>1e-4))})
    count=len(detections)
    uncertain=any(d['boundary_candidate'] for d in detections)
    observed='EMPTY' if count==0 else 'OK' if count<=3 else 'OVER_CAPACITY'
    return {'detections':detections,'count':count,'post_nms_total_before_five_limit':len(selected),
        'observed_capacity_state':observed,
        'capacity_state':'UNCERTAIN' if uncertain and count<4 else observed,
        'uncertain':uncertain,
        'thresholds':{'score':score_threshold,'class_agnostic_nms_iou':nms_iou},
        'scope':'single model declared logits; no second network; no board accuracy claim'}
