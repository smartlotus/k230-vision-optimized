"""Material-only and later two-class YOLOv8 proposals, never geometry or score.

The fixed firmware's native postprocessor scans all N rows and suppresses across
classes. Give it one contiguous float32 full-N buffer per class instead. Model
coordinates are shifted by the declared padding before native source scaling.
"""
from math import isfinite

CLASSES=('visible_top','paper_target')


def input_contract(size=320,padding='center',pad_value=114):
    if type(size) is not int or size not in (224,320):raise ValueError('Model size224/320 required')
    if padding not in ('center','top_left'):raise ValueError('Declared padding convention required')
    if type(pad_value) is not int or not 0<=pad_value<=255:raise ValueError('Uint8 pad required')
    # Fixed source binding, not the IDE preview's dimensions. Integer arithmetic
    # keeps 224x126 independent of float truncation on different runtimes.
    content_h=size*360//640;top=(size-content_h)//2 if padding=='center' else 0
    return {'source_wh':[640,360],'model_wh':[size,size],'resized_wh':[size,content_h],
        'pad_tblr':[top,size-content_h-top,0,0],'pad_rgb':[pad_value]*3,
        'gain':size/640,'input_dtype':'uint8','input_layout':'NCHW','color_order':'RGB',
        'normalization':'EXACTLY_ONCE_IN_COMPILED_MODEL_TO_FLOAT_0_1',
        'positions':sum((size//stride)**2 for stride in (8,16,32))}


def _numpy():
    try:from ulab import numpy as np
    except ImportError:import numpy as np
    return np


def decode_two_class_proposals(raw,contract,confidence=(.25,.25),nms=.45,limits=(16,16),native=None,np=None):
    return _decode_proposals(raw,contract,CLASSES,confidence,nms,limits,native,np)


def decode_material_proposals(raw,contract,confidence=.25,nms=.45,limit=16,native=None,np=None):
    """First material-only pilot: one visible_top class, no paper predictions."""
    return _decode_proposals(raw,contract,('visible_top',),(confidence,),nms,(limit,),native,np)


def _decode_proposals(raw,contract,class_names,confidence,nms,limits,native,np):
    """Decode each class without losing overlapping paper/top proposals.

    Standard exported YOLOv8 output must be decoded cxcywh + declared sigmoid
    scores, shape [1,4+classes,N], without NMS. Model metadata must confirm this
    contract before calling. No outputs are interpreted as fitted ellipses.
    """
    if np is None:np=_numpy()
    expected=input_contract(contract['model_wh'][0],
        'center' if contract['pad_tblr'][0] else 'top_left',contract['pad_rgb'][0])
    if contract!=expected:raise ValueError('Unsupported or inconsistent source/model transform')
    class_count=len(class_names)
    if (len(confidence)!=class_count or any(type(v) not in (int,float) or not isfinite(v) or not 0<v<1 for v in confidence)
            or type(nms) not in (int,float) or not isfinite(nms) or not 0<nms<1
            or len(limits)!=class_count or any(type(v) is not int or not 4<=v<=64 for v in limits)):
        raise ValueError('Bounded per-class confidence/NMS/capacity required')
    n=contract['positions'];size=contract['model_wh'][0]
    if tuple(raw.shape)!=(1,4+class_count,n):raise ValueError('Expected declared [1,4+classes,N] output; never shorten native rows')
    head=raw[0]
    if not np.all(np.isfinite(head)):raise ValueError('Nonfinite model output')
    if np.any(head[2:4,:]<0) or np.any(head[4:4+class_count,:]<0) or np.any(head[4:4+class_count,:]>1):
        raise ValueError('Expected decoded nonnegative sizes and sigmoid scores, not logits')
    if np.any(abs(head[0:4,:])>size*4):
        raise ValueError('Decoded coordinates exceed bounded native integer conversion range')
    # Native C clamps negative x/y without reducing width/height. Intersect in
    # model coordinates first, so pure-padding boxes cannot become image boxes.
    left=np.maximum(head[0,:]-head[2,:]/2,contract['pad_tblr'][2])
    right=np.minimum(head[0,:]+head[2,:]/2,contract['pad_tblr'][2]+contract['resized_wh'][0])
    top=np.maximum(head[1,:]-head[3,:]/2,contract['pad_tblr'][0])
    bottom=np.minimum(head[1,:]+head[3,:]/2,contract['pad_tblr'][0]+contract['resized_wh'][1])
    visible=(right>left)*(bottom>top)
    if native is None:
        import aidemo
        native=aidemo.yolov8_det_postprocess
    dtype=getattr(np,'float32',None)
    if dtype is None:dtype=np.float
    rows=[];counts=[];saturated=[]
    for class_id in range(class_count):
        # A fresh allocated array is contiguous, with all N rows retained even
        # when scores are zero. Native C reads a float* and computes N itself.
        values=np.zeros((n,5),dtype=dtype)
        values[:,0]=(left+right)/2-contract['pad_tblr'][2]
        values[:,1]=(top+bottom)/2-contract['pad_tblr'][0]
        values[:,2]=np.maximum(right-left,0);values[:,3]=np.maximum(bottom-top,0)
        values[:,4]=head[4+class_id,:]*visible
        if values.itemsize!=4 or tuple(values.strides)!=(20,4):
            raise ValueError('Native float32 contiguous Nx5 buffer required')
        if not np.all(np.isfinite(values)):raise ValueError('Native buffer conversion overflow')
        result=native(values,[360,640],[size,size],[360,640],1,
                      confidence[class_id],nms,limits[class_id])
        count=0
        if result is not None and len(result):
            if len(result)!=3 or len(result[0])!=len(result[1]) or len(result[0])!=len(result[2]):
                raise ValueError('Malformed native detection lists')
            count=len(result[0])
            for box,index,score in zip(result[0],result[1],result[2]):
                if int(index)!=0 or len(box)!=4 or not all(isfinite(float(v)) for v in box):
                    raise ValueError('Unexpected native class or coordinates')
                x,y,w,h=[float(v) for v in box];score=float(score)
                if not isfinite(score) or not 0<=score<=1:raise ValueError('Invalid native confidence')
                if w<=0 or h<=0:continue
                x0=max(0.0,x);y0=max(0.0,y);x1=min(640.0,x+w);y1=min(360.0,y+h)
                if x1<=x0 or y1<=y0:continue
                rows.append({'class_id':class_id,'class_name':class_names[class_id],
                    'bbox_xywh':[x0,y0,x1-x0,y1-y0],'score':score,
                    'coordinate_space':'SOURCE_IMAGE_PX','proposal_only':True,
                    'bbox_touches_source_edge':x0<=1 or y0<=1 or x1>=639 or y1>=359,
                    'geometry_verified':False,'identity_verified':False})
        counts.append(count);saturated.append(count>=limits[class_id])
    return {'status':'CANDIDATE_CAPACITY_REACHED' if any(saturated) else 'MODEL_PROPOSALS_ONLY',
        'complete':not any(saturated),'proposals':rows,'native_counts':counts,
        'per_class_saturated':saturated,'contract':contract,
        'maximum_materials_after_geometry':3,'physical_action':False,'official_points':None}
