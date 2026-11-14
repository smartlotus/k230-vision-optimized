"""Portable MicroPython/CPython 50-channel decoder; no hardware side effects.

No model/threshold is production-approved. read_value(position,channel) must
read the declared [1,N,50] float32 tensor. Caller validates shape/provenance.
CPU latency and target ulab tensor access remain unmeasured.
"""
import math


def _sigmoid(value):
    if value>=0:return 1.0/(1.0+math.exp(-value))
    exp=math.exp(value);return exp/(1.0+exp)


def _iou(one,two):
    area=max(0.0,min(one[2],two[2])-max(one[0],two[0]))*max(0.0,min(one[3],two[3])-max(one[1],two[1]))
    a=max(0.0,one[2]-one[0])*max(0.0,one[3]-one[1])
    b=max(0.0,two[2]-two[0])*max(0.0,two[3]-two[1])
    return area/max(a+b-area,1e-9)


def _decode_dfl(values,stride):
    output=[]
    for side in range(4):
        logits=values[6+side*8:14+side*8];peak=max(logits)
        exps=[math.exp(x-peak) for x in logits];total=sum(exps)
        output.append(sum(i*x for i,x in enumerate(exps))/total*stride)
    return output


def decode(read_value,width=320,height=192,positions=1275,channels=50,score_threshold=.25,nms_iou=.5):
    if channels!=50 or (width,height) not in ((224,128),(320,192),(384,224)):
        raise ValueError('Unregistered50-channel task contract')
    expected=sum(((width+s-1)//s)*((height+s-1)//s) for s in (8,16,32,64))
    if positions!=expected or not 0<score_threshold<1 or not 0<nms_iou<1:
        raise ValueError('Tensor locations/threshold differ from declared contract')
    gain=min(width/640.0,height/360.0);rw=int(round(640*gain));rh=int(round(360*gain))
    top=(height-rh)//2;left=(width-rw)//2
    logit_threshold=math.log(score_threshold/(1.0-score_threshold));candidates=[];index=0
    for stride in (8,16,32,64):
        for row in range((height+stride-1)//stride):
            for col in range((width+stride-1)//stride):
                classes=[float(read_value(index,c)) for c in range(6)]
                if not all(math.isfinite(x) for x in classes):raise ValueError('Nonfinite declared class logits')
                best=max(classes)
                if best>=logit_threshold:
                    values=classes+[float(read_value(index,c)) for c in range(6,50)]
                    if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite selected candidate tensor')
                    distances=_decode_dfl(values,stride);px=col*stride;py=row*stride
                    box=[px-distances[0],py-distances[1],px+distances[2],py+distances[3]]
                    visible=[max(left,box[0]),max(top,box[1]),min(left+rw,box[2]),min(top+rh,box[3])]
                    if visible[2]>visible[0] and visible[3]>visible[1]:
                        candidates.append({'index':index,'score':_sigmoid(best),'color_id':classes.index(best)+1,
                            'box_model':box,'nms_box':visible,'point':[px,py],'stride':stride,'values':values})
                index+=1
    candidates.sort(key=lambda c:(-c['score'],c['index']))
    selected=[]
    while candidates and len(selected)<200:
        candidate=candidates.pop(0);selected.append(candidate)
        candidates=[c for c in candidates if _iou(candidate['nms_box'],c['nms_box'])<=nms_iou]
    detections=[]
    for candidate in selected[:5]:
        values=candidate['values'];px,py=candidate['point'];stride=candidate['stride']
        unclip=[(candidate['box_model'][0]-left)/gain,(candidate['box_model'][1]-top)/gain,
                (candidate['box_model'][2]-left)/gain,(candidate['box_model'][3]-top)/gain]
        box=[(candidate['nms_box'][0]-left)/gain,(candidate['nms_box'][1]-top)/gain,
             (candidate['nms_box'][2]-left)/gain,(candidate['nms_box'][3]-top)/gain]
        physical=[(px+values[38]*stride-left)/gain,(py+values[39]*stride-top)/gain]
        conic=[(px+values[48]*stride-left)/gain,(py+values[49]*stride-top)/gain]
        norm=math.sqrt(values[44]*values[44]+values[45]*values[45]);ellipse=None
        if max(abs(values[42]),abs(values[43]))<20 and norm>1e-6:
            major=math.exp(values[42])*stride/gain;minor=math.exp(values[43])*stride/gain
            theta=.5*math.atan2(values[44],values[45])
            if minor>major:major,minor=minor,major;theta+=math.pi/2
            theta=(theta+math.pi/2)%math.pi-math.pi/2
            ellipse={'conic_center_xy':conic,'semi_major':major,'semi_minor':minor,
                'orientation_deg':theta*180/math.pi,'orientation_valid':minor/major<.98,
                'geometry_confidence':_sigmoid(values[46])}
        detections.append({'score':candidate['score'],'color_id':candidate['color_id'],
            'color_probabilities':[_sigmoid(v) for v in values[:6]],'bbox_xyxy_visible':box,
            'bbox_xyxy_unclipped':unclip,'center_xy_projected_3d':physical,
            'center_valid_probability':_sigmoid(values[40]),'center_visible_probability':_sigmoid(values[41]),
            'ellipse':ellipse,'visibility_quality':_sigmoid(values[47]),
            'boundary_candidate':any(abs(x-y)>1e-4 for x,y in zip(box,unclip))})
    count=len(detections);uncertain=any(d['boundary_candidate'] for d in detections)
    capacity='EMPTY' if count==0 else 'OK' if count<=3 else 'OVER_CAPACITY'
    return {'detections':detections,'count':count,'post_nms_total_before_five_limit':len(selected),
        'observed_capacity_state':capacity,'capacity_state':'UNCERTAIN' if uncertain and count<4 else capacity,
        'uncertain':uncertain,'thresholds':{'score':score_threshold,'class_agnostic_nms_iou':nms_iou}}
