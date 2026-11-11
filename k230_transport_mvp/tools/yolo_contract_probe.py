"""Native synthetic postprocessing checks; no task model or recognition claim."""


def probe():
    import time
    from ulab import numpy as np
    from vision.yolo_candidates import input_contract,decode_two_class_proposals
    rows=[]
    for size in (224,320):
        for padding,pad_value in (('center',114),('top_left',128)):
            contract=input_contract(size,padding,pad_value);gain=contract['gain'];pad=contract['pad_tblr'][0]
            raw=np.zeros((1,6,contract['positions']),dtype=np.float)
            raw[0,0,0]=340*gain;raw[0,1,0]=190*gain+pad
            raw[0,2,0]=80*gain;raw[0,3,0]=60*gain
            raw[0,4,0]=.9;raw[0,5,0]=.85
            start=time.ticks_us();result=decode_two_class_proposals(raw,contract,np=np)
            elapsed=time.ticks_diff(time.ticks_us(),start)/1000
            if result['native_counts']!=[1,1] or not result['complete']:raise ValueError('Cross-class candidate loss')
            for item in result['proposals']:
                if max(abs(a-b) for a,b in zip(item['bbox_xywh'],[300,160,80,60]))>2:
                    raise ValueError('Wrong source coordinate mapping')
            rows.append({'model_size':size,'padding':padding,'native_counts':result['native_counts'],
                         'boxes':[p['bbox_xywh'] for p in result['proposals']],
                         'elapsed_ms':elapsed,'itemsize':raw.itemsize,'proposal_only':True})
    contract=input_contract(320);raw=np.zeros((1,6,contract['positions']),dtype=np.float)
    for i in range(4):
        raw[0,0,i]=(80+i*140)*contract['gain'];raw[0,1,i]=120*contract['gain']+contract['pad_tblr'][0]
        raw[0,2,i]=30*contract['gain'];raw[0,3,i]=30*contract['gain'];raw[0,4,i]=.9
    result=decode_two_class_proposals(raw,contract,limits=(4,16),np=np)
    if result['complete'] or len(result['proposals'])<4:raise ValueError('Capacity/three-material guard lost candidates')
    capacity_status=result['status'];capacity_counts=result['native_counts']
    raw=np.zeros((1,6,contract['positions']),dtype=np.float)
    raw[0,0,0]=160;raw[0,1,0]=30;raw[0,2,0]=40;raw[0,3,0]=20;raw[0,4,0]=.9
    padding=decode_two_class_proposals(raw,contract,np=np)
    if padding['native_counts']!=[0,0]:raise ValueError('Pure padding became a source object')
    raw[0,1,0]=80;raw[0,3,0]=40
    crossing=decode_two_class_proposals(raw,contract,np=np)
    if len(crossing['proposals'])!=1 or not crossing['proposals'][0]['bbox_touches_source_edge']:
        raise ValueError('Crossing source boundary lost clipping flag')
    if max(abs(a-b) for a,b in zip(crossing['proposals'][0]['bbox_xywh'],[280,0,80,60]))>2:
        raise ValueError('Crossing source boundary inflated its visible box')
    return {'scope':'NATIVE_SYNTHETIC_POSTPROCESS_CONTRACT','rows':rows,
        'capacity_status':capacity_status,'capacity_counts':capacity_counts,
        'padding_native_counts':padding['native_counts'],'crossing_proposals':crossing['proposals'],
        'status':'PASS_CONTRACT_ONLY','task_model':False,'recognition_verified':False,'physical_action':False}


if __name__=='__main__':
    import sys,json
    if '/sdcard/transport_mvp' not in sys.path:sys.path.insert(0,'/sdcard/transport_mvp')
    try:result=probe()
    except Exception as error:
        result={'scope':'NATIVE_SYNTHETIC_POSTPROCESS_CONTRACT','status':'FAILED','error':str(error),'recognition_verified':False}
        sys.print_exception(error)
    print('YOLO_CONTRACT_PROBE:'+json.dumps(result))
