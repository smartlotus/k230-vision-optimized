"""Analyze supplied phone photos. Suggestions are never applied to K230 config."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def analyze(source_dir, output_dir):
    import cv2
    import numpy as np
    manifest = json.loads((ROOT/'data'/'photo_samples.json').read_text(encoding='utf-8'))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources, arrays, images = [], {}, []
    for sample in manifest['samples']:
        path = Path(source_dir)/sample['file']
        if not path.is_file():
            raise ValueError('Missing source image: ' + str(path))
        bgr = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('Cannot read image: ' + str(path))
        bgr = cv2.resize(bgr, (1024, round(bgr.shape[0]*1024/bgr.shape[1])))
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(float)
        lab[:, :, 0] *= 100/255
        lab[:, :, 1:] -= 128
        for color, (u, v) in sample['patches'].items():
            x, y = round(u*bgr.shape[1]), round(v*bgr.shape[0])
            patch = lab[max(0,y-8):y+9, max(0,x-8):x+9].reshape(-1,3)
            arrays.setdefault(color, []).append(patch)
        sources.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'kind': 'REAL_PHONE_PHOTO', 'camera_id': None})
        images.append((sample, bgr, lab))
    suggestions = {}
    for color, patches in arrays.items():
        values = np.concatenate(patches)
        lower = np.floor(np.percentile(values, 1, axis=0) - [6,8,8])
        upper = np.ceil(np.percentile(values, 99, axis=0) + [6,8,8])
        lower = np.maximum(lower, [0,-128,-128]).astype(int)
        upper = np.minimum(upper, [100,127,127]).astype(int)
        suggestions[color] = {'LAB': [int(lower[0]),int(upper[0]),int(lower[1]),int(upper[1]),int(lower[2]),int(upper[2])],
                              'median_LAB': np.median(values,axis=0).round(2).tolist(),
                              'pixel_samples':len(values), 'board_calibrated':False}
    detections = []
    for sample,bgr,lab in images:
        result = []
        overlay = bgr.copy()
        h,w = bgr.shape[:2]
        for color, entry in suggestions.items():
            l0,l1,a0,a1,b0,b1=entry['LAB']
            mask = np.all((lab >= [l0,a0,b0]) & (lab <= [l1,a1,b1]),axis=2).astype('uint8')*255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
            count,_,stats,_ = cv2.connectedComponentsWithStats(mask)
            candidates = [row for row in stats[1:] if row[4]>h*w*0.008 and row[2]>15 and row[3]>15]
            if not candidates:
                result.append({'color_id':int(color),'status':'NOT_FOUND'})
                continue
            x,y,bw,bh,area = [int(n) for n in max(candidates,key=lambda row:row[4])]
            u,v=x+bw/2,y+bh/2
            result.append({'color_id':int(color),'status':'OK','bbox_px':[x,y,bw,bh],
                           'center_px':[u,v], 'mask_pixels':area,
                           'coord_frame':'IMAGE_PX','x_mm':None,'y_mm':None})
            cv2.rectangle(overlay,(x,y),(x+bw,y+bh),(255,255,255),2)
            cv2.circle(overlay,(round(u),round(v)),4,(0,0,0),-1)
            cv2.putText(overlay,manifest['colors'][color],(x,max(20,y-4)),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),2)
        cv2.putText(overlay,'PHONE PHOTO / OFFLINE / UNCALIBRATED',(8,h-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
        output=output_dir/(Path(sample['file']).stem+'_annotated.jpg')
        ok, encoded=cv2.imencode('.jpg',overlay,[cv2.IMWRITE_JPEG_QUALITY,88])
        if not ok:
            raise ValueError('Cannot encode annotation')
        output.write_bytes(encoded.tobytes())
        detections.append({'file':sample['file'],'detections':result,'annotation':output.name,
                           'counts':{'observed_colors':sum(d['status']=='OK' for d in result),'expected_visible_colors':5}})
    report={'status':'OFFLINE_ANALYSIS','is_simulated':False,'board_calibrated':False,
            'scope':'3 supplied phone images; patches and evaluation use same photos. Not an independent accuracy benchmark.',
            'warning':'Approximate OpenCV Lab conversion; GC2093 exposure/ISP and RGB565 quantization require new samples. Black(5) absent.',
            'annotation_method':manifest['annotation_method'], 'sources':sources,
            'missing_colors':[5], 'threshold_suggestions':suggestions, 'images':detections}
    (output_dir/'photo_analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',required=True,type=Path)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'evidence'/'photos')
    args=parser.parse_args()
    try:
        report=analyze(args.source_dir,args.output_dir)
    except (ValueError,ImportError) as exc:
        parser.exit(2,str(exc)+'\n')
    print(json.dumps({'status':report['status'],'images':[x['counts'] for x in report['images']],
                      'missing_colors':report['missing_colors'],'board_calibrated':False}))
