"""Create and independently decode a task QR test card on the PC."""
import argparse
import hashlib
import json
from pathlib import Path
from core.task_parser import parse_task


def generate(payload, output):
    import cv2
    parse_task(payload)
    encoder=cv2.QRCodeEncoder_create()
    qr=encoder.encode(payload)
    qr=cv2.copyMakeBorder(qr,4,4,4,4,cv2.BORDER_CONSTANT,value=255)
    qr=cv2.resize(qr,None,fx=12,fy=12,interpolation=cv2.INTER_NEAREST)
    decoded,_,_=cv2.QRCodeDetector().detectAndDecode(qr)
    if decoded!=payload:
        raise ValueError('PC decoder did not recover generated task')
    ok,png=cv2.imencode('.png',qr)
    if not ok:
        raise ValueError('Could not encode QR card')
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(png.tobytes())
    return {'status':'PC_QR_ROUNDTRIP_PASS','payload':payload,'width_px':qr.shape[1],
            'printed_size_mm':None,'board_test':'HW_PENDING',
            'sha256':hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',default='156+123+516+231')
    parser.add_argument('--output',type=Path,default=Path('data/task_qr.png'))
    args=parser.parse_args()
    result=generate(args.task,args.output)
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
