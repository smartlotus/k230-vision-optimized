"""Create explicitly labeled file fixtures for native CanMV API checks."""
import json
from pathlib import Path
import cv2
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    dest=root/'data'/'board_fixtures'
    dest.mkdir(parents=True,exist_ok=True)
    phone=cv2.imread(str(root/'data'/'real_samples'/'IMG_20261031_210432.jpg'))
    small=cv2.resize(phone,(580,180))
    colors=np.full((320,640,3),255,np.uint8)
    colors[70:250,30:610]=small
    cv2.imwrite(str(dest/'phone-padded.jpg'),colors,[cv2.IMWRITE_JPEG_QUALITY,98])
    rings=np.full((320,640,3),255,np.uint8)
    layout=json.loads((dest/'synthetic-ring-layout.json').read_text(encoding='utf-8'))
    if layout.get('is_simulated') is not True:
        raise ValueError('Synthetic layout must be explicitly simulated')
    for gate in layout['center_gates'].values():
        x,y=gate['center']
        for radius in (24,36):
            cv2.circle(rings,(x,y),radius,(0,0,0),3)
    cv2.imwrite(str(dest/'synthetic-rings.jpg'),rings,[cv2.IMWRITE_JPEG_QUALITY,98])
    lane=np.full((320,640,3),255,np.uint8)
    lane[:,200:440]=128
    cv2.imwrite(str(dest/'synthetic-lane.jpg'),lane,[cv2.IMWRITE_JPEG_QUALITY,98])
    source=json.loads((root/'evidence'/'photos'/'photo_analysis.json').read_text(encoding='utf-8'))
    thresholds={k:v['LAB'] for k,v in source['threshold_suggestions'].items()}
    (dest/'thresholds.json').write_text(json.dumps(thresholds,indent=2),encoding='utf-8')
    (dest/'README.md').write_text('File fixtures only. phone-padded.jpg contains a resized user phone photo on padding. Other images are synthetic. No live camera or calibration claim.\n',encoding='utf-8')
    print(json.dumps({'fixture_dir':str(dest),'kind':'FILE_FIXTURE_NOT_CAMERA'}))


if __name__=='__main__':
    main()
