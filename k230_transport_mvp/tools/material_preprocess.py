"""Host reference for the declared material pilot input, not native AI2D proof."""
import hashlib
import cv2
import numpy as np
from vision.yolo_candidates import input_contract


def rgb565_to_rgb(raw):
    if len(raw)!=640*360*2:raise ValueError('Complete640x360 RGB565 bytes required')
    image=np.frombuffer(raw,np.uint8).reshape(360,640,2)
    return cv2.cvtColor(image,cv2.COLOR_BGR5652RGB)


def model_uint8(rgb,size=320):
    if rgb.shape!=(360,640,3) or rgb.dtype!=np.uint8:raise ValueError('640x360 RGB uint8 source required')
    contract=input_contract(size,'center',114)
    width,height=contract['resized_wh'];top,bottom,left,right=contract['pad_tblr']
    resized=cv2.resize(rgb,(width,height),interpolation=cv2.INTER_LINEAR)
    canvas=np.full((size,size,3),114,np.uint8)
    canvas[top:size-bottom,left:size-right]=resized
    return np.ascontiguousarray(canvas.transpose(2,0,1)[None,...]),contract


def float_reference(tensor):
    if tensor.dtype!=np.uint8:raise ValueError('Normalize original uint8 exactly once')
    return tensor.astype(np.float32)/255.0


def tensor_receipt(tensor,contract):
    return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'layout':'NCHW',
        'bytes':tensor.nbytes,'sha256':hashlib.sha256(tensor.tobytes()).hexdigest(),
        'contract':contract,'backend':'HOST_OPENCV_INTER_LINEAR_REFERENCE',
        'native_ai2d_equivalence_verified':False}
