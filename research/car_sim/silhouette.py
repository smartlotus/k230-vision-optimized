"""Black block on light station paper, using the measured rotational profile."""
import math

import cv2
import numpy as np
from scipy.optimize import minimize


def black_profile(image, window, camera_z, top_z):
    u0,v0,u1,v1=window
    fx=18.595373/36*640
    levels=((-.060,.025),(-.052,.025),(-.008,.015),(0.,.015))
    if not .15<camera_z-top_z<.45:
        return None
    radius=fx*.025/(camera_z-top_z+.052)
    pad=math.ceil(2.1*radius)
    left,up=max(0,u0-pad),max(0,v0-pad)
    crop=image[up:v1+pad,left:u1+pad]
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    # Isolate achromatic black material on the brighter station strip. Remove
    # the narrow concentric printing before fitting the much wider block.
    chroma=np.ptp(crop.astype(np.int16),axis=2)
    threshold=min(104,.59*float(np.percentile(gray,83)))
    ink=np.uint8((gray<threshold)&(chroma<24))*255
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    opened=cv2.morphologyEx(ink,cv2.MORPH_OPEN,kernel)
    count,regions,stats,centers=cv2.connectedComponentsWithStats(opened)
    theta=np.linspace(0,2*np.pi,48,endpoint=False)
    direction=np.stack([np.cos(theta),np.sin(theta)],axis=1)
    output=[]
    for idx in range(1,count):
        if not .68*math.pi*radius**2<stats[idx,4]<2.4*math.pi*radius**2:
            continue
        cx,cy=centers[idx]+[left,up]
        # A starting estimate near the base. The perspective-correct fit
        # evaluates the silhouette at all four radii and axial heights.
        ratio=(camera_z-top_z+.052)/(camera_z-top_z)
        guess=np.array([320+(cx-320)*ratio,180+(cy-180)*ratio])
        target=regions==idx
        def iou(point):
            vertices=[]
            for dz,r in levels:
                distance=camera_z-top_z-dz
                center=np.array([320,180])+(point-[320,180])*(camera_z-top_z)/distance
                vertices.extend(center-[left,up]+direction*(r*fx/distance))
            polygon=cv2.convexHull(np.int32(np.rint(vertices)))
            model=np.zeros_like(gray)
            cv2.fillConvexPoly(model,polygon,1)
            inside=model!=0
            return np.count_nonzero(inside & target)/np.count_nonzero(inside | target)
        opt=minimize(lambda p:-iou(p),guess,method="Nelder-Mead",options={
            "initial_simplex":np.array([guess,guess+[3.,0],guess+[0,3.]]),"xatol":.12,"fatol":.0001,
            "maxiter":60})
        u,v=map(float,opt.x)
        quality=iou(opt.x)
        if u0<=u<=u1 and v0<=v<=v1 and quality>=.86:
            output.append({"center":[u,v],"silhouette_iou":quality})
    return output[0] if len(output)==1 else None
