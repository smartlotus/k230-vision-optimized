"""Independent luminance-edge measurements for dark, merged color components.

Circle proposals are only search hints. Every radial direction must contain a
measured contrast edge; all measured points pass the existing ellipse gates.
"""
from math import cos, sin, pi, sqrt
from .ellipse_native import contour_fit

_EDGE_DIRECTIONS=tuple((cos(2*pi*i/48),sin(2*pi*i/48)) for i in range(48))
_EDGE_ORDER=tuple(range(0,48,6))+tuple(i for i in range(48) if i%6)


def _array(points):
    try:from ulab import numpy as np
    except ImportError:import numpy as np
    if hasattr(np,'float32'):return np.array(points,dtype=np.float32)
    return np.array(points)


def radial_edge_fit(gray,lab_mask,proposal,roi,min_area,max_area,cv):
    cx,cy,r=[float(v) for v in proposal]
    height,width=gray.shape
    if not 6<=r<=70 or cx-r-7<=0 or cy-r-7<=0 or cx+r+7>=width or cy+r+7>=height:
        return {'accepted':False,'reason':'EDGE_SEARCH_CLIPPED','center':None}
    # Pixel reads use rounded integers. A floating bound below width/height
    # can still round to the first out-of-range index. Check the extrema of
    # every possible d+/-3 sample before starting any radial direction.
    extent=int(r+4)+3
    if (int(round(cx-extent))<0 or int(round(cy-extent))<0
            or int(round(cx+extent))>=width or int(round(cy+extent))>=height):
        return {'accepted':False,'reason':'EDGE_SEARCH_CLIPPED','center':None}
    # A dense interior check rejects other colors before doing 48 edge searches.
    matched=total=0
    for iy in range(-4,5):
        for ix in range(-4,5):
            if ix*ix+iy*iy>16:continue
            total+=1
            if lab_mask[int(round(cy+iy*r*.16)),int(round(cx+ix*r*.16))]:matched+=1
    if matched/total<.95:return {'accepted':False,'reason':'EDGE_INTERIOR_COLOR','center':None}
    # Spread early checks across the whole boundary so broken background edges
    # fail sooner. Successful measurements retain all48 original ordered points.
    points=[None]*48;strengths=[0]*48
    for i in _EDGE_ORDER:
        ct,st=_EDGE_DIRECTIONS[i]
        best=-1;distance=None
        for d in range(int(r-4),int(r+4)+1):
            inner=int(gray[int(round(cy+st*(d-3))),int(round(cx+ct*(d-3)))])
            outer=int(gray[int(round(cy+st*(d+3))),int(round(cx+ct*(d+3)))])
            strength=abs(inner-outer)
            if strength>best:best=strength;distance=d
        if best<4:return {'accepted':False,'reason':'INCOMPLETE_LUMA_EDGE','center':None}
        strengths[i]=best;points[i]=[cx+ct*distance,cy+st*distance]
    fit=contour_fit(points,roi=[0,0,width,height],native_fit=cv.fitEllipse(_array(points)))
    if not fit['accepted']:return fit
    a,b=fit['semiaxes']
    area=pi*a*b
    if not min_area<=area<=max_area:return {'accepted':False,'reason':'AREA_LIMIT','center':None}
    # Check the full fitted interior, including holes, against native LAB pixels.
    interior=lab_mask.copy()
    interior[:,:]=0
    cv.ellipse(interior,tuple(int(round(v)) for v in fit['center']),
        (int(round(a)),int(round(b))),fit['angle_rad']*180/pi,0,360,(255,),-1)
    pixels=cv.countNonZero(interior)
    fill=cv.countNonZero(cv.bitwise_and(interior,lab_mask))/max(1,pixels)
    if fill<.90:return {'accepted':False,'reason':'EDGE_FULL_INTERIOR_COLOR','center':None}
    fit.update(center=[fit['center'][0]+roi[0],fit['center'][1]+roi[1]],
        backend='LUMA_EDGE_ELLIPSE',interior_fill=fill,
        edge_directions=48,min_edge_contrast=min(strengths))
    return fit


def extract_dark_edge_ellipses(gray,lab_mask,roi,components,min_area,max_area,existing=(),cv=None):
    if cv is None:import cv2 as cv
    height,width=gray.shape
    accepted=[];diagnostics=[];proposals=0
    for component in components:
        if component['fit']['accepted'] or component['fit']['reason'] in ('AREA_LIMIT','ROI_OR_IMAGE_CLIPPED'):
            continue
        bx,by,bw,bh=component['bbox'];x=max(0,bx-roi[0]-8);y=max(0,by-roi[1]-8)
        right=min(width,bx-roi[0]+bw+8);bottom=min(height,by-roi[1]+bh+8)
        if min(bw,bh)<16 or max(bw,bh)>160:continue
        sub=gray[y:bottom,x:right].copy()
        # CanMV exposes HoughCircles but some builds omit the OpenCV enum alias.
        circles=cv.HoughCircles(cv.GaussianBlur(sub,(5,5),1),getattr(cv,'HOUGH_GRADIENT',3),
            1,8,param1=60,param2=18,minRadius=max(6,int(min(bw,bh)*.20)),maxRadius=min(70,int(min(bw,bh)*.65)))
        if circles is None:continue
        rows=circles.tolist()
        if not rows:continue
        if isinstance(rows[0][0],(list,tuple)):rows=rows[0]
        seen=[]
        for row in rows:
            if any((row[0]-q[0])**2+(row[1]-q[1])**2<12**2 for q in seen):continue
            seen.append(row)
            if len(seen)>6:break
            # Hough's center/radius is a proposal, not the measured edge. A
            # bounded local refinement handles dark top faces whose sidewall
            # shifts the proposal a few pixels.
            candidates=[]
            for dx,dy,dr in ((0,0,0),(-2,0,-4),(0,-2,-4),(-2,-2,-4)):
                    proposals+=1
                    if proposals>128:return {'accepted':[],'components':diagnostics,'complete':False,'reason':'EDGE_PROPOSAL_LIMIT'}
                    fit=radial_edge_fit(gray,lab_mask,[row[0]+x+dx,row[1]+y+dy,row[2]+dr],roi,min_area,max_area,cv)
                    if fit['accepted']:
                        candidates.append(fit)
            fit=min(candidates,key=lambda f:(f.get('rms_px',999)+f.get('max_error_px',999),f.get('max_error_px',999))) if candidates else {'accepted':False,'reason':'EDGE_NO_VALID_PROPOSAL','center':None}
            diagnostics.append({'fit':fit,'source_bbox':component['bbox'],'backend':'LUMA_EDGE_ELLIPSE'})
            if not fit['accepted']:continue
            if any(sqrt((fit['center'][0]-f['center'][0])**2+(fit['center'][1]-f['center'][1])**2)<max(6,.25*min(f['semiaxes'])) for f in list(existing)+accepted):
                continue
            accepted.append(fit)
    return {'accepted':accepted,'components':diagnostics,'complete':True,'reason':'FRESH_LUMA_EDGES'}
