"""Coarse concentric-edge target geometry; no numbered identity or score."""
from math import sqrt,cos,sin,pi,isfinite
from .ellipse_native import contour_fit


def group_concentric(fits, center_tolerance=8, radius_gap=6, min_levels=3):
    """Collapse stroke edges before requiring distinct nested ellipse levels."""
    groups=[]
    for fit in sorted(fits,key=lambda f:(f['center'][0],f['center'][1],f['semiaxes'][0])):
        cx,cy=fit['center']
        group=next((g for g in groups if all((cx-f['center'][0])**2+(cy-f['center'][1])**2
                    <=center_tolerance**2 for f in g)),None)
        if group is None:groups.append([fit])
        else:group.append(fit)
    targets=[]
    for group in groups:
        levels=[]
        for fit in sorted(group,key=lambda f:f.get('rms_px',24-f.get('paper_stroke_directions',0))):
            radius=sum(fit['semiaxes'])/2
            if all(abs(radius-sum(f['semiaxes'])/2)>=radius_gap for f in levels):
                levels.append(fit)
        if len(levels)<min_levels:continue
        center=[sum(f['center'][i] for f in levels)/len(levels) for i in (0,1)]
        targets.append({'center':center,'support':len(levels),
            'boundaries':sorted(levels,key=lambda f:f['semiaxes'][0]),
            'center_spread_px':max(sqrt((f['center'][0]-center[0])**2+(f['center'][1]-center[1])**2) for f in levels),
            'identity_verified':False})
    return targets




def paper_stroke_support(gray,roi,fit):
    """Dark thin-stroke evidence from pixels only, independent of material color.

    A paper line is darker than both sides of its boundary. A solid material
    disk edge typically has only one bright side and must not count as a paper
    radius level. This is a coarse appearance gate, not verified target identity.
    """
    height,width=gray.shape;cx=fit['center'][0]-roi[0];cy=fit['center'][1]-roi[1]
    a,b=fit['semiaxes'];ct,st=cos(fit['angle_rad']),sin(fit['angle_rad'])
    support=0
    for i in range(24):
        u,v=cos(2*pi*i/24),sin(2*pi*i/24)
        px,py=cx+ct*a*u-st*b*v,cy+st*a*u+ct*b*v
        nx,ny=ct*u/a-st*v/b,st*u/a+ct*v/b
        norm=sqrt(nx*nx+ny*ny);nx/=norm;ny/=norm
        points=[(int(round(px+nx*d)),int(round(py+ny*d))) for d in (-6,-2,0,2,6)]
        if any(x<0 or y<0 or x>=width or y>=height for x,y in points):continue
        values=[int(gray[y,x]) for x,y in points]
        if min(values[0],values[4])-min(values[1:4])>=4:support+=1
    return support


def detect_paper_targets(gray,roi,cv=None,pixel_scale=2):
    """Independent paper path: only its own grayscale ROI and stroke geometry."""
    result=extract_ring_edges(gray,roi,cv=cv,pixel_scale=pixel_scale)
    if not result['complete']:return result
    paper=[]
    for fit in result['boundaries']:
        support=paper_stroke_support(gray,roi,fit)
        if support>=15:
            row=dict(fit);row['paper_stroke_directions']=support;paper.append(row)
    groups=group_concentric(paper,min_levels=2)
    result.update(groups=groups,boundaries=paper,
        reason='INDEPENDENT_PAPER_GEOMETRY' if groups else 'NO_INDEPENDENT_PAPER_TARGET')
    return result


def detect_paper_circle_proposals(gray,roi,cv=None,pixel_scale=2):
    """Opt-in gray-only paper proposals; never a complete boundary/ID claim.

    Native Hough voting handles fragmented strokes. Every retained radius must
    independently show thin dark ink on at least 18/24 directions, fit inside
    the ROI with sampling margin and form two distinct concentric levels.
    No material detections or colors are inputs to this path.
    """
    if cv is None:import cv2 as cv
    height,width=gray.shape
    scale=2 if pixel_scale==2 and width%2==0 and height%2==0 else 1
    working=cv.resize(gray,(width//2,height//2)) if scale==2 else gray
    working=cv.GaussianBlur(working,(3,3),0)
    found=cv.HoughCircles(working,getattr(cv,'HOUGH_GRADIENT',3),1,2,param1=30,param2=28,
                         minRadius=16//scale,maxRadius=80//scale)
    raw=[] if found is None else found.tolist()
    if len(raw)==1 and raw[0] and isinstance(raw[0][0],(list,tuple)):raw=raw[0]
    edges=cv.Canny(working,10,30)
    if scale==2:edges=cv.resize(edges,(width,height),interpolation=getattr(cv,'INTER_NEAREST',0))
    result={'complete':True,'reason':'NO_PAPER_CIRCLE_PROPOSAL','groups':[],'boundaries':[],
            'edges':edges,'proposal_only':True,'raw_circle_count':len(raw),'pixel_scale':scale}
    if len(raw)>128:
        result.update(complete=False,reason='PAPER_CIRCLE_PROPOSAL_LIMIT');return result
    paper=[]
    for values in raw:
        if len(values)!=3 or not all(isfinite(float(v)) for v in values):continue
        x,y,r=[float(v) for v in values];x=(x+.5)*scale-.5;y=(y+.5)*scale-.5;r*=scale
        if not 16<=r<=80 or min(x-r,y-r,width-1-x-r,height-1-y-r)<6:continue
        fit={'center':[roi[0]+x,roi[1]+y],'semiaxes':[r,r],'angle_rad':0,
             'backend':'NATIVE_CV2_HOUGH_PAPER_STROKE','proposal_only':True,
             'boundary_complete_verified':False,'identity_verified':False}
        support=paper_stroke_support(gray,roi,fit)
        if support>=18:
            fit['paper_stroke_directions']=support;paper.append(fit)
    # Proposal sampling tolerates radial offsets around a dark stroke; do not
    # let its opposite edges or nearby duplicate votes count as two levels.
    groups=group_concentric(paper,min_levels=2,radius_gap=10)
    for group in groups:group.update(proposal_only=True,geometry_basis='GRAY_STROKE_CIRCLE_PROPOSAL')
    result.update(groups=groups,boundaries=paper,
                  reason='PAPER_STROKE_CIRCLE_PROPOSAL' if groups else 'NO_PAPER_CIRCLE_PROPOSAL')
    return result


def associate_material_target(materials,paper_targets):
    """Consume already detected objects; association never changes detection."""
    if len(materials)!=1:
        return {'reason':'UNIQUE_MATERIAL_REQUIRED','material':None,'target':None}
    material=materials[0];matches=[]
    for target in paper_targets:
        if target.get('proposal_only') or any(f.get('proposal_only') or f.get('boundary_complete_verified') is False for f in target['boundaries']):
            continue
        outer=target['boundaries'][-1]
        dx,dy=material['center'][0]-outer['center'][0],material['center'][1]-outer['center'][1]
        ct,st=cos(outer['angle_rad']),sin(outer['angle_rad'])
        u,v=ct*dx+st*dy,-st*dx+ct*dy
        a,b=outer['semiaxes']
        if (u/a)**2+(v/b)**2<1:matches.append(target)
    if len(matches)!=1:
        return {'reason':'UNIQUE_OVERLAPPING_PAPER_TARGET_REQUIRED','material':material,'target':None}
    return {'reason':'ROUGH_UNNUMBERED_ASSOCIATION','material':material,'target':matches[0],
            'association_basis':'UNIQUE_IMAGE_CONTAINMENT_ONLY'}


def extract_ring_edges(gray, roi, cv=None, pixel_scale=1,edge_thresholds=(10,30)):
    """One native edge pass; validate full contours without trimming boundaries."""
    if cv is None:import cv2 as cv
    height,width=gray.shape
    scale=2 if pixel_scale==2 and width%2==0 and height%2==0 else 1
    working=cv.resize(gray,(width//2,height//2)) if scale==2 else gray
    if (not isinstance(edge_thresholds,(list,tuple)) or len(edge_thresholds)!=2
            or any(type(v) is not int for v in edge_thresholds)
            or not 1<=edge_thresholds[0]<edge_thresholds[1]<=255):
        raise ValueError('ordered integer edge thresholds in 1..255 required')
    edges=cv.Canny(cv.GaussianBlur(working,(3,3),0),*edge_thresholds)
    contours,_=cv.findContours(edges,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
    if len(contours)>1024:
        if scale==2:edges=cv.resize(edges,(width,height),interpolation=getattr(cv,'INTER_NEAREST',0))
        return {'complete':False,'reason':'RING_CONTOUR_LIMIT','groups':[],'boundaries':[],'edges':edges,
                'contours':len(contours),'pixel_scale':scale}
    fits=[]
    for contour in contours:
        if not 40<=len(contour)<=4096:continue
        points=contour.tolist()
        if points and isinstance(points[0][0],(list,tuple)):points=[p[0] for p in points]
        fit=contour_fit(points,roi=(0,0,width//scale,height//scale),native_fit=cv.fitEllipse(contour))
        if fit['accepted'] and fit['semiaxes'][1]>=6:
            fit['center']=[roi[i]+(fit['center'][i]+.5)*scale-.5 for i in (0,1)]
            fit['semiaxes']=[v*scale for v in fit['semiaxes']]
            fit['rms_px']*=scale;fit['max_error_px']*=scale
            fit['pixel_scale']=scale
            fits.append(fit)
    groups=group_concentric(fits)
    if scale==2:edges=cv.resize(edges,(width,height),interpolation=getattr(cv,'INTER_NEAREST',0))
    return {'complete':True,'reason':'CONCENTRIC_EDGE_GEOMETRY' if groups else 'NO_CONCENTRIC_TARGET',
            'groups':groups,'boundaries':fits,'edges':edges,'contours':len(contours),'pixel_scale':scale}
