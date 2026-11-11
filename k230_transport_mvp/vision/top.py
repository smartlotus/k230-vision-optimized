"""Explicit pixel-shape checks for a separately sampled top-face profile."""
from .common import finite


def validate_circle_parameters(value):
    allowed={'threshold','x_margin','y_margin','r_margin','r_min','r_max','r_step','x_stride','y_stride'}
    required=('threshold','r_min','r_max','r_step')
    if not isinstance(value,dict) or set(value)-allowed or any(k not in value for k in required):
        raise ValueError('Explicit native circle parameters required')
    if any(type(v) is not int for v in value.values()):
        raise ValueError('Circle parameters must be integers')
    if (value['threshold']<=0 or value['r_min']<=0 or value['r_max']<value['r_min'] or value['r_step']<=0
            or any(value.get(k,1)<=0 for k in ('x_stride','y_stride'))
            or any(value.get(k,0)<0 for k in ('x_margin','y_margin','r_margin'))):
        raise ValueError('Invalid circle threshold, radius, stride or margin')
    return dict(value)


def validate_circle_config(cfg):
    ids=cfg.get('circle_refine_ids',())
    if not isinstance(ids,(list,tuple)) or any(type(v) is not int or v not in (1,2,3,4,5,6) for v in ids):
        raise ValueError('Circle color IDs must be integers1..6')
    tolerance=cfg.get('circle_center_tolerance_px',6)
    if not finite(tolerance) or tolerance<=0:raise ValueError('Positive circle tolerance required')
    if ids:validate_circle_parameters(cfg.get('circle_parameters'))
    if cfg.get('circle_boundary') is not None:validate_boundary_parameters(cfg['circle_boundary'])


def validate_boundary_parameters(value):
    keys={'inner_ratio','outer_ratio','min_l_contrast','min_supported'}
    if not isinstance(value,dict) or set(value)!=keys:raise ValueError('Explicit circle boundary parameters required')
    if (any(not finite(value[k]) for k in ('inner_ratio','outer_ratio','min_l_contrast'))
            or not 0<value['inner_ratio']<1<value['outer_ratio']<=2
            or not 0<value['min_l_contrast']<=100
            or type(value['min_supported']) is not int or not 1<=value['min_supported']<=8):
        raise ValueError('Invalid circle boundary parameters')


def circle_boundary_support(image,rgb_to_lab,x,y,radius,parameters):
    """Measured radial contrast, not proof of object identity or a hand detector."""
    validate_boundary_parameters(parameters)
    directions=((1000,0),(707,707),(0,1000),(-707,707),(-1000,0),(-707,-707),(0,-1000),(707,-707))
    deltas=[]
    for dx,dy in directions:
        xi=x+int(round(radius*parameters['inner_ratio']*dx/1000))
        yi=y+int(round(radius*parameters['inner_ratio']*dy/1000))
        xo=x+int(round(radius*parameters['outer_ratio']*dx/1000))
        yo=y+int(round(radius*parameters['outer_ratio']*dy/1000))
        if not (0<=xi<image.width() and 0<=yi<image.height() and 0<=xo<image.width() and 0<=yo<image.height()):
            return {'verified':False,'reason':'BOUNDARY_OUT_OF_IMAGE','deltas':deltas,'supported':0}
        deltas.append(abs(rgb_to_lab(*image.get_pixel(xo,yo))[0]-rgb_to_lab(*image.get_pixel(xi,yi))[0]))
    supported=sum(d>=parameters['min_l_contrast'] for d in deltas)
    return {'verified':supported>=parameters['min_supported'],'deltas':deltas,'supported':supported}


def circle_surfaces(circles,target_id,center_tolerance=6):
    """Cluster only interior-color-confirmed circles, never infer a center from history."""
    if not finite(center_tolerance) or center_tolerance<=0:raise ValueError('Positive circle tolerance required')
    groups=[]
    valid=[c for c in circles if c.get('interior_verified') is True
           and all(finite(c.get(k)) for k in ('x','y','r')) and c['r']>0]
    valid.sort(key=lambda c:(c['x'],c['y'],c['r']))
    for circle in valid:
        group=next((g for g in groups if all((circle['x']-c['x'])**2+(circle['y']-c['y'])**2<=center_tolerance**2 for c in g)),None)
        if group is None:groups.append([circle])
        else:group.append(circle)
    result=[]
    for group in groups:
        u=sum(c['x'] for c in group)/len(group);v=sum(c['y'] for c in group)/len(group)
        radius=sum(c['r'] for c in group)/len(group)
        result.append({'target_id':target_id,'bbox':(u-radius,v-radius,2*radius,2*radius),
            'circle_support':len(group),'score_hint':min(100,len(group)*25)})
    return result


def filter_surfaces(candidates,shape):
    required=('min_axis_ratio','max_axis_ratio','min_density')
    if not isinstance(shape,dict) or any(not finite(shape.get(k)) for k in required):
        raise ValueError('Measured top-surface shape limits required')
    low,high,density=(shape[k] for k in required)
    if not 0<low<=high or not 0<density<=1:
        raise ValueError('Invalid top-surface shape limits')
    kept=[]
    for candidate in candidates:
        override=shape.get('per_color',{}).get(candidate.get('target_id'),
            shape.get('per_color',{}).get(str(candidate.get('target_id')),{}))
        limits={k:override.get(k,shape[k]) for k in required}
        if any(not finite(v) for v in limits.values()):raise ValueError('Invalid per-color surface limits')
        low,high,density=(limits[k] for k in required)
        if not 0<low<=high or not 0<density<=1:raise ValueError('Invalid per-color surface limits')
        x,y,w,h=candidate['bbox']
        pixels=candidate.get('pixels')
        if w>0 and h>0 and finite(pixels) and low<=w/h<=high and pixels/(w*h)>=density:
            kept.append(candidate)
    return kept
