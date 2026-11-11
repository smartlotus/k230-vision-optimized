"""Circular vertical projection containment, independent of image fitting.

Inputs must describe the complete material projection and measured concentric
zone boundaries in one common plane/unit. A top-face center or hand drawing
does not establish these inputs. Results are zone candidates, never scores.
"""
from math import hypot
from .common import finite


def classify_circular_projection(dx, dy, material_radius, zone_outer_radii, *,
                                 uncertainty=0, geometry_verified=False):
    """Bound the farthest projected edge from the target center.

    The one supplied uncertainty must bound the combined radial error of the
    center offset, material footprint and target boundaries. A classification
    is returned only if every value in that interval belongs to the same zone.
    Exact boundary contact is UNKNOWN because visibility is not established.
    """
    if type(geometry_verified) is not bool:raise ValueError('geometry_verified must be boolean')
    if any(not finite(v) for v in (dx,dy,material_radius,uncertainty)):
        raise ValueError('finite projection measurements required')
    if material_radius<=0 or uncertainty<0:
        raise ValueError('positive material radius and non-negative uncertainty required')
    if (not isinstance(zone_outer_radii,(list,tuple)) or len(zone_outer_radii)!=6
            or any(not finite(v) or v<=0 for v in zone_outer_radii)
            or any(a>=b for a,b in zip(zone_outer_radii,zone_outer_radii[1:]))):
        raise ValueError('six strictly increasing positive zone radii required')
    center_error=hypot(dx,dy)
    extent=center_error+material_radius
    if not finite(extent+uncertainty):raise ValueError('projection bound overflow')
    low,high=max(0,extent-uncertainty),extent+uncertainty
    result={'status':'UNKNOWN','reason':'VERIFIED_PROJECTION_REQUIRED',
        'scoring_zone':None,'placement_state':'UNKNOWN',
        'center_error':center_error,'farthest_extent':extent,
        'extent_interval':[low,high],'official_points':None}
    if not geometry_verified:return result
    # Contact or uncertainty crossing any boundary can change the visible ring.
    if any(low<=r<=high for r in zone_outer_radii):
        result['reason']='ZONE_BOUNDARY_UNCERTAIN';return result
    if low>zone_outer_radii[-1]:
        result.update(status='CLASSIFIED',reason='PROJECTED_FOOTPRINT_OUTSIDE',placement_state='OUTSIDE')
        return result
    zone=next(i+1 for i,r in enumerate(zone_outer_radii) if high<r)
    result.update(status='CLASSIFIED',reason='WHOLE_CIRCULAR_PROJECTION_CONTAINED',
        scoring_zone=zone,placement_state='IN_RING')
    return result
