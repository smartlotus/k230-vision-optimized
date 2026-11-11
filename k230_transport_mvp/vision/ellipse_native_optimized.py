"""Full-resolution native mask contours with bounded, fresh ellipse measurements."""
from math import cos, sin, sqrt, pi, atan2, isfinite
from .ellipse import fit_ellipse

_BOUNDARY_NP=None
_KERNEL_CV=None
_CLOSE_KERNELS={}


def configure_boundary_backend(vector=False):
    """Explicit session opt-in; unsupported firmware must fail before capture."""
    global _BOUNDARY_NP
    if type(vector) is not bool:raise ValueError('Explicit vector backend flag required')
    if not vector:
        _BOUNDARY_NP=None;return 'SCALAR_FULL_BOUNDARY'
    from ulab import numpy as np
    for name in ('array','all','isfinite','sqrt','min','max','sum','arctan2'):
        if not hasattr(np,name):raise ValueError('Native array function unavailable: '+name)
    _BOUNDARY_NP=np
    return 'ULAB_FULL_BOUNDARY'


def vector_boundary_metrics(points,cx,cy,a,b,ct,st,np):
    """Same full-point residual and angular checks executed by array kernels."""
    data=np.array(points)
    if not np.all(np.isfinite(data)):return None
    dx,dy=data[:,0]-cx,data[:,1]-cy
    u,v=dx*ct+dy*st,dy*ct-dx*st
    gradient=np.sqrt((u/(a*a))**2+(v/(b*b))**2)*2
    if float(np.min(gradient))<1e-10:return None
    error=abs((u/a)**2+(v/b)**2-1)/gradient
    rms=sqrt(float(np.sum(error*error))/len(points));maximum=float(np.max(error))
    if not isfinite(rms) or not isfinite(maximum):return None
    inliers=int(np.sum(error<=3))
    angles=np.arctan2(v/b,u/a).tolist()
    sectors={min(11,int((theta%(2*pi))*12/(2*pi))) for theta in angles}
    return rms,maximum,sectors,inliers


def contour_fit(points, origin=(0, 0), roi=None, pixel_area=None, native_fit=None):
    """Validate the complete boundary, including points omitted from the fit."""
    rejected = lambda reason: {"accepted": False, "reason": reason, "center": None}
    if not 16 <= len(points) <= 4096:
        return rejected("CONTOUR_POINT_LIMIT")
    # Native scalar fitting already supplies center/axes: stream coordinates
    # instead of allocating a second full contour of translated tuples.
    translate_native = native_fit is not None and _BOUNDARY_NP is None
    ox, oy = origin
    if not translate_native:
        points = [(float(p[0])+ox, float(p[1])+oy) for p in points]
    if roi is not None:
        x, y, w, h = roi
        bounds_points = ((float(p[0])+ox, float(p[1])+oy) for p in points) if translate_native else points
        if any(px <= x or py <= y or px >= x+w-1 or py >= y+h-1 for px, py in bounds_points):
            return rejected("ROI_OR_IMAGE_CLIPPED")
    if native_fit is None:
        count = min(len(points), 128)
        sample = [points[i*len(points)//count] for i in range(count)]
        fit = fit_ellipse(sample)
    else:
        center, axes, degrees = native_fit
        first, second = float(axes[0])/2, float(axes[1])/2
        major, minor = max(first, second), min(first, second)
        angle = (float(degrees)+(90 if first < second else 0))*pi/180 % pi
        if (any(not isfinite(float(v)) for v in (center[0], center[1], major, minor, angle)) or
                not 3 <= minor <= major <= 120 or major/minor > 3):
            return rejected("AXIS_LIMIT")
        fit = {"accepted": True, "reason": "GEOMETRY_ONLY",
               "center": [float(center[0])+origin[0], float(center[1])+origin[1]],
               "semiaxes": [major, minor], "angle_rad": angle,
               "points": len(points), "backend": "NATIVE_CV2_FIT_ELLIPSE"}
    if not fit["accepted"]:
        return fit
    cx, cy = fit["center"]
    a, b = fit["semiaxes"]
    ct, st = cos(fit["angle_rad"]), sin(fit["angle_rad"])
    square_sum = maximum = 0
    sectors = set()
    inliers = 0
    if _BOUNDARY_NP is not None:
        metrics=vector_boundary_metrics(points,cx,cy,a,b,ct,st,_BOUNDARY_NP)
        if metrics is None:return rejected('DEGENERATE')
        rms,maximum,sectors,inliers=metrics
    else:
        aa, bb = a*a, b*b
        for px, py in points:
            if translate_native:
                px, py = float(px)+ox, float(py)+oy
            dx, dy = px-cx, py-cy
            u, v = ct*dx+st*dy, -st*dx+ct*dy
            gradient = 2*sqrt((u/aa)**2+(v/bb)**2)
            if gradient < 1e-10:
                return rejected("DEGENERATE")
            error = abs((u/a)**2+(v/b)**2-1)/gradient
            square_sum += error*error
            if error > maximum:maximum = error
            if error <= 3:inliers += 1
            theta = atan2(v/b, u/a) % (2*pi)
            sectors.add(min(11, int(theta*12/(2*pi))))
        rms = sqrt(square_sum/len(points))
    if rms > 2 or maximum > 5:
        diag=rejected("FULL_BOUNDARY_RESIDUAL")
        diag.update(rms_px=rms,max_error_px=maximum,inlier_fraction=inliers/len(points),contour_points=len(points),
                    semiaxes=[a,b],angle_rad=fit["angle_rad"],center=fit["center"])
        return diag
    if len(sectors) < 11:
        diag=rejected("INCOMPLETE_ANGULAR_COVERAGE")
        diag.update(rms_px=rms,max_error_px=maximum,inlier_fraction=inliers/len(points),contour_points=len(points),
                    sectors=len(sectors),semiaxes=[a,b],angle_rad=fit["angle_rad"],center=fit["center"])
        return diag
    fill = pixel_area/(pi*a*b) if pixel_area is not None else None
    if fill is not None and not .7 <= fill <= 1.2:
        return rejected("INTERIOR_FILL")
    fit.update(rms_px=rms, max_error_px=maximum, contour_points=len(points), interior_fill=fill, sectors=len(sectors))
    return fit


def extract_mask_ellipses(binary, roi, min_area=120, max_area=60000, cv=None, render_mask=None, close_kernel=0):
    """Extract all candidates. No reference-based identity or cached geometry."""
    if cv is None:
        import cv2 as cv
    if cv.countNonZero(binary)==0:
        return {"complete": True, "reason": "NO_MATCH_PIXELS", "area_semantics": "NATIVE_CONTOUR_AREA", "close_kernel": close_kernel, "components": [], "accepted": []}
    if close_kernel:
        if type(close_kernel) is not int or close_kernel < 3 or close_kernel % 2 == 0 or close_kernel > 7:
            raise ValueError("close_kernel must be an odd integer in 3..7")
        global _KERNEL_CV, _CLOSE_KERNELS
        if _KERNEL_CV is not cv:
            _KERNEL_CV=cv;_CLOSE_KERNELS={}
        kernel=_CLOSE_KERNELS.get(close_kernel)
        if kernel is None:
            kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(close_kernel,close_kernel))
            _CLOSE_KERNELS[close_kernel]=kernel
        binary=cv.morphologyEx(binary,cv.MORPH_CLOSE,kernel,iterations=1)
    contours, hierarchy = cv.findContours(binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    if len(contours) > 2048:
        return {"complete": False, "reason": "CONTOUR_LIMIT", "components": [], "accepted": []}
    # RETR_CCOMP provides a two-level hierarchy. Subtract real holes from fill.
    hs = hierarchy.tolist() if hasattr(hierarchy, "tolist") else hierarchy
    if hs is None:
        hs = []
    if len(hs) == 1 and len(hs[0]) == len(contours) and isinstance(hs[0][0], (list, tuple)):
        hs = hs[0]
    components, accepted = [], []
    for index, contour in enumerate(contours):
        if hs and int(hs[index][3]) >= 0:
            continue
        area = float(cv.contourArea(contour))
        x, y, w, h = [int(v) for v in cv.boundingRect(contour)]
        if area < min_area:
            continue
        if len(components) >= 32:
            return {"complete": False, "reason": "COMPONENT_LIMIT", "components": [], "accepted": []}
        record = {"bbox": [x+roi[0], y+roi[1], w, h], "contour_area": area}
        if not min_area <= area <= max_area:
            fit = {"accepted": False, "reason": "AREA_LIMIT", "center": None}
        else:
            child = int(hs[index][2]) if hs else -1
            while child >= 0:
                area -= float(cv.contourArea(contours[child]))
                child = int(hs[child][0])
            if len(contour) > 4096:
                fit = {"accepted": False, "reason": "CONTOUR_POINT_LIMIT", "center": None}
            else:
                raw = contour.tolist()
                # Desktop OpenCV returns Nx1x2; CanMV returns Nx2.
                points = [p[0] if len(p) == 1 else p for p in raw]
                try:
                    native_fit = cv.fitEllipse(contour)
                    fit = contour_fit(points, origin=roi[:2], roi=roi, pixel_area=area, native_fit=native_fit)
                except (ValueError, RuntimeError):
                    fit = {"accepted": False, "reason": "NATIVE_FIT_ERROR", "center": None}
        record["fit"] = fit
        components.append(record)
        if fit["accepted"]:
            accepted.append(fit)
            if render_mask is not None:
                cv.drawContours(render_mask, contours, index, (255,), -1)
    return {"complete": True, "reason": "FRESH_NATIVE_CONTOURS", "area_semantics": "NATIVE_CONTOUR_AREA", "close_kernel": close_kernel, "components": components, "accepted": accepted}


def ellipse_outline(fit, steps=48):
    """Source-space polygon avoids native drawing rotation convention differences."""
    cx, cy = fit["center"]
    a, b = fit["semiaxes"]
    ct, st = cos(fit["angle_rad"]), sin(fit["angle_rad"])
    result = []
    for i in range(steps):
        u, v = a*cos(2*pi*i/steps), b*sin(2*pi*i/steps)
        result.append((int(round(cx+ct*u-st*v)), int(round(cy+st*u+ct*v))))
    return result
