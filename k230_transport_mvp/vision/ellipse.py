"""Bounded, dependency-free ellipse fitting for reviewed complete contours.

Geometry is a shape measurement, not object identity. Rejected fits have no center.
This exploratory module is not enabled in VisionAdapter or the live service.
"""
from math import atan2, cos, sin, sqrt, pi, isfinite


def _solve(matrix, vector):
    rows = [list(row)+[v] for row, v in zip(matrix, vector)]
    size = len(vector)
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-10:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        divisor = rows[col][col]
        rows[col] = [v/divisor for v in rows[col]]
        for row in range(size):
            if row != col:
                factor = rows[row][col]
                rows[row] = [a-factor*b for a, b in zip(rows[row], rows[col])]
    return [row[-1] for row in rows]


def fit_ellipse(points, min_points=16, max_points=128, min_axis=3, max_axis=120,
                max_ratio=3, max_rms=2, max_error=5, min_sectors=11):
    """Fit a conic in normalized coordinates, then check geometric residual/coverage.

    Axes are semiaxes in source pixels, angle is major-axis radians in [0, pi).
    No boundary trimming: an attached tail must not disappear as an 'outlier'.
    """
    rejected = lambda reason: {'accepted': False, 'reason': reason, 'center': None}
    if (type(min_points) is not int or type(max_points) is not int or
            not 6 <= min_points <= max_points <= 256 or type(min_sectors) is not int or
            not 1 <= min_sectors <= 12 or any(type(v) not in (int, float) or not isfinite(v) or v <= 0
                for v in (min_axis, max_axis, max_ratio, max_rms, max_error)) or
            min_axis > max_axis or max_ratio < 1):
        raise ValueError('Invalid ellipse limits')
    if not isinstance(points, (list, tuple)) or not min_points <= len(points) <= max_points:
        return rejected('POINT_COUNT')
    if any(not isinstance(p, (list, tuple)) or len(p) != 2 or any(
            type(v) not in (int, float) or not isfinite(v) for v in p) for p in points):
        return rejected('INVALID_POINT')
    if len(set(tuple(p) for p in points)) < min_points:
        return rejected('DUPLICATE_POINTS')
    mx = sum(p[0] for p in points)/len(points)
    my = sum(p[1] for p in points)/len(points)
    scale = max(max(p[a] for p in points)-min(p[a] for p in points) for a in (0, 1))/2
    if scale < 1e-6:
        return rejected('DEGENERATE')
    normalized = [((x-mx)/scale, (y-my)/scale) for x, y in points]
    normal = [[0.0]*5 for _ in range(5)]
    vector = [0.0]*5
    for x, y in normalized:
        features = (x*x, x*y, y*y, x, y)
        for i in range(5):
            vector[i] += features[i]
            for j in range(i, 5):
                normal[i][j] += features[i]*features[j]
    for i in range(5):
        for j in range(i):
            normal[i][j] = normal[j][i]
    coefficients = _solve(normal, vector)
    if coefficients is None:
        return rejected('DEGENERATE')
    a, b, c, d, e = coefficients
    det = 4*a*c-b*b
    if det <= 1e-10:
        return rejected('NOT_ELLIPSE')
    cx, cy = (b*e-2*c*d)/det, (b*d-2*a*e)/det
    level = 1-(a*cx*cx+b*cx*cy+c*cy*cy+d*cx+e*cy)
    if abs(level) < 1e-10:
        return rejected('DEGENERATE')
    a, b, c = a/level, b/level, c/level
    discriminant = sqrt((a-c)**2+b*b)
    low, high = (a+c-discriminant)/2, (a+c+discriminant)/2
    if low <= 1e-10 or high <= 1e-10:
        return rejected('NOT_ELLIPSE')
    major, minor = scale/sqrt(low), scale/sqrt(high)
    if not min_axis <= minor <= major <= max_axis or major/minor > max_ratio:
        return rejected('AXIS_LIMIT')
    angle = (0.5*atan2(-b, c-a)) % pi
    ct, st = cos(angle), sin(angle)
    errors, sectors = [], set()
    for x, y in normalized:
        x, y = x-cx, y-cy
        value = a*x*x+b*x*y+c*y*y
        gradient = 2*sqrt((a*x+b*y/2)**2+(b*x/2+c*y)**2)
        if gradient < 1e-10:
            return rejected('DEGENERATE')
        errors.append(abs(value-1)/gradient*scale)
        theta = atan2((-st*x+ct*y)*scale/minor, (ct*x+st*y)*scale/major) % (2*pi)
        sectors.add(min(11, int(theta*12/(2*pi))))
    rms = sqrt(sum(v*v for v in errors)/len(errors))
    if rms > max_rms or max(errors) > max_error:
        return rejected('BOUNDARY_RESIDUAL')
    if len(sectors) < min_sectors:
        return rejected('INCOMPLETE_ANGULAR_COVERAGE')
    return {'accepted': True, 'reason': 'GEOMETRY_ONLY', 'center': [mx+cx*scale, my+cy*scale],
            'semiaxes': [major, minor], 'angle_rad': angle, 'rms_px': rms,
            'max_error_px': max(errors), 'sectors': len(sectors), 'points': len(points)}
