"""Exploratory complete-component contours from a frozen native LAB grid.

The grid is sparse. Mapped contours and pixel areas are approximations, not native
full-resolution segmentation. No nearest-candidate choice or production gate here.
"""
from math import isfinite, pi
from .ellipse import fit_ellipse


def segment_lab_grid(sample, threshold, roi, min_cells=25, max_components=32, max_points=128, min_fill=.7, **fit_limits):
    if (sample.get('source') != 'REAL_CAMERA_NATIVE_LAB' or sample.get('camera_id') != 1 or
            sample.get('is_simulated') is not False):
        raise ValueError('Explicit native CSI1 LAB grid required')
    grid = sample.get('lab_grid')
    width, height = sample.get('width'), sample.get('height')
    stride, x0, y0 = sample.get('stride'), sample.get('x0'), sample.get('y0')
    if (any(type(v) is not int for v in (width, height, stride, x0, y0)) or
            not 1 <= stride <= 32 or width <= 0 or height <= 0 or not 0 <= x0 < width or not 0 <= y0 < height or
            not isinstance(grid, list) or not grid or not isinstance(grid[0], list) or not grid[0] or
            len(grid)*len(grid[0]) > 14400):
        raise ValueError('Bounded rectangular grid required')
    rows, cols = len(grid), len(grid[0])
    if x0+(cols-1)*stride >= width or y0+(rows-1)*stride >= height or any(
            not isinstance(row, list) or len(row) != cols for row in grid):
        raise ValueError('Grid geometry mismatch')
    if (not isinstance(threshold, (list, tuple)) or len(threshold) != 6 or
            any(type(v) is not int for v in threshold) or
            any(not lower <= threshold[2*a] <= threshold[2*a+1] <= upper
                for a, (lower, upper) in enumerate(((0, 100), (-128, 127), (-128, 127))))):
        raise ValueError('Ordered LAB threshold required')
    if (not isinstance(roi, (list, tuple)) or len(roi) != 4 or any(type(v) is not int for v in roi) or
            min(roi[:2]) < 0 or min(roi[2:]) <= 0 or roi[0]+roi[2] > width or roi[1]+roi[3] > height or
            type(min_cells) is not int or min_cells < 1 or type(max_components) is not int or not 1 <= max_components <= 64 or
            type(max_points) is not int or not 16 <= max_points <= 256 or
            type(min_fill) not in (int, float) or not isfinite(min_fill) or not 0 < min_fill <= 1):
        raise ValueError('Invalid component bounds')
    rx, ry, rw, rh = roi
    mask = bytearray(rows*cols)
    for y, row in enumerate(grid):
        for x, lab in enumerate(row):
            if not isinstance(lab, (tuple, list)) or len(lab) != 3 or any(
                    type(v) not in (int, float) or not isfinite(v) or not low <= v <= high
                    for v, (low, high) in zip(lab, ((0, 100), (-128, 127), (-128, 127)))):
                raise ValueError('Invalid native LAB point')
            px, py = x0+x*stride, y0+y*stride
            if rx <= px < rx+rw and ry <= py < ry+rh and all(
                    threshold[a*2] <= lab[a] <= threshold[a*2+1] for a in range(3)):
                mask[y*cols+x] = 1
    # Mark background reachable from image borders. Holes stay in the threshold
    # mask but do not become extra interior contours in the ellipse fit.
    exterior = bytearray(len(mask))
    pending = []
    for y in range(rows):
        for x in range(cols):
            index = y*cols+x
            if (x in (0, cols-1) or y in (0, rows-1)) and not mask[index] and not exterior[index]:
                exterior[index] = 1
                pending.append(index)
    while pending:
        y, x = divmod(pending.pop(), cols)
        for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            if 0 <= nx < cols and 0 <= ny < rows:
                index = ny*cols+nx
                if not mask[index] and not exterior[index]:
                    exterior[index] = 1
                    pending.append(index)
    visited = bytearray(len(mask))
    components = []
    for seed, matched in enumerate(mask):
        if not matched or visited[seed]:
            continue
        stack, cells, border = [seed], [], []
        hole_border_cells = 0
        visited[seed] = 1
        while stack:
            index = stack.pop()
            cells.append(index)
            y, x = divmod(index, cols)
            boundary = hole_boundary = False
            for nx, ny in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if nx < 0 or ny < 0 or nx >= cols or ny >= rows:
                    boundary = True
                elif not mask[ny*cols+nx]:
                    if exterior[ny*cols+nx]:
                        boundary = True
                    else:
                        hole_boundary = True
                elif not visited[ny*cols+nx]:
                    visited[ny*cols+nx] = 1
                    stack.append(ny*cols+nx)
            if boundary:
                border.append((x0+x*stride, y0+y*stride))
            if hole_boundary:
                hole_border_cells += 1
        if len(cells) < min_cells:
            continue
        if len(components) >= max_components:
            return {'complete': False, 'reason': 'COMPONENT_LIMIT', 'components': [], 'accepted': 0}
        left = x0+min(v % cols for v in cells)*stride-stride/2
        top = y0+min(v // cols for v in cells)*stride-stride/2
        right = x0+max(v % cols for v in cells)*stride+stride/2
        bottom = y0+max(v // cols for v in cells)*stride+stride/2
        contour = sorted(border)
        points = [contour[i*len(contour)//max_points] for i in range(max_points)] if len(contour) > max_points else contour
        if left <= rx or top <= ry or right >= rx+rw or bottom >= ry+rh:
            fit = {'accepted': False, 'reason': 'ROI_OR_IMAGE_CLIPPED', 'center': None}
        else:
            fit = fit_ellipse(points, max_points=max_points, **fit_limits)
        fill = len(cells)*stride*stride/(pi*fit['semiaxes'][0]*fit['semiaxes'][1]) if fit['accepted'] else None
        if fill is not None and fill < min_fill:
            fit = {'accepted': False, 'reason': 'LOW_INTERIOR_FILL', 'center': None}
        components.append({'grid_cells': len(cells), 'bbox': [left, top, right-left, bottom-top],
                           'bbox_area_estimate_px2': (right-left)*(bottom-top), 'contour_points': len(contour),
                           'interior_boundary_cells': hole_border_cells, 'interior_fill_estimate': fill,
                           'sampled_contour': points, 'fit': fit})
    return {'complete': True, 'reason': 'SPARSE_GRID_EXPLORATION_ONLY', 'matched_cells': sum(mask),
            'components': components, 'accepted': sum(c['fit']['accepted'] for c in components)}
