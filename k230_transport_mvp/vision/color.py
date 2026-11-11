"""Select a complete object bbox inside the measured grasp window."""
from .common import finite, in_roi


def select_candidate(candidates, target_id, roi, reference, min_area, max_area, excluded=None):
    if roi is None or reference is None or min_area is None or max_area is None:
        return None
    selected = []
    for item in candidates:
        if item.get("target_id") != target_id or item.get("consumed", False):
            continue
        box = item.get("bbox", ())
        if len(box) != 4 or not all(finite(v) for v in box):
            continue
        x, y, w, h = box
        if w <= 0 or h <= 0 or not min_area <= w*h <= max_area:
            continue
        # Touching either the observation ROI or image edge means the object may be clipped.
        if not (roi[0] < x and roi[1] < y and x+w < roi[0]+roi[2] and y+h < roi[1]+roi[3]):
            continue
        center = (x+w/2, y+h/2)
        if excluded and any((center[0]-p[0])**2+(center[1]-p[1])**2 <= p[2]**2 for p in excluded):
            continue
        selected.append(((center[0]-reference[0])**2+(center[1]-reference[1])**2, center, item))
    if not selected:
        return None
    selected.sort(key=lambda item: item[0])
    _, center, item = selected[0]
    return {"center": center, "bbox": item["bbox"], "score_hint": item.get("score_hint", 50)}
