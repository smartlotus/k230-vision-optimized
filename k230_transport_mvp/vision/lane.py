"""Auxiliary lane offset from multiple measured grayscale scan lines."""


def estimate_lane(rows, threshold, min_width, max_width, max_center_spread=20):
    if threshold is None or min_width is None or max_width is None:
        return None
    centers = []
    bounds = []
    for row in rows:
        values = row["values"]
        runs = []
        start = None
        for i in range(len(values)+1):
            hit = i < len(values) and threshold[0] <= values[i] <= threshold[1]
            if hit and start is None:
                start = i
            elif not hit and start is not None:
                # Both edges must be visible; a full-width gray ROI is ambiguous.
                if start > 0 and i < len(values) and min_width <= i-start <= max_width:
                    runs.append((start, i))
                start = None
        if len(runs) != 1:
            return None
        left, right = runs[0]
        step = row.get("step", 1)
        origin = row.get("x", 0)
        centers.append(origin+(left+right-1)*step/2)
        bounds.append((origin+left*step, origin+(right-1)*step))
    if len(centers) < 3 or max(centers)-min(centers) > max_center_spread:
        return None
    return {"center_x": sum(centers)/len(centers), "bounds": bounds, "score_hint": 50}
