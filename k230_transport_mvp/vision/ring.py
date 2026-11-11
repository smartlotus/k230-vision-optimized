"""Concentric-circle clustering and explicitly measured ring identification."""
from .common import finite, in_roi


def cluster_circles(circles, center_tolerance=8):
    groups = []
    for circle in circles:
        if not all(finite(circle.get(k)) for k in ("x", "y", "r")) or circle["r"] <= 0:
            continue
        # Require compatibility with every member to prevent chain-merging adjacent rings.
        group = next((g for g in groups if all((circle["x"]-c["x"])**2+(circle["y"]-c["y"])**2 <= center_tolerance**2 for c in g)), None)
        if group is None:
            groups.append([circle])
        else:
            group.append(circle)
    return [{"center": (sum(c["x"] for c in g)/len(g), sum(c["y"] for c in g)/len(g)),
             "support": len(g)} for g in groups]


def identify_ring(groups, target_id, config):
    if not config or config.get("verified") is not True:
        return None, "ID_UNKNOWN"
    if "center_gates" in config:
        # These identities come from measured parking-pose references. Never
        # derive reference centers from this image or from the requested ID.
        return _identify_by_reference(groups, target_id, config["center_gates"])
    rois = config.get("id_rois")
    if rois:
        matches = []
        for group in groups:
            ids = [ring_id for ring_id, roi in rois.items() if in_roi(group["center"][0], group["center"][1], roi)]
            if len(ids) == 1 and ids[0] == target_id:
                matches.append(group)
        if len(matches) == 1:
            return matches[0], None
        return None, "ID_UNKNOWN"
    order = config.get("ring_id_order")
    if len(groups) != 3 or not order or sorted(order) != [1, 2, 3]:
        return None, "ID_UNKNOWN"
    ordered = sorted(groups, key=lambda item: item["center"][0])
    min_spacing = config.get("min_spacing_px")
    if min_spacing is None or any(ordered[i+1]["center"][0]-ordered[i]["center"][0] < min_spacing for i in (0, 1)):
        return None, "ID_UNKNOWN"
    return ordered[order.index(target_id)], None


def _identify_by_reference(groups, target_id, gates):
    if not isinstance(gates, dict) or not gates or type(target_id) is not int or target_id not in gates:
        return None, "ID_UNKNOWN"
    checked = []
    for ring_id, gate in gates.items():
        if type(ring_id) is not int or ring_id not in (1, 2, 3) or not isinstance(gate, dict):
            return None, "ID_UNKNOWN"
        center = gate.get("center")
        radius = gate.get("radius_px")
        gap = gate.get("min_distance_gap_px")
        support = gate.get("min_support")
        if (not isinstance(center, (tuple, list)) or len(center) != 2
                or not all(finite(v) for v in center)
                or not finite(radius) or radius <= 0
                or not finite(gap) or not 0 < gap <= radius
                or type(support) is not int or support < 1):
            return None, "ID_UNKNOWN"
        checked.append((ring_id, gate))
    # Overlapping numbered regions cannot provide an unambiguous identity.
    for index, (_, a) in enumerate(checked):
        for _, b in checked[index+1:]:
            distance2 = sum((a["center"][axis]-b["center"][axis])**2 for axis in (0, 1))
            if distance2 <= (a["radius_px"]+b["radius_px"])**2:
                return None, "ID_UNKNOWN"
    gate = gates[target_id]
    candidates = []
    for group in groups:
        center, support = group.get("center"), group.get("support")
        if (not isinstance(center, (list, tuple)) or len(center) != 2
                or not all(finite(v) for v in center)
                or type(support) is not int or support < gate["min_support"]):
            continue
        distance2 = sum((center[axis]-gate["center"][axis])**2 for axis in (0, 1))
        if distance2 <= gate["radius_px"]**2:
            candidates.append((distance2**0.5, group))
    candidates.sort(key=lambda item: item[0])
    if not candidates or (len(candidates) > 1
            and candidates[1][0]-candidates[0][0] < gate["min_distance_gap_px"]):
        return None, "ID_UNKNOWN"
    selected = dict(candidates[0][1])
    selected.update(selection_method="BOUND_CENTER_REFERENCE", reference_distance_px=candidates[0][0])
    return selected, None
