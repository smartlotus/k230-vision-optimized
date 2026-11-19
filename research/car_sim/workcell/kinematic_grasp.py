"""Geometric opposing-jaw contact for the PC workcell plant."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MaterialBody:
    batch: int
    color: int
    bottom_xyz_mm: tuple

    @property
    def identity(self):
        return self.batch, self.color


@dataclass(frozen=True)
class GripContact:
    identity: tuple
    supported_diameter_mm: float
    cross_center_offset_mm: float
    along_center_offset_mm: float
    height_overlap_mm: float


def profile_diameter(cad, height_mm):
    geometry = cad["geometry"]
    base, top, slope = (geometry[k] for k in ("bottom_cylinder", "top_cylinder", "middle_frustum"))
    if height_mm <= base["height"]:
        return float(base["diameter"])
    if height_mm >= geometry["overall_height"] - top["height"]:
        return float(top["diameter"])
    relative = (height_mm - base["height"]) / slope["height"]
    return float(base["diameter"] + relative * (top["diameter"] - base["diameter"]))


def opposed_contact(tcp_xyz_mm, theta_rad, open_gap_mm, closing_gap_mm, tool, cad, bodies):
    """Inspect physical poses independently of image selection or intended color."""
    if not 0 <= closing_gap_mm < open_gap_mm <= tool.open_gap_mm:
        raise ValueError("Jaw request outside configured travel")
    x, y, z = tcp_xyz_mm
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    candidates = []
    for body in bodies:
        bx, by, bz = body.bottom_xyz_mm
        dx, dy = bx - x, by - y
        cross, along = c * dx + s * dy, -s * dx + c * dy
        overlap_lo = max(bz, z - tool.finger_height_mm / 2)
        overlap_hi = min(bz + cad["geometry"]["overall_height"], z + tool.finger_height_mm / 2)
        overlap = overlap_hi - overlap_lo
        if overlap < 5.:
            continue
        diameter = profile_diameter(cad, overlap_lo - bz)
        if (abs(along) > tool.finger_depth_mm / 2 - 5. or abs(cross) > 4.
                or diameter + 2 * abs(cross) > open_gap_mm
                or closing_gap_mm > diameter - 2 * abs(cross)):
            continue
        candidates.append(GripContact(body.identity, diameter, cross, along, overlap))
    if len(candidates) != 1:
        return None, "EMPTY_CONTACT" if not candidates else "AMBIGUOUS_CONTACT"
    return candidates[0], "OPPOSED_JAW_CONTACT"


def test_lift(contact, tcp_before_mm, tcp_after_mm, *, slips=False):
    if contact is None:
        return {"holding": False, "reason": "NO_CONFIRMED_CONTACT"}
    before, after = tcp_before_mm, tcp_after_mm
    amount = after[2] - before[2]
    if not 6 <= amount <= 15 or math.dist(before[:2], after[:2]) > 2.5:
        return {"holding": False, "reason": "TRIAL_LIFT_OUTSIDE_BOUNDS"}
    if slips:
        return {"holding": False, "reason": "SLIPPED_DURING_TRIAL_LIFT", "lift_mm": amount}
    return {"holding": True, "reason": "HELD_AFTER_TRIAL_LIFT", "lift_mm": amount}
