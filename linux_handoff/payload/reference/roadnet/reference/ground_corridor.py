"""Host-side RoadNet boundary and corridor measurement in ground coordinates."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GroundGrid:
    """Grid cell (row, col) is centered at (left, near) + (col, row) * cell_mm.

    x increases to the robot's right and y increases forward.
    """

    left_mm: float
    near_mm: float
    cell_mm: float
    rows: int
    cols: int


@dataclass(frozen=True)
class ProjectedFrame:
    timestamp_ms: int
    road: np.ndarray
    road_confidence: np.ndarray
    known_ground: np.ndarray
    image_to_current_ground_mm: np.ndarray


def _project(frame, metric_to_grid, shape, min_confidence):
    road = np.asarray(frame.road)
    known = np.asarray(frame.known_ground)
    confidence = np.asarray(frame.road_confidence, dtype=np.float32)
    if (road.ndim != 2 or not road.size or road.dtype != np.bool_
            or known.dtype != np.bool_ or road.shape != known.shape
            or confidence.shape != road.shape):
        raise ValueError("Each frame needs matching 2D road, known-ground and confidence arrays")
    matrix = np.asarray(frame.image_to_current_ground_mm, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("Each frame needs an invertible image-to-ground homography")

    candidate = road & known & (confidence >= min_confidence) & (confidence <= 1)
    image_to_grid = metric_to_grid @ matrix
    grid_to_image = np.linalg.inv(image_to_grid)
    grid_to_image /= np.max(np.abs(grid_to_image[2]))
    y, x = np.indices(shape, dtype=np.float64)
    corners = np.array([[-.5, -.5], [-.5, .5], [.5, -.5], [.5, .5]])
    homogeneous = np.stack((x + corners[:, 0, None, None],
                            y + corners[:, 1, None, None],
                            np.ones((4, *shape))))
    u, v, w = (grid_to_image @ homogeneous.reshape(3, -1)).reshape(3, 4, *shape)
    same_side = (w > 1e-9).all(axis=0) | (w < -1e-9).all(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = u / w, v / w

    # The bounding rectangle covers the entire projected cell, including thin
    # unknown or non-road regions that would fall between center samples.
    left, right = u.min(axis=0), u.max(axis=0)
    top, bottom = v.min(axis=0), v.max(axis=0)
    in_frame = (same_side & (left >= -.5) & (right <= road.shape[1] - .5)
                & (top >= -.5) & (bottom <= road.shape[0] - .5))
    x0 = np.floor(left[in_frame] + .5).astype(np.intp)
    y0 = np.floor(top[in_frame] + .5).astype(np.intp)
    x1 = np.ceil(right[in_frame] + .5).astype(np.intp)
    y1 = np.ceil(bottom[in_frame] + .5).astype(np.intp)
    pixel_counts = (x1 - x0) * (y1 - y0)
    summed = cv2.integral(candidate.astype(np.uint8))
    accepted = np.zeros(shape, dtype=bool)
    accepted[in_frame] = ((pixel_counts > 0)
                          & (summed[y1, x1] - summed[y0, x1] - summed[y1, x0]
                             + summed[y0, x0] == pixel_counts))
    scores = cv2.warpPerspective(confidence, image_to_grid, shape[::-1], flags=cv2.INTER_NEAREST)
    return accepted, scores


def measure_corridor(frames, grid, *, min_confidence, stable_frames, max_frame_gap_ms,
                     footprint_half_width_mm, footprint_half_length_mm, side_margin_mm,
                     braking_distance_mm, lookahead_mm):
    """Measure ground boundaries and a footprint-constrained corridor on the host.

    Supply consecutive frames oldest first. The homography for each undistorted
    image must include registration to the latest frame's robot ground axes.
    Only cells confirmed now and throughout the recent window can enter the result.
    The road confidence is the minimum center score across frames.
    """
    dimensions = np.asarray([
        grid.left_mm, grid.near_mm, grid.cell_mm, footprint_half_width_mm,
        footprint_half_length_mm, side_margin_mm, braking_distance_mm,
    ], dtype=float)
    if (not np.isfinite(dimensions).all() or np.any(dimensions[2:5] <= 0)
            or np.any(dimensions[5:] < 0) or grid.rows <= 0 or grid.cols <= 0):
        raise ValueError("Ground grid, footprint and clearance dimensions must be valid")
    if (not 0 < min_confidence <= 1 or stable_frames < 2
            or not np.isfinite(max_frame_gap_ms) or max_frame_gap_ms <= 0):
        raise ValueError("Specify a valid confidence and a temporal window of at least two frames")

    window = list(frames)[-stable_frames:]
    timestamps = np.asarray([frame.timestamp_ms for frame in window], dtype=np.float64)
    gaps = np.diff(timestamps)
    if not np.isfinite(timestamps).all() or np.any(gaps <= 0):
        raise ValueError("Frame timestamps must be finite and strictly increasing")

    shape = (grid.rows, grid.cols)
    divisor = grid.cell_mm
    metric_to_grid = np.array([[1, 0, -grid.left_mm], [0, 1, -grid.near_mm], [0, 0, divisor]]) / divisor
    stable = np.zeros(shape, dtype=bool)
    confidence = np.zeros(shape, dtype=np.float32)
    if len(window) == stable_frames and np.all(gaps <= max_frame_gap_ms):
        stable.fill(True)
        confidence.fill(1)
        for frame in window:
            accepted, scores = _project(frame, metric_to_grid, shape, min_confidence)
            stable &= accepted
            np.minimum(confidence, scores, out=confidence)
    confidence[~stable] = 0

    support = stable.astype(np.uint8)
    # Reserve a half-cell for the center's position within its accepted grid cell.
    half_width = int(np.ceil((footprint_half_width_mm + side_margin_mm) / divisor + 0.5))
    behind = int(np.ceil((footprint_half_length_mm + side_margin_mm) / divisor + 0.5))
    ahead = int(np.ceil((footprint_half_length_mm + side_margin_mm + braking_distance_mm)
                        / divisor + 0.5))
    corridor = np.zeros(shape, dtype=bool)
    if 2 * half_width + 1 <= grid.cols and behind + ahead + 1 <= grid.rows:
        footprint = np.ones((behind + ahead + 1, 2 * half_width + 1), dtype=np.uint8)
        corridor = cv2.erode(support, footprint, anchor=(half_width, behind),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)

    distance = cv2.distanceTransform(np.pad(support, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    clearance_mm = np.maximum(distance * divisor - divisor / np.sqrt(2), 0)
    contours, _ = cv2.findContours(support, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    origin = np.array([grid.left_mm, grid.near_mm])
    lookahead = {}
    for y_mm in lookahead_mm:
        if not np.isfinite(y_mm):
            raise ValueError("Look-ahead coordinates must be finite")
        row = round((y_mm - grid.near_mm) / divisor)
        if not 0 <= row < grid.rows:
            raise ValueError("Look-ahead coordinates must be inside the ground grid")
        edges = np.flatnonzero(np.diff(np.r_[False, stable[row], False]))
        spans = [(grid.left_mm + (first - 0.5) * divisor,
                  grid.left_mm + (end - 0.5) * divisor)
                 for first, end in edges.reshape(-1, 2)]
        lookahead[grid.near_mm + row * divisor] = spans

    return {
        "stable_road": stable,
        "safe_corridor": corridor,
        "road_confidence": confidence,
        "clearance_mm": clearance_mm,
        "boundary_contours_mm": [curve[:, 0, :] * divisor + origin for curve in contours],
        "lookahead_mm": lookahead,
    }
