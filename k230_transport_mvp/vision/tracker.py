"""Consecutive fresh-frame stability, with a bounded total motion window."""
from .common import elapsed_ms, finite


class StableTracker:
    def __init__(self, frames=3, max_motion_px=4, max_gap_ms=300):
        if frames < 1 or max_motion_px < 0 or max_gap_ms <= 0:
            raise ValueError("invalid stability settings")
        self.frames = frames
        self.max_motion_px = max_motion_px
        self.max_gap_ms = max_gap_ms
        self.reset()

    def reset(self):
        self.key = None
        self.points = []
        self.last_frame = None
        self.last_time = None

    def update(self, key, frame_id, timestamp_ms, center):
        if center is None or not all(finite(v) for v in center):
            self.reset()
            return False
        if self.key != key:
            self.reset()
            self.key = key
        if self.last_frame is not None:
            if frame_id == self.last_frame:
                return False
            gap = elapsed_ms(timestamp_ms, self.last_time)
            if frame_id < self.last_frame or gap <= 0 or gap > self.max_gap_ms:
                self.reset()
                self.key = key
        if any((center[0]-p[0])**2 + (center[1]-p[1])**2 > self.max_motion_px**2 for p in self.points):
            self.points = []
        self.points.append(tuple(center))
        self.points = self.points[-self.frames:]
        self.last_frame = frame_id
        self.last_time = timestamp_ms
        return len(self.points) >= self.frames


class TurntableTracker:
    """Window-only turntable observation; motion prediction remains unavailable."""
    def __init__(self, frames=3, max_motion_px=4, max_gap_ms=300):
        self.window = StableTracker(frames, max_motion_px, max_gap_ms)
        self.reset()

    def reset(self):
        self.window.reset()
        self.context = None
        self.angle = None
        self.omega = None
        self.phase = None
        self.last_seen = None

    def update_window(self, context, frame_id, timestamp_ms, center):
        if context != self.context:
            self.reset()
            self.context = context
        valid = isinstance(center, (tuple, list)) and len(center) == 2 and all(finite(v) for v in center)
        previous_frame = self.window.last_frame
        stable = self.window.update(context, frame_id, timestamp_ms, center if valid else None)
        if valid and frame_id != previous_frame and (self.last_seen is None or elapsed_ms(timestamp_ms, self.last_seen) > 0):
            self.last_seen = timestamp_ms
        current = tuple(center) if valid else None
        return {"stable": stable, "center_px": current,
                "window_candidate": current if stable else None,
                "last_seen": self.last_seen, "angle": self.angle,
                "omega": self.omega, "phase": self.phase}
