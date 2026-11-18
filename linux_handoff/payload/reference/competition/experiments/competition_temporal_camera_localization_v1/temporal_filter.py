"""Reachable-grid Bayesian filter for a camera optical center in FIELD_MAP."""
import math
import numpy as np


class TemporalCameraFilter:
    def __init__(self, valid_mask, cell_mm=100., origin_x_mm=-1200.,
                 origin_y_mm=-1200., top_k=5, bad_updates_to_lost=5,
                 min_reachable_visual_mass=.05, global_recovery_mix=.02,
                 tracking_peak_mass=.15, max_gap_s=1., hard_gate_enabled=False):
        self.valid = np.array(valid_mask, dtype=bool, copy=True)
        if self.valid.ndim != 2 or not self.valid.any():
            raise ValueError("valid_mask must contain valid cells")
        values = (cell_mm, origin_x_mm, origin_y_mm, min_reachable_visual_mass,
                  global_recovery_mix, tracking_peak_mass, max_gap_s)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("configuration must be finite")
        if cell_mm <= 0 or max_gap_s <= 0 or top_k < 1 or bad_updates_to_lost < 1:
            raise ValueError("positive cell, gap, candidate and failure limits required")
        if not all(0 <= v <= 1 for v in values[3:6]):
            raise ValueError("probability thresholds must be in [0,1]")
        self.h, self.w = self.valid.shape
        self.cell = float(cell_mm)
        self.origin_x = float(origin_x_mm)
        self.origin_y = float(origin_y_mm)
        self.top_k = int(top_k)
        self.bad_limit = int(bad_updates_to_lost)
        self.min_mass = float(min_reachable_visual_mass)
        self.recovery_mix = float(global_recovery_mix)
        self.tracking_mass = float(tracking_peak_mass)
        self.max_gap = float(max_gap_s)
        self.hard_gate = bool(hard_gate_enabled)
        self.uniform = self.valid.astype(float) / self.valid.sum()
        self.belief = None
        self.last_ts = None
        self.bad = 0
        self.state = "UNINITIALIZED"
        self.recovery_belief = None
        self.recovery_streak = 0

    def _norm(self, values):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != self.valid.shape or not np.isfinite(array).all() or (array < 0).any():
            raise ValueError("likelihood must be finite, nonnegative and match the grid")
        array = np.where(self.valid, array, 0.)
        scale = float(array.max())
        if scale == 0:
            return None
        array = array / scale
        return array / array.sum()

    def _timestamp(self, timestamp):
        value = float(timestamp)
        if not math.isfinite(value) or (self.last_ts is not None and value < self.last_ts):
            raise ValueError("timestamp must be finite and monotonic")
        return value

    def initialize(self, prior=None, timestamp_s=None):
        ts = None if timestamp_s is None else self._timestamp(timestamp_s)
        p = self.uniform.copy() if prior is None else self._norm(prior)
        if p is None:
            raise ValueError("initial prior has no valid mass")
        self.belief, self.last_ts, self.bad = p, ts, 0
        self.recovery_belief, self.recovery_streak = None, 0
        self.state = "DEGRADED"
        return self.result()

    def predict(self, dt_s, v_camera_max_mm_s, process_margin_mm, belief=None):
        """Uniform motion from each source cell, conserving its probability mass."""
        values = (dt_s, v_camera_max_mm_s, process_margin_mm)
        if any(v is None or not math.isfinite(float(v)) or v < 0 for v in values):
            raise ValueError("prediction requires known finite nonnegative motion bounds")
        radius = v_camera_max_mm_s * dt_s + process_margin_mm
        source_belief = self.belief if belief is None else belief
        if source_belief is None:
            raise ValueError("initialize the filter before predicting")
        if radius >= math.hypot(self.h * self.cell, self.w * self.cell):
            return self.uniform.copy()
        offsets = [(0, 0)]
        if radius > 0:
            offsets = []
            # Minimum separation of cell areas avoids rejecting edge positions.
            extent = min(int(math.ceil(radius / self.cell)) + 1, max(self.h, self.w) - 1)
            for dy in range(-extent, extent + 1):
                for dx in range(-extent, extent + 1):
                    distance = math.hypot(max(0, abs(dx) - 1), max(0, abs(dy) - 1)) * self.cell
                    if distance <= radius and abs(dy) < self.h and abs(dx) < self.w:
                        offsets.append((dy, dx))
        slices = []
        counts = np.zeros(self.valid.shape)
        for dy, dx in offsets:
            sy = slice(max(0, -dy), min(self.h, self.h - dy))
            sx = slice(max(0, -dx), min(self.w, self.w - dx))
            ty = slice(max(0, dy), min(self.h, self.h + dy))
            tx = slice(max(0, dx), min(self.w, self.w + dx))
            counts[sy, sx] += self.valid[ty, tx]
            slices.append((sy, sx, ty, tx))
        source = np.divide(source_belief, counts, out=np.zeros_like(counts), where=counts > 0)
        output = np.zeros_like(counts)
        for sy, sx, ty, tx in slices:
            output[ty, tx] += source[sy, sx] * self.valid[ty, tx]
        return self._norm(output)

    def _top_candidates(self, belief):
        if belief is None:
            return []
        flat = belief.ravel()
        ids = np.flatnonzero(flat > 0)
        ids = ids[np.argsort(-flat[ids], kind="stable")[:self.top_k]]
        top = []
        for idx in ids:
            row, col = divmod(int(idx), self.w)
            top.append(dict(x_mm=self.origin_x + (col + .5) * self.cell,
                            y_mm=self.origin_y + (row + .5) * self.cell,
                            probability=float(flat[idx])))
        return top

    def result(self, relocalized=False):
        if self.belief is None:
            return dict(camera_x_mm=None, camera_y_mm=None, confidence=0., entropy=None,
                        top_k=[], state=self.state, relocalized=False)
        flat = self.belief.ravel()
        top = self._top_candidates(self.belief)
        positive = flat[flat > 0]
        return dict(camera_x_mm=top[0]["x_mm"], camera_y_mm=top[0]["y_mm"],
                    confidence=top[0]["probability"], confidence_status="UNCALIBRATED_POSTERIOR_MASS",
                    entropy=float(-(positive * np.log(positive)).sum()), top_k=top,
                    state=self.state, relocalized=bool(relocalized),
                    recovery_top_k=self._top_candidates(self.recovery_belief),
                    recovery_updates=self.recovery_streak)

    def update(self, visual_likelihood, timestamp_s, v_camera_max_mm_s=None,
               process_margin_mm=None, visual_quality=1.):
        ts = self._timestamp(timestamp_s)
        quality = float(visual_quality)
        if not math.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError("visual_quality must be finite in [0,1]")
        vis = None if visual_likelihood is None else self._norm(visual_likelihood)
        known = v_camera_max_mm_s is not None and process_margin_mm is not None
        if known:
            if any(not math.isfinite(float(v)) or v < 0 for v in (v_camera_max_mm_s, process_margin_mm)):
                raise ValueError("motion bounds must be finite and nonnegative")
        if self.hard_gate and not known:
            raise ValueError("hard gating requires established camera motion bounds")
        dt = 0. if self.last_ts is None else ts - self.last_ts
        gap = dt > self.max_gap
        fresh = self.belief is None
        prior = self.uniform.copy()
        if not fresh and known:
            prior = self.predict(dt, v_camera_max_mm_s, process_margin_mm)
        usable = vis is not None and quality >= .15
        mass = float(vis[prior > 0].sum()) if usable else 0.
        conflict = usable and known and not fresh and mass < self.min_mass
        bad = self.bad + 1 if not usable or conflict else 0
        relocalized = False
        recovery, streak = None, 0
        if conflict and self.hard_gate:
            recovery_prior = self.uniform
            if self.recovery_belief is not None:
                recovery_prior = self.predict(dt, v_camera_max_mm_s, process_margin_mm,
                                              belief=self.recovery_belief)
            recovery_prior = (1 - self.recovery_mix) * recovery_prior + self.recovery_mix * self.uniform
            recovery = self._norm(recovery_prior * vis)
            previous_recovery = self.recovery_belief
            if recovery is None:
                recovery, previous_recovery = vis, None
            streak = 1
            if previous_recovery is not None:
                old_row, old_col = divmod(int(previous_recovery.argmax()), self.w)
                new_row, new_col = divmod(int(recovery.argmax()), self.w)
                separation = math.hypot(max(0, abs(new_col - old_col) - 1),
                                        max(0, abs(new_row - old_row) - 1)) * self.cell
                if separation <= v_camera_max_mm_s * dt + process_margin_mm:
                    streak = self.recovery_streak + 1
        if not usable:
            posterior = prior
            state = "LOST" if bad >= self.bad_limit or gap else "DEGRADED"
        elif conflict and self.hard_gate:
            posterior = prior
            state = "LOST" if bad >= self.bad_limit else "DEGRADED"
            if streak >= self.bad_limit:
                confirmed_origin = int(recovery.argmax())
                confirm_dt = 0.
                if previous_recovery is not None:
                    confirmed_origin, confirm_dt = int(previous_recovery.argmax()), dt
                mode = np.zeros_like(prior)
                mode.flat[confirmed_origin] = 1.
                confirmed = self.predict(confirm_dt, v_camera_max_mm_s, process_margin_mm, belief=mode)
                posterior = self._norm(recovery * (confirmed > 0))
                state, bad, relocalized = "DEGRADED", 0, True
                recovery, streak = None, 0
        elif conflict and bad < self.bad_limit:
            posterior, state = prior, "DEGRADED"
        elif conflict:
            posterior, state, bad = vis, "DEGRADED", 0
            relocalized = True
        else:
            if not self.hard_gate:
                prior = (1 - self.recovery_mix) * prior + self.recovery_mix * self.uniform
            posterior = self._norm(prior * vis)
            if posterior is None:
                posterior, bad, state = prior, bad + 1, "DEGRADED"
            else:
                confident = posterior.max() >= self.tracking_mass and quality >= .5
                state = "TRACKING" if confident and known and self.hard_gate and not gap else "DEGRADED"
        self.belief, self.last_ts, self.bad, self.state = posterior, ts, bad, state
        self.recovery_belief, self.recovery_streak = recovery, streak
        result = self.result(relocalized)
        result.update(dt_s=dt, reachable_visual_mass=mass, motion_bound_known=known,
                      hard_gate_enabled=self.hard_gate, bad_updates=bad)
        return result

    def reset_global(self, timestamp_s=None):
        self.initialize(timestamp_s=timestamp_s)
        self.state = "LOST"
        return self.result()


TemporalGridFilter = TemporalCameraFilter
