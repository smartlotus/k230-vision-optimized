"""Pure-Python fixed-grid runtime for CanMV. Uses only the math module.

Inputs are flat row-major probability arrays. Keep this object alive between
frames. The host parity test binds its behavior to temporal_filter.py.
"""
import math


class TemporalCameraFilter:
    def __init__(self, valid_mask, width, cell_mm=50., origin_x_mm=-1200.,
                 origin_y_mm=-1200., top_k=5, bad_updates_to_lost=5,
                 min_reachable_visual_mass=.05, global_recovery_mix=.02,
                 tracking_peak_mass=.15, max_gap_s=1., hard_gate_enabled=False):
        self.valid = [bool(v) for v in valid_mask]
        self.w = int(width)
        if self.w < 1 or len(self.valid) % self.w or not any(self.valid):
            raise ValueError('Invalid map mask/width')
        self.h = len(self.valid) // self.w
        self.cell, self.origin_x, self.origin_y = cell_mm, origin_x_mm, origin_y_mm
        self.top_k, self.bad_limit = top_k, bad_updates_to_lost
        self.min_mass, self.recovery_mix = min_reachable_visual_mass, global_recovery_mix
        self.tracking_mass, self.max_gap = tracking_peak_mass, max_gap_s
        self.hard_gate = hard_gate_enabled
        values = (cell_mm, origin_x_mm, origin_y_mm, max_gap_s)
        if not all(math.isfinite(v) for v in values) or cell_mm <= 0 or max_gap_s <= 0:
            raise ValueError('Invalid metric configuration')
        if top_k < 1 or bad_updates_to_lost < 1 or not all(0 <= v <= 1 for v in
                (min_reachable_visual_mass, global_recovery_mix, tracking_peak_mass)):
            raise ValueError('Invalid state thresholds')
        count = sum(self.valid)
        self.uniform = [1. / count if v else 0. for v in self.valid]
        self.belief = None
        self.last_ts, self.bad, self.state = None, 0, 'UNINITIALIZED'
        self.recovery_belief, self.recovery_streak = None, 0

    def _norm(self, values):
        if len(values) != len(self.valid):
            raise ValueError('Flat likelihood size differs from map')
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('Likelihood must be finite and nonnegative')
        output = [float(v) if self.valid[i] else 0. for i, v in enumerate(values)]
        peak = max(output)
        if peak == 0:
            return None
        output = [v / peak for v in output]
        total = sum(output)
        return [v / total for v in output]

    def _timestamp(self, timestamp):
        value = float(timestamp)
        if not math.isfinite(value) or (self.last_ts is not None and value < self.last_ts):
            raise ValueError('Timestamp must be finite and monotonic')
        return value

    def initialize(self, prior=None, timestamp_s=None):
        ts = None if timestamp_s is None else self._timestamp(timestamp_s)
        p = list(self.uniform) if prior is None else self._norm(prior)
        if p is None:
            raise ValueError('Initial prior has no valid mass')
        self.belief, self.last_ts, self.bad, self.state = p, ts, 0, 'DEGRADED'
        self.recovery_belief, self.recovery_streak = None, 0
        return self.result()

    def predict(self, dt_s, v_camera_max_mm_s, process_margin_mm, belief=None):
        if any(v is None or not math.isfinite(v) or v < 0 for v in
               (dt_s, v_camera_max_mm_s, process_margin_mm)):
            raise ValueError('Established nonnegative motion bounds required')
        source_belief = self.belief if belief is None else belief
        if source_belief is None:
            raise ValueError('Initialize before prediction')
        radius = dt_s * v_camera_max_mm_s + process_margin_mm
        if radius >= math.hypot(self.h * self.cell, self.w * self.cell):
            return list(self.uniform)
        offsets = [(0, 0)]
        if radius > 0:
            offsets = []
            extent = min(int(math.ceil(radius / self.cell)) + 1, max(self.h, self.w) - 1)
            for dy in range(-extent, extent + 1):
                for dx in range(-extent, extent + 1):
                    distance = math.hypot(max(0, abs(dx) - 1), max(0, abs(dy) - 1)) * self.cell
                    if distance <= radius and abs(dy) < self.h and abs(dx) < self.w:
                        offsets.append((dy, dx))
        output = [0.] * len(self.valid)
        for source, mass in enumerate(source_belief):
            if mass == 0:
                continue
            row, col = divmod(source, self.w)
            destinations = []
            for dy, dx in offsets:
                rr, cc = row + dy, col + dx
                if 0 <= rr < self.h and 0 <= cc < self.w:
                    target = rr * self.w + cc
                    if self.valid[target]:
                        destinations.append(target)
            contribution = mass / len(destinations)
            for target in destinations:
                output[target] += contribution
        return self._norm(output)

    def _top_candidates(self, belief):
        if belief is None:
            return []
        ids = [i for i, p in enumerate(belief) if p > 0]
        ids.sort(key=lambda i: -belief[i])
        top = []
        for index in ids[:self.top_k]:
            row, col = divmod(index, self.w)
            top.append(dict(x_mm=self.origin_x + (col + .5) * self.cell,
                            y_mm=self.origin_y + (row + .5) * self.cell,
                            probability=belief[index]))
        return top

    def result(self, relocalized=False):
        if self.belief is None:
            return dict(camera_x_mm=None, camera_y_mm=None, confidence=0., entropy=None,
                        top_k=[], state=self.state, relocalized=False)
        top = self._top_candidates(self.belief)
        return dict(camera_x_mm=top[0]['x_mm'], camera_y_mm=top[0]['y_mm'],
                    confidence=top[0]['probability'], confidence_status='UNCALIBRATED_POSTERIOR_MASS',
                    entropy=-sum(p * math.log(p) for p in self.belief if p > 0), top_k=top,
                    state=self.state, relocalized=bool(relocalized),
                    recovery_top_k=self._top_candidates(self.recovery_belief),
                    recovery_updates=self.recovery_streak)

    def update(self, visual_likelihood, timestamp_s, v_camera_max_mm_s=None,
               process_margin_mm=None, visual_quality=1.):
        ts = self._timestamp(timestamp_s)
        if not math.isfinite(visual_quality) or not 0 <= visual_quality <= 1:
            raise ValueError('Invalid visual quality')
        vis = None if visual_likelihood is None else self._norm(visual_likelihood)
        known = v_camera_max_mm_s is not None and process_margin_mm is not None
        if known and any(not math.isfinite(v) or v < 0 for v in (v_camera_max_mm_s, process_margin_mm)):
            raise ValueError('Invalid motion bounds')
        if self.hard_gate and not known:
            raise ValueError('Hard gating requires known camera motion bounds')
        dt = 0. if self.last_ts is None else ts - self.last_ts
        gap = dt > self.max_gap
        fresh = self.belief is None
        prior = list(self.uniform)
        if not fresh and known:
            prior = self.predict(dt, v_camera_max_mm_s, process_margin_mm)
        usable = vis is not None and visual_quality >= .15
        mass = sum(v for i, v in enumerate(vis) if prior[i] > 0) if usable else 0.
        conflict = usable and known and not fresh and mass < self.min_mass
        bad = self.bad + 1 if not usable or conflict else 0
        relocalized = False
        recovery, streak = None, 0
        if conflict and self.hard_gate:
            recovery_prior = self.uniform
            if self.recovery_belief is not None:
                recovery_prior = self.predict(dt, v_camera_max_mm_s, process_margin_mm,
                                              belief=self.recovery_belief)
            recovery_prior = [(1 - self.recovery_mix) * p + self.recovery_mix * self.uniform[i]
                              for i, p in enumerate(recovery_prior)]
            recovery = self._norm([p * v for p, v in zip(recovery_prior, vis)])
            previous_recovery = self.recovery_belief
            if recovery is None:
                recovery, previous_recovery = vis, None
            streak = 1
            if previous_recovery is not None:
                old_index = max(range(len(previous_recovery)), key=lambda i: previous_recovery[i])
                new_index = max(range(len(recovery)), key=lambda i: recovery[i])
                old_row, old_col = divmod(old_index, self.w)
                new_row, new_col = divmod(new_index, self.w)
                separation = math.hypot(max(0, abs(new_col - old_col) - 1),
                                        max(0, abs(new_row - old_row) - 1)) * self.cell
                if separation <= v_camera_max_mm_s * dt + process_margin_mm:
                    streak = self.recovery_streak + 1
        if not usable:
            posterior = prior
            state = 'LOST' if bad >= self.bad_limit or gap else 'DEGRADED'
        elif conflict and self.hard_gate:
            posterior = prior
            state = 'LOST' if bad >= self.bad_limit else 'DEGRADED'
            if streak >= self.bad_limit:
                confirmed_origin = max(range(len(recovery)), key=lambda i: recovery[i])
                confirm_dt = 0.
                if previous_recovery is not None:
                    confirmed_origin = max(range(len(previous_recovery)), key=lambda i: previous_recovery[i])
                    confirm_dt = dt
                mode = [0.] * len(prior)
                mode[confirmed_origin] = 1.
                confirmed = self.predict(confirm_dt, v_camera_max_mm_s, process_margin_mm, belief=mode)
                posterior = self._norm([p if confirmed[i] > 0 else 0. for i, p in enumerate(recovery)])
                state, bad, relocalized = 'DEGRADED', 0, True
                recovery, streak = None, 0
        elif conflict and bad < self.bad_limit:
            posterior, state = prior, 'DEGRADED'
        elif conflict:
            posterior, state, bad, relocalized = vis, 'DEGRADED', 0, True
        else:
            if not self.hard_gate:
                prior = [(1 - self.recovery_mix) * p + self.recovery_mix * self.uniform[i]
                         for i, p in enumerate(prior)]
            posterior = self._norm([p * v for p, v in zip(prior, vis)])
            if posterior is None:
                posterior, bad, state = prior, bad + 1, 'DEGRADED'
            else:
                confident = max(posterior) >= self.tracking_mass and visual_quality >= .5
                state = 'TRACKING' if confident and known and self.hard_gate and not gap else 'DEGRADED'
        self.belief, self.last_ts, self.bad, self.state = posterior, ts, bad, state
        self.recovery_belief, self.recovery_streak = recovery, streak
        result = self.result(relocalized)
        result.update(dt_s=dt, reachable_visual_mass=mass, motion_bound_known=known,
                      hard_gate_enabled=self.hard_gate, bad_updates=bad)
        return result

    def reset_global(self, timestamp_s=None):
        self.initialize(timestamp_s=timestamp_s)
        self.state = 'LOST'
        return self.result()
