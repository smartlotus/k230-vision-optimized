"""Opt-in block tracker; native detection and acceptance filters are inherited."""
from .block_detector import BlockDetector as LegacyBlockDetector
from .common import elapsed_ms


class BlockDetector(LegacyBlockDetector):
    def update(self, candidates, frame_id, timestamp_ms):
        p = self.p
        if self.last_frame is not None:
            gap = elapsed_ms(timestamp_ms, self.last_time)
            if frame_id <= self.last_frame or gap <= 0 or gap > p['max_gap_ms']:
                self.tracks = []
        self.last_frame, self.last_time = frame_id, timestamp_ms
        old = self.tracks
        used = set()
        updated = []
        match_limit = p['track_distance_px'] ** 2
        motion_limit = p['stable_motion_px'] ** 2
        frames = p['stable_frames']
        for candidate in candidates:
            current = dict(candidate)
            center = current['center']
            cx, cy = center
            best_index = None
            best_distance = second_distance = None
            # Only the nearest two distances affect the legacy ambiguity rule.
            # Strict comparison preserves its stable-sort order for equal ties.
            for index, track in enumerate(old):
                if index in used or track['target_id'] != current['target_id']:
                    continue
                tx, ty = track['center']
                distance = (cx - tx) ** 2 + (cy - ty) ** 2
                if distance <= match_limit:
                    if best_distance is None or distance < best_distance:
                        second_distance = best_distance
                        best_distance = distance
                        best_index = index
                    elif second_distance is None or distance < second_distance:
                        second_distance = distance
            if best_index is not None and (second_distance is None or
                                          second_distance - best_distance > 16):
                previous = old[best_index]
                used.add(best_index)
                history = previous['_points']
                for px, py in history:
                    if (cx - px) ** 2 + (cy - py) ** 2 > motion_limit:
                        history = []
                        break
                current['track_id'] = previous['track_id']
            else:
                current['track_id'] = self.next_id
                self.next_id += 1
                history = []
            # Keep the legacy window expression and its exact list sizing.
            points = (history + [center])[-frames:]
            current['_points'] = points
            current['stable'] = len(points) >= frames and current['eligible']
            current['frame_id'] = frame_id
            current['timestamp_ms'] = timestamp_ms
            updated.append(current)
        self.tracks = updated
        return [{k: v for k, v in track.items() if k != '_points'} for track in updated]
