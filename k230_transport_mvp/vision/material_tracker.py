"""Bounded continuity of <=3 measured tops; never physical instance identity."""
from .common import finite


class MaterialTracker:
    def __init__(self, frames=3, motion_px=4, gap_ms=300, match_px=30):
        if (type(frames) is not int or not 3 <= frames <= 20 or not finite(motion_px) or
                not 0 < motion_px <= 30 or type(gap_ms) is not int or not 0 < gap_ms <= 1000 or
                not finite(match_px) or not motion_px <= match_px <= 60):
            raise ValueError('Invalid bounded material tracking parameters')
        self.frames, self.motion_px, self.gap_ms, self.match_px = frames, motion_px, gap_ms, match_px
        self.reset()

    def reset(self):
        self.key = None
        self.last_frame = self.last_time = None
        self.tracks = []

    def invalidate(self):
        """Forget continuity while retaining frame/time high-water marks."""
        self.tracks = []

    def update(self, key, frame_id, timestamp_ms, objects):
        if (type(frame_id) is not int or not 0 <= frame_id <= 4294967295 or
                type(timestamp_ms) is not int or not 0 <= timestamp_ms <= 4294967295 or
                not isinstance(objects, (list, tuple)) or len(objects) > 3):
            self.invalidate()
            raise ValueError('Invalid material tracker input')
        for row in objects:
            if (type(row.get('color_id')) is not int or not 1 <= row['color_id'] <= 6 or
                    row.get('geometry_valid') is not True or
                    any(not finite(row.get(k)) for k in ('u_px','v_px','semimajor_px','semiminor_px')) or
                    not 0 < row['semiminor_px'] <= row['semimajor_px']):
                self.invalidate()
                raise ValueError('Measured known-color geometry required')
        if key != self.key:
            self.reset()
            self.key = key
        if self.last_frame is not None:
            gap = (timestamp_ms-self.last_time) & 0xffffffff
            if frame_id <= self.last_frame or not gap or gap > 0x7fffffff:
                self.invalidate()
                raise ValueError('Material source replay or rollback')
            if gap > self.gap_ms:
                self.invalidate()
        self.last_frame, self.last_time = frame_id, timestamp_ms
        choices = []
        for row in objects:
            matches = []
            for i, old in enumerate(self.tracks):
                distance = (row['u_px']-old['row']['u_px'])**2 + (row['v_px']-old['row']['v_px'])**2
                if (row['color_id'] == old['row']['color_id'] and distance <= self.match_px**2 and
                        all(1/1.35 <= row[k]/old['row'][k] <= 1.35
                            for k in ('semimajor_px','semiminor_px'))):
                    matches.append(i)
            choices.append(matches)
        updated, output = [], []
        for row, matches in zip(objects, choices):
            unique = (len(matches) == 1 and sum(matches[0] in other for other in choices) == 1)
            history = list(self.tracks[matches[0]]['history']) if unique else []
            point = (row['u_px'],row['v_px'],row['semimajor_px'],row['semiminor_px'])
            # Total window displacement and size change, not just consecutive deltas.
            if any((point[0]-p[0])**2+(point[1]-p[1])**2 > self.motion_px**2 or
                   any(not 1/1.2 <= point[k]/p[k] <= 1.2 for k in (2,3)) for p in history):
                history = []
            history.append(point)
            history = history[-self.frames:]
            current = dict(row, stable=len(history) >= self.frames)
            updated.append({'row':dict(row),'history':history})
            output.append(current)
        self.tracks = updated  # Missing objects are removed immediately.
        return output
