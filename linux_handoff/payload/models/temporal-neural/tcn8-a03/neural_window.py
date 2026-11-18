"""Pack sampled RGB frames oldest-to-newest for the fixed-window KModel."""


class FrameWindow:
    def __init__(self, np, window, sample_us, max_gap_us, tick_difference):
        if window not in (1, 4, 8) or not 0 < sample_us <= max_gap_us:
            raise ValueError('Invalid neural window sampling settings')
        self.window = window
        self.sample_us, self.max_gap_us = sample_us, max_gap_us
        self.tick_difference = tick_difference
        self.history = np.zeros((window * 3, 96, 160), dtype=np.uint8)
        self.tensor = np.zeros((1, window * 3, 96, 160), dtype=np.uint8)
        self.last_ts = None
        self.history_timestamps_us = [None] * window
        self.tensor_timestamps_us = [None] * window
        self.count = self.next_slot = self.resets = 0

    def due(self, timestamp):
        return self.last_ts is None or self.tick_difference(timestamp, self.last_ts) >= self.sample_us

    def append(self, frame, timestamp):
        if frame.shape != (1, 3, 96, 160) or not self.due(timestamp):
            raise ValueError('Expected the next sampled planar RGB frame')
        if self.last_ts is not None and self.tick_difference(timestamp, self.last_ts) > self.max_gap_us:
            self.count = self.next_slot = 0
            self.resets += 1
        offset = 3 * self.next_slot
        self.history[offset:offset + 3] = frame[0]
        self.history_timestamps_us[self.next_slot] = timestamp
        self.last_ts = timestamp
        self.next_slot = (self.next_slot + 1) % self.window
        self.count = min(self.count + 1, self.window)
        oldest = (self.next_slot - self.count) % self.window
        pad = self.window - self.count
        for i in range(self.window):
            slot = (oldest + max(0, i - pad)) % self.window
            offset = 3 * slot
            self.tensor[0, 3 * i:3 * i + 3] = self.history[offset:offset + 3]
            self.tensor_timestamps_us[i] = self.history_timestamps_us[slot]
        return self.tensor
