"""Opt-in UART TX ring; legacy adapter and wire protocol remain unchanged."""
from .uart_adapter import UartAdapter as LegacyUartAdapter


class UartAdapter(LegacyUartAdapter):
    """Single-threaded queue with fixed capacity, allocated on first enqueue.

    UART writes still use bounded bytes chunks. No queued tail is copied after
    a partial write. Reading tx_buffer creates a diagnostic snapshot; runtime
    code uses pending_bytes instead.
    """
    @property
    def pending_bytes(self):
        return self._used

    @property
    def tx_buffer(self):
        if not self._used:
            return b""
        first = min(self._used, self.max_tx_buffer - self._head)
        result = bytes(self._view[self._head:self._head + first])
        if first < self._used:
            result += bytes(self._view[:self._used - first])
        return result

    @tx_buffer.setter
    def tx_buffer(self, value):
        # The inherited constructor and close() both reset through this setter.
        if len(value) > self.max_tx_buffer:
            raise ValueError("UART queue capacity exceeded")
        self._storage = None
        self._view = None
        self._head = 0
        self._used = 0
        if value:
            self._append(value)

    def _append(self, frame):
        if not frame:
            return
        if self._storage is None:
            self._storage = bytearray(self.max_tx_buffer)
            self._view = memoryview(self._storage)
        tail = (self._head + self._used) % self.max_tx_buffer
        first = min(len(frame), self.max_tx_buffer - tail)
        source = memoryview(frame)
        self._view[tail:tail + first] = source[:first]
        if first < len(frame):
            self._view[:len(frame) - first] = source[first:]
        self._used += len(frame)

    def _send(self, frame):
        self.last_frame = frame
        if not self.enabled or self.closed:
            return False
        if self._used + len(frame) > self.max_tx_buffer:
            self.reason = "UART_TX_BACKPRESSURE"
            return False
        self._append(frame)
        self._flush()
        return True

    def _flush(self):
        if not self.enabled or self.closed or not self._used:
            return
        try:
            size = min(self._used, self.max_write_bytes,
                       self.max_tx_buffer - self._head)
            chunk = bytes(self._view[self._head:self._head + size])
            count = self.transport.write(chunk)
            if isinstance(count, int) and 0 < count <= size:
                self._head = (self._head + count) % self.max_tx_buffer
                self._used -= count
                if not self._used:
                    self._head = 0
        except Exception:
            self.reason = "UART_WRITE_ERROR"
            self.enabled = False

    def drain(self, max_writes=16):
        """Bounded driver acceptance only; not proof of MCU consumption."""
        budget = min(max_writes, 16) if type(max_writes) is int and max_writes > 0 else 0
        for unused in range(budget):
            before = self._used
            if not before or not self.enabled or self.closed:
                break
            self._flush()
            if self._used >= before:
                break
        return not self._used
