"""Bounded nonblocking UART observations; no motor command API."""
import time
from ports.protocol import StreamParser, encode_observation, encode_ack, encode_material_snapshot


def _clock_ms():
    return time.ticks_ms() if hasattr(time, "ticks_ms") else int(time.monotonic()*1000)


def _limit(config, key, default, maximum):
    value = config.get(key, default)
    return min(maximum, value) if type(value) is int and value > 0 else default


class UartAdapter:
    def __init__(self, config, transport=None, clock_ms=None):
        self.config = config.get("uart", config.get("UART", config)) if isinstance(config, dict) else getattr(config, "UART", {})
        self.clock_ms = clock_ms or _clock_ms
        self.max_read_bytes = _limit(self.config, "max_read_bytes", 512, 512)
        self.max_write_bytes = _limit(self.config, "max_write_bytes", 256, 256)
        self.max_tx_buffer = _limit(self.config, "max_tx_buffer", 3072, 3072)
        self.parser = StreamParser(timeout_ms=self.config.get("frame_timeout_ms", 250))
        self.transport = None
        self.enabled = False
        self.reason = "UART_DISABLED"
        self.tx_seq = 0
        self.tx_buffer = b""
        self.last_heartbeat_ms = None
        self.last_frame = None
        self.closed = False
        if self.config.get("enabled") is not True:
            return
        if self.config.get("pins_verified") is not True:
            self.reason = "UART_PINS_UNVERIFIED"
            return
        if any(self.config.get(k) is None for k in ("uart_id", "tx_pin", "rx_pin")):
            self.reason = "UART_PIN_CONFIG_MISSING"
            return
        if transport is not None:
            self.transport = transport
        else:
            try:
                from machine import UART, FPIOA
                uart_id = self.config["uart_id"]
                fpioa = FPIOA()
                fpioa.set_function(self.config["tx_pin"], getattr(FPIOA, "UART%d_TXD" % uart_id))
                fpioa.set_function(self.config["rx_pin"], getattr(FPIOA, "UART%d_RXD" % uart_id))
                self.transport = UART(uart_id, baudrate=self.config.get("baudrate", 115200), bits=8, parity=None, stop=1, timeout=0)
            except Exception:
                self.reason = "UART_INIT_FAILED"
                return
        self.enabled = True
        self.reason = "UART_READY"

    def _flush(self):
        if not self.enabled or self.closed or not self.tx_buffer:
            return
        try:
            chunk = self.tx_buffer[:self.max_write_bytes]
            count = self.transport.write(chunk)
            if isinstance(count, int) and 0 < count <= len(chunk):
                self.tx_buffer = self.tx_buffer[count:]
        except Exception:
            self.reason = "UART_WRITE_ERROR"
            self.enabled = False

    def _send(self, frame):
        self.last_frame = frame
        if not self.enabled or self.closed:
            return False
        limit = self.max_tx_buffer
        if len(self.tx_buffer) + len(frame) > limit:
            self.reason = "UART_TX_BACKPRESSURE"
            return False
        self.tx_buffer += frame
        self._flush()
        return True

    def _next_seq(self):
        if self.tx_seq > 65535:
            raise ValueError("UART session exhausted; disable motors before session reset")
        seq = self.tx_seq
        self.tx_seq += 1
        return seq

    def poll(self):
        now = self.clock_ms()
        self.parser.expire(now)
        if not self.enabled or self.closed:
            return []
        self._flush()
        try:
            available = self.transport.any()
            if not isinstance(available, int) or available <= 0:
                return []
            data = self.transport.read(min(available, self.max_read_bytes))
            return [frame for frame in self.parser.feed(data or b"", now) if frame["type"] == "C"]
        except Exception:
            self.reason = "UART_READ_ERROR"
            self.enabled = False
            return []

    def send_observation(self, observation):
        try:
            return self._send(encode_observation(self._next_seq(), observation))
        except (ValueError, TypeError, OverflowError, UnicodeError):
            self.reason = "UART_OBSERVATION_INVALID"
            return False

    def send_ack(self, seq, status, reason=""):
        try:
            return self._send(encode_ack(self._next_seq(), status, reason, command_seq=seq))
        except (ValueError, TypeError, OverflowError, UnicodeError):
            self.reason = "UART_ACK_INVALID"
            return False

    def send_material_snapshot(self, snapshot):
        try:
            return self._send(encode_material_snapshot(self._next_seq(), snapshot))
        except (KeyError, ValueError, TypeError, OverflowError, UnicodeError):
            self.reason = "UART_SNAPSHOT_INVALID"
            return False

    def heartbeat(self, now_ms):
        interval = self.config.get("heartbeat_ms", 1000)
        elapsed = (time.ticks_diff(now_ms, self.last_heartbeat_ms) if hasattr(time, "ticks_diff") else now_ms-self.last_heartbeat_ms) if self.last_heartbeat_ms is not None else interval
        if self.last_heartbeat_ms is None or elapsed >= interval:
            accepted = self.send_ack(0, "PONG", "HEARTBEAT")
            if accepted:
                self.last_heartbeat_ms = now_ms
            return accepted
        return False

    def close(self):
        self.closed = True
        self.enabled = False
        self.tx_buffer = b""
        if self.transport is not None and hasattr(self.transport, "deinit"):
            self.transport.deinit()

    def drain(self, max_writes=16):
        """Try pending bytes without sleeping; True means driver accepted bytes.

        A UART driver's accepted bytes are still not proof of MCU consumption.
        """
        budget = min(max_writes, 16) if type(max_writes) is int and max_writes > 0 else 0
        for unused in range(budget):
            before = len(self.tx_buffer)
            if not before or not self.enabled or self.closed:
                break
            self._flush()
            if len(self.tx_buffer) >= before:
                break
        return not self.tx_buffer
