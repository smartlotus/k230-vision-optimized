"""Host transport models only; no physical UART access."""
import random
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from k230.uart_adapter import UartAdapter as Legacy
from k230.uart_adapter_optimized import UartAdapter
from ports.protocol import StreamParser, encode_control


def config(**overrides):
    result = dict(enabled=True, pins_verified=True, uart_id=1, tx_pin=3, rx_pin=4)
    result.update(overrides)
    return result


class Transport:
    def __init__(self, limit=0):
        self.limit = limit
        self.outgoing = bytearray()
        self.incoming = b''
        self.calls = 0
        self.closed = False
        self.failure = False

    def write(self, chunk):
        self.calls += 1
        if self.failure:
            raise OSError('test write failure')
        count = min(len(chunk), self.limit)
        self.outgoing.extend(chunk[:count])
        return count

    def any(self):
        return len(self.incoming)

    def read(self, count):
        result, self.incoming = self.incoming[:count], self.incoming[count:]
        return result

    def deinit(self):
        self.closed = True


class OptimizedUartTests(unittest.TestCase):
    def test_lazy_buffer_disabled_and_pin_gates(self):
        for cfg in (config(enabled=False), config(pins_verified=False)):
            adapter = UartAdapter(cfg, transport=Transport())
            self.assertFalse(adapter.send_ack(1, 'ACK'))
            self.assertIsNone(adapter._storage)
            self.assertEqual(adapter.pending_bytes, 0)

    def test_wrap_partial_writes_and_exact_capacity(self):
        transport = Transport()
        adapter = UartAdapter(config(max_tx_buffer=80, max_write_bytes=13), transport)
        self.assertTrue(adapter._send(bytes(range(80))))
        storage = adapter._storage
        self.assertFalse(adapter._send(b'x'))
        transport.limit = 7
        adapter.drain(5)
        transport.limit = 0
        self.assertTrue(adapter._send(bytes(range(80, 115))))
        self.assertEqual(adapter.tx_buffer, bytes(range(35, 115)))
        transport.limit = 13
        self.assertTrue(adapter.drain(16))
        self.assertEqual(transport.outgoing, bytes(range(115)))
        self.assertIs(adapter._storage, storage)
        self.assertEqual(adapter.pending_bytes, 0)

    def test_random_fragmentation_preserves_fifo_and_bound(self):
        for seed in range(10):
            rng = random.Random(seed)
            transport = Transport()
            adapter = UartAdapter(config(max_tx_buffer=97, max_write_bytes=17), transport)
            accepted = bytearray()
            for _ in range(300):
                transport.limit = rng.randrange(20)
                frame = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 40)))
                if adapter._send(frame):
                    accepted.extend(frame)
                adapter.drain(rng.randrange(4))
                self.assertEqual(bytes(transport.outgoing) + adapter.tx_buffer, accepted)
                self.assertLessEqual(adapter.pending_bytes, 97)
            transport.limit = 17
            while adapter.pending_bytes:
                adapter.drain()
            self.assertEqual(transport.outgoing, accepted)

    def test_wire_bytes_match_legacy_ack_heartbeat_and_stop(self):
        outputs = []
        for implementation in (Legacy, UartAdapter):
            transport = Transport(3)
            adapter = implementation(config(), transport, clock_ms=lambda: 100)
            self.assertTrue(adapter.send_ack(40, 'ACK'))
            self.assertTrue(adapter.send_ack(40, 'DONE'))
            self.assertTrue(adapter.heartbeat(100))
            self.assertFalse(adapter.heartbeat(101))
            transport.incoming = encode_control(3, 'STOP')
            self.assertEqual(adapter.poll()[0]['command'], 'STOP')
            transport.limit = 256
            self.assertTrue(adapter.drain())
            outputs.append(bytes(transport.outgoing))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(StreamParser().feed(outputs[1])), 3)

    def test_stall_error_close_and_bounded_drain(self):
        transport = Transport()
        adapter = UartAdapter(config(), transport)
        adapter._send(b'x' * 100)
        calls = transport.calls
        self.assertFalse(adapter.drain())
        self.assertEqual(transport.calls, calls + 1)
        transport.limit = 1
        calls = transport.calls
        self.assertFalse(adapter.drain(1000))
        self.assertEqual(transport.calls, calls + 16)
        before = adapter.tx_buffer
        transport.failure = True
        adapter._flush()
        self.assertFalse(adapter.enabled)
        self.assertEqual(adapter.reason, 'UART_WRITE_ERROR')
        self.assertEqual(adapter.tx_buffer, before)
        adapter.close()
        self.assertTrue(transport.closed)
        self.assertEqual(adapter.tx_buffer, b'')
        self.assertIsNone(adapter._storage)

    def test_invalid_write_counts_leave_queue_intact(self):
        for result in (None, 0, -1, 1000, '1'):
            transport = Transport()
            adapter = UartAdapter(config(), transport)
            adapter._send(b'abcdef')
            with patch.object(transport, 'write', return_value=result):
                adapter._flush()
            self.assertEqual(adapter.tx_buffer, b'abcdef')

    def test_no_tail_snapshot_in_send_flush_or_drain(self):
        class NoSnapshot(UartAdapter):
            @property
            def tx_buffer(self):
                raise AssertionError('hot path copied queue')

            @tx_buffer.setter
            def tx_buffer(self, value):
                UartAdapter.tx_buffer.fset(self, value)
        adapter = NoSnapshot(config(), Transport(4))
        self.assertTrue(adapter.send_ack(1, 'ACK'))
        self.assertTrue(adapter.drain())

    def test_optimized_entry_rejects_host_before_hardware_import(self):
        from k230.main_optimized import main
        with self.assertRaises(SystemExit):
            main()

    def test_separate_script_rejects_host_and_cached_installation(self):
        source = Path(__file__).resolve().parents[1] / 'run_block_service_optimized.py'
        code = compile(source.read_text(encoding='utf-8'), str(source), 'exec')
        with self.assertRaisesRegex(SystemExit, 'Use CanMV'):
            exec(code, {'__name__': '__main__'})
        before = list(sys.path)
        with patch.object(sys, 'implementation', types.SimpleNamespace(name='micropython')):
            with self.assertRaisesRegex(SystemExit, 'already loaded'):
                exec(code, {'__name__': '__main__'})
        self.assertEqual(sys.path, before)

    def test_camera_constructor_failure_closes_uart(self):
        from k230.main_optimized import main
        transport = Transport()
        adapter = UartAdapter(config(), transport)
        implementation = types.SimpleNamespace(**{
            key: getattr(sys.implementation, key) for key in dir(sys.implementation)
            if not key.startswith('_')})
        implementation.name = 'micropython'
        with patch.object(sys, 'implementation', implementation), \
             patch('k230.vision_adapter.VisionAdapter', return_value=object()), \
             patch('k230.uart_adapter_optimized.UartAdapter', return_value=adapter), \
             patch('k230.camera_manager.CameraManager', side_effect=ValueError('bad camera')):
            with self.assertRaisesRegex(ValueError, 'bad camera'):
                main({'UART': config()})
        self.assertTrue(transport.closed)
