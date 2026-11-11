import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from ports.protocol import (MAX_LINE, StreamParser, encode_ack, encode_control,
                            encode_observation, escape_field)
from k230.uart_adapter import UartAdapter


def raw(body):
    checksum = 0
    for byte in body.encode("ascii"):
        checksum ^= byte
    return ("$%s*%02X\n" % (body, checksum)).encode("ascii")


def observation():
    return dict(status="OK", kind="COLOR", camera_id="CSI2", frame_id=4,
                timestamp_ms=100, target_id=1, coord_frame="IMAGE_PX",
                u_px=200, v_px=150, x_mm=None, y_mm=None, angle_deg=0,
                score_hint=76, stable=True, is_simulated=False,
                calibration_id=None, station="RAW", plane="TOP",
                image_profile=dict(width=640, height=480, crop=[0, 0, 640, 480],
                                   hmirror=False, vflip=False, rotation=0, pose_id="park1"),
                target_pose_valid=False, reason="待标定, H*unknown", raw_payload=None)


class FakeUart:
    def __init__(self, incoming=b"", write_limit=8):
        self.incoming = incoming
        self.outgoing = b""
        self.write_limit = write_limit
        self.closed = False

    def any(self):
        return len(self.incoming)

    def read(self, count):
        data, self.incoming = self.incoming[:count], self.incoming[count:]
        return data

    def write(self, data):
        data = data[:self.write_limit]
        self.outgoing += data
        return len(data)

    def deinit(self):
        self.closed = True


class ProtocolTests(unittest.TestCase):
    def test_control_round_trip_fragmented_noisy_replayed(self):
        parser = StreamParser()
        line = encode_control(2, "MODE", "COLOR", 1, "RAW")
        self.assertEqual(parser.feed(b"noise\x00" + line[:8], 0), [])
        result = parser.feed(line[8:], 1)
        self.assertEqual(result[0]["target_id"], 1)
        self.assertEqual(result[0]["mode"], "COLOR")
        self.assertEqual(parser.feed(line, 2), [])
        self.assertEqual(parser.feed(encode_control(1, "STOP"), 3), [])
        self.assertEqual(parser.feed(encode_control(3, "STOP"), 4)[0]["command"], "STOP")

    def test_checksum_and_fields_and_ranges(self):
        bad = [b"$C,1,PING,-,-,-*00\n", raw("C,1,MOVE,-,-,-"),
               raw("C,65536,PING,-,-,-"), raw("C,-1,PING,-,-,-"),
               raw("C,1,STOP,COLOR,-,-"), raw("C,1,MODE,COLOR,7,RAW"),
               raw("C,1,MODE,COLOR,0,RAW"), raw("C,1,PING,-,-"),
               raw("C,1,PING,-,-,"), raw("A,1,0,MOTOR,done")]
        for line in bad:
            with self.subTest(line=line):
                self.assertEqual(StreamParser().feed(line), [])
        with self.assertRaises(ValueError):
            encode_control(True, "PING")

    def test_timeout_overflow_resync_and_explicit_session(self):
        parser = StreamParser(timeout_ms=10)
        line = encode_control(1, "PING")
        parser.feed(line[:6], 0)
        self.assertEqual(parser.feed(line[6:], 11), [])
        self.assertEqual(parser.feed(b"$" + b"x" * (MAX_LINE+2), 12), [])
        self.assertLessEqual(len(parser.buffer), MAX_LINE)
        self.assertEqual(parser.feed(line, 13)[0]["seq"], 1)
        with self.assertRaises(ValueError):
            parser.reset_session()
        parser.reset_session(motors_disabled=True)
        self.assertEqual(parser.feed(line, 14)[0]["seq"], 1)

    def test_full_observation_round_trip_percent_profile(self):
        obs = observation()
        decoded = StreamParser().feed(encode_observation(7, obs))[0]
        for key, value in obs.items():
            self.assertEqual(decoded[key], value, key)
        self.assertEqual(StreamParser().feed(encode_ack(3, "ACK", "a,b*中", command_seq=1))[0]["reason"], "a,b*中")
        reordered = dict(reversed(list(obs["image_profile"].items())))
        other = dict(obs, image_profile=reordered)
        self.assertEqual(encode_observation(7, obs), encode_observation(7, other))

    def test_nonfinite_and_coordinate_semantics(self):
        for key, value in (("u_px", float("nan")), ("angle_deg", float("inf")),
                           ("stable", 1), ("is_simulated", None), ("score_hint", 101),
                           ("x_mm", 10), ("reason", "x" * 500)):
            obs = dict(observation(), **{key: value})
            if key == "reason":
                self.assertTrue(StreamParser().feed(encode_observation(1, obs)))
            else:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    encode_observation(1, obs)
        with self.assertRaises(ValueError):
            encode_observation(1, dict(observation(), image_profile={"a": "x"*500}))

    def test_vision_default_unavailable_and_long_unicode_diagnostics(self):
        from vision.common import observation as default_observation
        obs = default_observation("COLOR", status="UNAVAILABLE", reason="设备不可用,*" * 100)
        frame = encode_observation(1, obs)
        decoded = StreamParser().feed(frame)[0]
        self.assertEqual(decoded["status"], "UNAVAILABLE")
        self.assertIsNone(decoded["frame_id"])
        self.assertIsNone(decoded["camera_id"])
        self.assertTrue(decoded["reason"].endswith("..."))
        adapter = UartAdapter({})
        self.assertFalse(adapter.send_observation({"status": "invalid"}))
        self.assertEqual(adapter.reason, "UART_OBSERVATION_INVALID")

    def test_uart_default_disabled_pin_gate_and_bounded_io(self):
        fake = FakeUart(encode_control(2, "MODE", "QR"))
        adapter = UartAdapter({}, transport=fake)
        self.assertFalse(adapter.enabled)
        self.assertFalse(adapter.send_observation(observation()))
        self.assertEqual(fake.outgoing, b"")
        adapter = UartAdapter({"enabled": True}, transport=fake)
        self.assertFalse(adapter.enabled)
        config = dict(enabled=True, pins_verified=True, uart_id=1, tx_pin=3, rx_pin=4,
                      max_read_bytes=4, max_write_bytes=8, max_tx_buffer=768)
        adapter = UartAdapter(config, transport=fake, clock_ms=lambda: 0)
        controls = []
        for unused in range(20):
            controls += adapter.poll()
        self.assertEqual(controls[0]["mode"], "QR")
        self.assertTrue(adapter.send_observation(observation()))
        self.assertLessEqual(len(fake.outgoing), 8)
        for unused in range(100):
            adapter.poll()
        self.assertEqual(len(StreamParser().feed(fake.outgoing)), 1)
        adapter.close()
        self.assertTrue(fake.closed)
        self.assertEqual(adapter.poll(), [])

    def test_uart_ack_correlation_and_backpressure(self):
        fake = FakeUart(write_limit=0)
        config = dict(enabled=True, pins_verified=True, uart_id=1, tx_pin=3, rx_pin=4,
                      max_tx_buffer=80)
        adapter = UartAdapter(config, transport=fake)
        self.assertTrue(adapter.send_ack(40, "ACK"))
        self.assertTrue(adapter.send_ack(40, "DONE"))
        frames = StreamParser().feed(adapter.tx_buffer)
        self.assertEqual([f["command_seq"] for f in frames], [40, 40])
        self.assertNotEqual(frames[0]["seq"], frames[1]["seq"])
        self.assertFalse(adapter.send_observation(observation()))
        self.assertEqual(adapter.reason, "UART_TX_BACKPRESSURE")

    def test_uart_drain_is_bounded_and_retains_unsent(self):
        fake = FakeUart(write_limit=2)
        config = dict(enabled=True, pins_verified=True, uart_id=1, tx_pin=3, rx_pin=4)
        adapter = UartAdapter(config, transport=fake)
        adapter.send_ack(1, "STOPPED")
        pending = len(adapter.tx_buffer)
        self.assertFalse(adapter.drain(max_writes=1))
        self.assertEqual(len(adapter.tx_buffer), pending-2)
        fake.write_limit = 0
        pending = len(adapter.tx_buffer)
        self.assertFalse(adapter.drain())
        self.assertEqual(len(adapter.tx_buffer), pending)
        fake.write_limit = 8
        self.assertTrue(adapter.drain())
        self.assertEqual(StreamParser().feed(fake.outgoing)[0]["status"], "STOPPED")

    def test_display_and_jsonl_callbacks(self):
        import io
        import json
        from ports.display_port import ConsoleDisplay
        from ports.event_logger import EventLogger
        messages = []
        display = ConsoleDisplay(messages.append)
        display.show_task("156+123+516+231")
        display.show_stats({"picks": 12, "placements": 12})
        self.assertEqual(len(messages), 2)
        self.assertEqual(display.counters["picks"], 12)
        stream = io.StringIO()
        logger = EventLogger(stream=stream)
        self.assertTrue(logger({"event": "ACK", "completed": False}))
        self.assertFalse(json.loads(stream.getvalue())["completed"])
        self.assertFalse(EventLogger(stream=stream, max_record_bytes=5)({"event": "LONG"}))
        logger.close()
        self.assertFalse(logger({"event": "LATE"}))

    def test_python_c_shared_stream_vectors(self):
        compiler = shutil.which("gcc")
        if compiler is None and Path("C:/msys64/mingw64/bin/gcc.exe").exists():
            compiler = "C:/msys64/mingw64/bin/gcc.exe"
        if compiler is None:
            self.skipTest("C compiler unavailable: STM32 host parser UNKNOWN")
        root = Path(__file__).resolve().parents[1]
        obs = observation()
        valid_vision = encode_observation(5, obs)
        nan_vision = valid_vision[1:-4].decode("ascii").split(",")
        nan_vision[9] = "NaN"
        vectors = [b"noise", encode_control(1, "MODE", "COLOR", 2, "RAW"),
                   encode_control(1, "STOP"), raw("C,2,MODE,INVALID,1,RAW"),
                   raw("C,2,MODE,COLOR,0,RAW"), raw("C,2,STOP,-,-,"),
                   raw("C,65536,PING,-,-,-"), b"$" + b"z"*800 + b"\n",
                   encode_control(2, "STOP"), raw(",".join(nan_vision)),
                   valid_vision, encode_ack(6, "ACK", "ready", command_seq=2),
                   encode_ack(7, "DONE", "complete", command_seq=2)]
        stream = b"".join(vectors)
        expected = "".join("%s,%d\n" % (f["type"], f["seq"]) for f in StreamParser().feed(stream))
        self.assertEqual(expected, "C,1\nC,2\nV,5\nA,6\nA,7\n")
        with tempfile.TemporaryDirectory(prefix="k230-uart-") as temporary:
            exe = Path(temporary) / ("vectors.exe" if os.name == "nt" else "vectors")
            env = dict(os.environ)
            env["PATH"] = str(Path(compiler).parent) + os.pathsep + env.get("PATH", "")
            result = subprocess.run([compiler, "-std=c99", "-Wall", "-Wextra", "-Werror",
                                     str(root/"stm32/protocol.c"), str(root/"stm32/host_vectors.c"),
                                     "-lm", "-o", str(exe)], capture_output=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            checked = subprocess.run([str(exe), "--selftest"], capture_output=True, env=env)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertIn(b"SELFTEST_OK", checked.stdout)
            result = subprocess.run([str(exe)], input=stream, capture_output=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.decode("ascii").replace("\r\n", "\n"), expected)


if __name__ == "__main__":
    unittest.main()
