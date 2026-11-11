"""Shared byte streams and actual compiled C receiver, without board access."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from ports.protocol import (MAX_LINE, StreamParser, encode_ack, encode_control,
                            encode_material_snapshot, encode_observation)
from k230.uart_adapter import UartAdapter


def object_row(color=1, u=200):
    return dict(color_id=color, u_px=u, v_px=150, semimajor_px=25,
                semiminor_px=20, angle_rad=0.4, score_hint=None,
                stable=True, geometry_valid=True)


def snapshot(count=1, frame=10, timestamp=1000000, **changes):
    value = dict(version=1, status="OK" if count else "NOT_FOUND", camera_id="CSI2",
                 frame_id=frame, timestamp_ms=timestamp, profile_id="a"*64,
                 source_session="b"*32, width=640, height=360, coord_frame="IMAGE_PX",
                 is_simulated=False, complete=True, reason=None,
                 objects=[object_row(i+1, 100+i*120) for i in range(count)])
    value.update(changes)
    return value


def raw(parts):
    body = ",".join(str(v) for v in parts)
    checksum = 0
    for byte in body.encode("ascii"):
        checksum ^= byte
    return ("$%s*%02X\n" % (body, checksum)).encode("ascii")


def parts(value=None, seq=1):
    return encode_material_snapshot(seq, value or snapshot())[1:-4].decode("ascii").split(",")


class MaterialProtocolTests(unittest.TestCase):
    def test_zero_one_two_three_and_same_color_round_trip(self):
        for count in range(4):
            value = snapshot(count)
            for item in value["objects"]:
                item["color_id"] = 5  # Colors are not unique instance IDs.
            encoded = encode_material_snapshot(3, value)
            decoded = StreamParser().feed(encoded)[0]
            self.assertEqual(decoded["objects"], value["objects"])
            self.assertEqual(decoded["count"], count)
            self.assertEqual(decoded["source_session"], "b"*32)
            self.assertLessEqual(len(encoded), MAX_LINE)

    def test_invalid_headers_objects_and_explicit_evidence(self):
        mutations = [(2,"2"),(3,"UNCALIBRATED"),(4,"CSI0"),(5,"-"),(6,"4294967296"),
                     (7,"width640"),(8,"0"),(9,"4097"),(10,"ROBOT_MM"),(11,"true"),
                     (12,"0"),(13,"4"),(15,"run"),(16,"7"),(17,"NaN"),(18,"inf"),
                     (19,"-"),(20,"0"),(21,str(math.pi)),(22,"101"),(23,"2"),(24,"0"),
                     (17,"5"),(19,"19")]  # Clipped ellipse and major < minor.
        for index, value in mutations:
            candidate = parts()
            candidate[index] = value
            with self.subTest(index=index, value=value):
                self.assertEqual(StreamParser().feed(raw(candidate)), [])
        for key in ("complete", "is_simulated"):
            with self.assertRaises(ValueError):
                encode_material_snapshot(1, snapshot(**{key: None}))
        value = snapshot()
        del value["objects"][0]["stable"]
        with self.assertRaises(ValueError):
            encode_material_snapshot(1, value)
        with self.assertRaises(ValueError):
            encode_material_snapshot(1, snapshot(4))

    def test_failure_is_empty_unknown_never_successful_truncation(self):
        for reason in ("SCENE_LIMIT:4", "SCAN_INCOMPLETE", "THRESHOLD_REQUIRED"):
            value = snapshot(0, status="UNKNOWN", complete=False, reason=reason)
            result = StreamParser().feed(encode_material_snapshot(1, value))[0]
            self.assertFalse(result["complete"])
            self.assertEqual(result["status"], "UNKNOWN")
            self.assertEqual(result["objects"], [])
        for changes in (dict(status="OK"), dict(status="UNKNOWN", complete=False),
                        dict(status="NOT_FOUND", complete=False)):
            with self.assertRaises(ValueError):
                encode_material_snapshot(1, snapshot(0, **changes))

    def test_snapshot_uart_stays_disabled_and_sequence_shared(self):
        class Transport:
            output = b""
            def write(self, data):
                self.output += data
                return len(data)
            def any(self): return 0
        fake = Transport()
        adapter = UartAdapter({}, transport=fake)
        self.assertFalse(adapter.send_material_snapshot(snapshot()))
        self.assertFalse(fake.output)
        adapter = UartAdapter(dict(enabled=True, pins_verified=True, uart_id=1,
                                   tx_pin=3, rx_pin=4), transport=fake)
        self.assertTrue(adapter.send_material_snapshot(snapshot()))
        self.assertTrue(adapter.send_ack(8, "ACK"))
        self.assertTrue(adapter.drain())
        result = StreamParser().feed(fake.output)
        self.assertEqual([(row["type"], row["seq"]) for row in result], [("S",0),("A",1)])
        self.assertFalse(adapter.send_material_snapshot({}))
        self.assertEqual(adapter.reason, "UART_SNAPSHOT_INVALID")


class MaterialReceiverCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("gcc")
        if compiler is None and Path("C:/msys64/mingw64/bin/gcc.exe").exists():
            compiler = "C:/msys64/mingw64/bin/gcc.exe"
        if compiler is None:
            raise unittest.SkipTest("No gcc: C receiver verification UNKNOWN")
        cls.temp = tempfile.TemporaryDirectory(prefix="k230-material-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.env = dict(os.environ)
        cls.env["PATH"] = str(Path(compiler).parent)+os.pathsep+cls.env.get("PATH", "")
        root = Path(__file__).resolve().parents[1]
        cls.runner = Path(cls.temp.name)/"receiver.exe"
        cls.stream_runner = Path(cls.temp.name)/"stream.exe"
        for target, sources in ((cls.runner, ["protocol.c","material_receiver.c","host_material.c"]),
                                (cls.stream_runner, ["protocol.c","host_vectors.c"])):
            compiled = subprocess.run([compiler,"-std=c99","-Wall","-Wextra","-Werror",
                                       *[str(root/"stm32"/name) for name in sources],"-lm","-o",str(target)],
                                      capture_output=True,env=cls.env)
            if compiled.returncode:
                raise AssertionError(compiled.stderr.decode(errors="replace"))

    def run_receiver(self, commands):
        data = "".join(commands).encode("ascii")
        result = subprocess.run([str(self.runner)],input=data,capture_output=True,env=self.env)
        self.assertEqual(result.returncode,0,result.stderr)
        return [json.loads(line) for line in result.stdout.decode().splitlines()]

    @staticmethod
    def feed(now, seq, value):
        return "F %d %s\n" % (now,encode_material_snapshot(seq,value).hex())

    def test_shared_mixed_stream_and_negative_vectors(self):
        # Accepted C/A/V bytes remain compatible, while S validates identically.
        from vision.common import observation
        stream = [b"noise",encode_control(1,"PING"),encode_ack(2,"PONG"),
                  encode_observation(3,observation("TOP",status="UNAVAILABLE")),
                  encode_material_snapshot(4,snapshot(3))]
        for index, token in ((13,"4"),(17,"nan"),(8,"0"),(15,"Z"*32),(21,"-1")):
            item = parts(seq=5); item[index] = token; stream.append(raw(item))
        item = parts(seq=5); item.append("extra"); stream.append(raw(item))
        stream += [b"$"+b"x"*800+b"\n",encode_material_snapshot(5,snapshot(0)),
                   encode_material_snapshot(5,snapshot()),encode_material_snapshot(6,snapshot())]
        data = b"".join(stream)
        expected = "".join("%s,%d\n"%(r["type"],r["seq"]) for r in StreamParser().feed(data))
        actual = subprocess.run([str(self.stream_runner)],input=data,capture_output=True,env=self.env)
        self.assertEqual(actual.returncode,0,actual.stderr)
        self.assertEqual(actual.stdout.decode().replace("\r\n","\n"),expected)
        self.assertEqual(expected,"C,1\nA,2\nV,3\nS,4\nS,5\nS,6\n")

    def test_snapshot_copy_three_to_one_to_zero_and_unknown(self):
        records = self.run_receiver([self.feed(10,1,snapshot(3)),
                                    "F 20 "+b"$junk".hex()+"\n",
                                    self.feed(30,2,snapshot(1,11,1000001)),
                                    self.feed(40,3,snapshot(0,12,1000002)),
                                    self.feed(50,4,snapshot(0,13,1000003,status="UNKNOWN",
                                                          complete=False,reason="SCENE_LIMIT:4"))])
        self.assertEqual([r["count"] for r in records],[3,3,1,0,0])
        self.assertEqual(records[0]["objects"],records[1]["objects"])
        self.assertEqual(records[0]["usable"],1)
        self.assertEqual(records[-2]["status"],2)
        self.assertEqual(records[-1]["status"],3)
        self.assertEqual(records[-1]["usable"],0)

    def test_replayed_frame_timestamp_heartbeat_and_silence_do_not_renew(self):
        commands = [self.feed(10,1,snapshot()),self.feed(20,2,snapshot()),
                    self.feed(30,3,snapshot(frame=9,timestamp=1000001)),
                    self.feed(40,4,snapshot(frame=11,timestamp=1000000)),
                    self.feed(50,5,snapshot(frame=12,timestamp=999999)),
                    "F 90 "+encode_ack(6,"PONG").hex()+"\n","P 110\n","P 111\n","P 120\n"]
        rows = self.run_receiver(commands)
        self.assertEqual([r["accepted"] for r in rows],[1,0,0,0,0,0,0,0,0])
        self.assertEqual(rows[6]["usable"],1)  # age == TTL
        self.assertEqual(rows[7]["status"],7)
        self.assertFalse(rows[7]["fresh"])
        self.assertEqual(rows[8]["count"],0)

    def test_source_bindings_fail_closed_and_sim_unstable_unknown_color(self):
        cases = [snapshot(camera_id="CSI1"),snapshot(profile_id="c"*64),
                 snapshot(width=320),snapshot(source_session="c"*32)]
        for value in cases:
            rows = self.run_receiver([self.feed(0,1,snapshot()),self.feed(1,2,value)])
            self.assertEqual(rows[-1]["status"],6)
            self.assertFalse(rows[-1]["usable"])
            self.assertEqual(rows[-1]["count"],0)
        for change in (dict(is_simulated=True),dict(objects=[dict(object_row(),stable=False)]),
                       dict(objects=[object_row(0)])):
            row = self.run_receiver([self.feed(0,1,snapshot(**change))])[0]
            self.assertEqual(row["status"],1)
            self.assertFalse(row["usable"])

    def test_local_and_source_clock_wrap_and_explicit_new_session(self):
        rows = self.run_receiver([self.feed(4294967280,65534,snapshot(timestamp=4294967280)),
                                 self.feed(5,65535,snapshot(frame=11,timestamp=5)),"P 105\n","P 106\n",
                                 "R 0 "+"c"*32+"\n","R 1 "+"b"*32+"\n",
                                 "R 1 "+"c"*32+"\n",
                                 self.feed(110,0,snapshot(frame=0,timestamp=0,source_session="c"*32)),
                                 self.feed(111,1,snapshot(frame=12,timestamp=6))])
        self.assertEqual([r["accepted"] for r in rows],[1,1,0,0,0,0,1,1,0])
        self.assertTrue(rows[2]["usable"])
        self.assertEqual(rows[3]["status"],7)
        self.assertTrue(rows[7]["usable"])
        self.assertEqual(rows[8]["status"],6)  # Old source session cannot revive.

    def test_partial_timeout_noise_corruption_and_owned_data(self):
        wire = encode_material_snapshot(1,snapshot(3))
        commands = ["F 0 "+wire[:20].hex()+"\n","F 51 "+wire[20:].hex()+"\n",
                    "F 52 "+(b"noise"+wire[:20]).hex()+"\n","F 53 "+wire[20:].hex()+"\n",
                    "F 54 "+b"$C,2,PING,-,-,-*00\n".hex()+"\n","P 154\n"]
        rows = self.run_receiver(commands)
        self.assertEqual([r["accepted"] for r in rows],[0,0,0,1,0,0])
        self.assertEqual(rows[3]["objects"],rows[4]["objects"])
        self.assertEqual(rows[-1]["status"],7)

    def test_internal_nul_never_becomes_a_field_separator(self):
        wire = bytearray(encode_material_snapshot(1,snapshot()))
        commas = [i for i,value in enumerate(wire) if value == ord(",")]
        # Two replacements preserve XOR, isolating syntax from checksum failure.
        wire[commas[0]] = wire[commas[1]] = 0
        self.assertEqual(StreamParser().feed(bytes(wire)),[])
        actual = subprocess.run([str(self.stream_runner)],input=bytes(wire),capture_output=True,env=self.env)
        self.assertEqual(actual.returncode,0,actual.stderr)
        self.assertEqual(actual.stdout,b"")

    def test_float_storage_does_not_promote_a_degenerate_ellipse(self):
        value = snapshot(frame=11,timestamp=1000001,
                         objects=[dict(object_row(),semimajor_px=1e-300,semiminor_px=1e-300)])
        wire = encode_material_snapshot(2,value)
        self.assertTrue(StreamParser().feed(wire))  # Valid double, not float geometry.
        rows = self.run_receiver([self.feed(0,1,snapshot()),"F 1 "+wire.hex()+"\n"])
        self.assertFalse(rows[-1]["usable"])
        self.assertEqual(rows[-1]["count"],0)
        self.assertEqual(rows[-1]["status"],3)

    def test_reason_escape_conformance(self):
        stream = []
        for seq,token in enumerate(("%FF","%C0%AF","%ED%A0%80","%F4%90%80%80",
                                    "%E4%B8%AD","reason%2Cmore","%","%0g"),1):
            item = parts(seq=seq); item[14] = token
            stream.append(raw(item))
        data = b"".join(stream)
        expected = "".join("S,%d\n"%r["seq"] for r in StreamParser().feed(data))
        actual = subprocess.run([str(self.stream_runner)],input=data,capture_output=True,env=self.env)
        self.assertEqual(actual.returncode,0,actual.stderr)
        self.assertEqual(actual.stdout.decode().replace("\r\n","\n"),expected)

    def test_measured_producer_to_real_c_cache_with_stability_and_unknown(self):
        from k230.tuning.material import MaterialSnapshotProducer
        from k230.tuning.schema import defaults,image_profile
        active=defaults()
        for color in range(1,7):active['contexts']['TOP_BUFFER']['thresholds'][str(color)]=[0,30,-5,5,-5,5]
        p=MaterialSnapshotProducer(active,active,"b"*32,[1,2,3,4,5,6],"SCALAR_FULL_BOUNDARY")
        # Reference runner uses fixed test digest 'a'; substitute only the
        # expected digest token in its configuration by compiling a fixture copy.
        root=Path(__file__).resolve().parents[1]
        compiler=shutil.which("gcc") or "C:/msys64/mingw64/bin/gcc.exe"
        driver=Path(self.temp.name)/"producer_driver.c"
        driver.write_text((root/"stm32/host_material.c").read_text().replace("a"*64,p.profile_id))
        runner=Path(self.temp.name)/"producer.exe"
        result=subprocess.run([compiler,"-std=c99","-Wall","-Wextra","-Werror","-I",str(root/"stm32"),
                               str(root/"stm32/protocol.c"),str(root/"stm32/material_receiver.c"),str(driver),
                               "-lm","-o",str(runner)],capture_output=True,env=self.env)
        self.assertEqual(result.returncode,0,result.stderr)
        commands=[]
        fits=[dict(accepted=True,reason="GEOMETRY_ONLY",color_id=c,center=[u,150],
                   semiaxes=[20,16],angle_rad=.2) for c,u in ((1,100),(1,250),(5,400))]
        for n in range(1,5):
            meta=dict(camera_id=2,frame_id=n,timestamp_ms=n*100,is_simulated=False,
                      image_profile=image_profile(active),active_revision=0,width=640,height=360)
            geometry=dict(material_fits=fits,raw_material_fit_count=3,
                          material_complete=True,material_limit_exceeded=False,
                          material_algorithm_revision=0,material_scan_colors=[1,2,3,4,5,6],
                          material_by_color=[dict(color_id=i,scan_complete=True,threshold_available=True,
                              top_count=sum(row['color_id']==i for row in fits)) for i in range(1,7)])
            if n==4:geometry["material_complete"]=False
            value=p.build(meta,geometry)
            commands.append(self.feed(n*10,n,value))
        result=subprocess.run([str(runner)],input="".join(commands).encode(),capture_output=True,env=self.env)
        self.assertEqual(result.returncode,0,result.stderr)
        rows=[json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([row["usable"] for row in rows],[0,0,1,0])
        self.assertEqual([row["count"] for row in rows],[3,3,3,0])
        self.assertEqual(rows[-1]["status"],3)


if __name__ == "__main__":
    unittest.main()
