"""Run from the project root: python -m sim.run_mission --task 156+123+516+231."""
import argparse
import json
from pathlib import Path

from core.mission_state import MissionEngine
from core.task_parser import parse_task
from sim.camera_sim import CameraSim
from sim.field_sim import FieldSim
from sim.motion_sim import MotionSim

FAULTS = ("missing-color", "wrong-color", "carrier-full", "ring-not-found",
          "timeout", "dropped-frame", "ack-loss", "obstacle")


class DisplaySim:
    def __init__(self):
        self.task, self.counters = None, None

    def show_task(self, qr):
        self.task = qr

    def show_stats(self, counters):
        self.counters = dict(counters)


def run_mission(task="156+123+516+231", start="START_1", capacity=3, inject=None):
    if inject is not None and inject not in FAULTS:
        raise ValueError("Unknown fault injection: %s" % inject)
    plan = parse_task(task)
    field = FieldSim(plan, capacity, start)
    motion = MotionSim(field, inject)
    camera, display = CameraSim(task, inject), DisplaySim()
    engine = MissionEngine(motion, camera, display=display)
    summary = engine.run(task, capacity, start)
    summary["field_complete"] = field.is_complete()
    summary["max_carrier_occupancy"] = motion.max_carrier_occupancy
    if summary["status"] == "DONE" and not summary["field_complete"]:
        summary.update(status="ERROR", reason="FIELD_FINAL_AUDIT_FAILED")
    return {"summary": summary, "events": engine.events, "field": field.snapshot(),
            "transitions": field.transitions, "receipts": motion.receipts,
            "display": {"task": display.task, "counters": display.counters},
            "source": "DETERMINISTIC_SIM_ONLY", "injection": inject}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="156+123+516+231")
    parser.add_argument("--start", choices=("START_1", "START_2"), default="START_1")
    parser.add_argument("--capacity", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--inject", choices=FAULTS)
    parser.add_argument("--output", help="Write the complete audit report as JSON")
    args = parser.parse_args(argv)
    try:
        report = run_mission(args.task, args.start, args.capacity, args.inject)
    except ValueError as exc:
        print(json.dumps({"summary": {"status": "ERROR", "reason": str(exc), "is_simulated": True}}))
        return 2
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for event in report["events"]:
        if event["event"] == "COMMAND":
            print(json.dumps(event, ensure_ascii=False))
    print(json.dumps({"summary": report["summary"]}, ensure_ascii=False))
    return 0 if report["summary"]["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
