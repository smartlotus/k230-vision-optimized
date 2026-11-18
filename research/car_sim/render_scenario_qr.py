"""Render a campaign QR case through the Blender vehicle's front camera."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_scene import prepare_round, shoot, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    case = json.loads(args.case.read_text(encoding="utf-8"))
    scene, rig, front, _, blocks, plan, records, _, cad = prepare_round(
        args.hub, case["layout_task"], case["start_id"], case["orders"], args.out / "texture.png"
    )
    scene.view_settings.exposure += case["exposure_ev"]
    scene.frame_set(records[0]["end"])
    shoot(scene, front, args.out / "qr.png")
    write_json(args.out / "render_state.json", {
        "layout_task": plan["task"], "start_id": plan["start_id"], "frame": scene.frame_current,
        "camera": front.name, "world_from_camera": [list(row) for row in front.matrix_world],
        "body_position_m": list(rig["root"].matrix_world.translation),
        "orders": case["orders"], "inventory": sorted(blocks), "cad": cad,
        "render_device": scene.cycles.device, "exposure_ev": scene.view_settings.exposure,
    })


if __name__ == "__main__":
    main()
