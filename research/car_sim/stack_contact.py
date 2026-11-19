"""Probe second-layer release in MuJoCo using the supplied CAD section dimensions."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np


OFFSETS_MM = (0, 5, 10, 14, 16, 20, 24)
FRICTIONS = (0.35, 0.7)
HEIGHTS_MM = (1, 5)
AZIMUTHS_DEG = (0, 45)
TIMESTEP = 0.0005
HORIZON_S = 1.5
CHECK_WINDOW_S = 0.15
DENSITY = 900


def scene(dimensions, friction, fixtures=""):
    geo = dimensions["geometry"]
    lower, upper, slope = (geo[name] for name in
                           ("bottom_cylinder", "top_cylinder", "middle_frustum"))
    if dimensions["units"] != "mm":
        raise ValueError("Expected millimetres")
    r0, r1 = lower["diameter"] / 2000, upper["diameter"] / 2000
    h0, h1 = lower["height"] / 1000, upper["height"] / 1000
    height = geo["overall_height"] / 1000
    if (not math.isclose(h0 + slope["height"] / 1000 + h1, height)
            or slope["bottom_diameter"] != lower["diameter"]
            or slope["top_diameter"] != upper["diameter"]):
        raise ValueError("Unexpected CAD axial profile")
    # Decompose the fused, non-convex axial profile into two cylinders plus
    # one convex frustum. All three share a free body and one density.
    vertices = []
    for z, r in ((h0, r0), (height - h1, r1)):
        for i in range(64):
            theta = math.tau * i / 64
            vertices.extend((r * math.cos(theta), r * math.sin(theta), z))
    frustum = " ".join(f"{x:.10g}" for x in vertices)

    def block(name):
        return f"""
          <body name="{name}"><freejoint/>
            <geom type="cylinder" size="{r0} {h0/2}" pos="0 0 {h0/2}"/>
            <geom type="mesh" mesh="frustum"/>
            <geom type="cylinder" size="{r1} {h1/2}" pos="0 0 {height-h1/2}"/>
          </body>"""

    xml = f"""
    <mujoco><compiler inertiafromgeom="true"/>
      <option timestep="{TIMESTEP}" gravity="0 0 -9.81" iterations="100">
        <flag multiccd="enable"/>
      </option>
      <default><geom density="{DENSITY}" friction="{friction} 0.0001 0.00001"
                     condim="3" solref="0.006 1" solimp="0.90 0.95 0.0002"/></default>
      <asset><mesh name="frustum" vertex="{frustum}"/></asset>
      <worldbody><geom name="floor" type="plane" size="1 1 0.1"/>
        {block("lower")}{block("upper")}
        {fixtures}
      </worldbody></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml), height


def probe(model, height, offset_mm, azimuth, gap_mm):
    data = mujoco.MjData(model)
    theta = math.radians(azimuth)
    data.qpos[7:10] = [offset_mm * math.cos(theta) / 1000,
                       offset_mm * math.sin(theta) / 1000,
                       height + gap_mm / 1000]
    lower, upper = (model.body(name).id for name in ("lower", "upper"))
    # The quaternions default to identity. Both bodies are free from t=0.
    frames = round(HORIZON_S / TIMESTEP)
    window = round(CHECK_WINDOW_S / TIMESTEP)
    extrema = dict(min_height_mm=math.inf, max_height_mm=-math.inf, max_tilt_deg=0.,
                   max_speed_mm_s=0., max_spin_deg_s=0.)
    supported_steps = 0
    for i in range(frames):
        mujoco.mj_step(model, data)
        if i < frames - window:
            continue
        pair = any({model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]} == {lower, upper}
                   and c.efc_address >= 0 for c in data.contact)
        supported_steps += int(pair)
        dz = (data.xpos[upper, 2] - data.xpos[lower, 2]) * 1000
        tilts = np.rad2deg(np.arccos(np.clip(data.xmat[[lower, upper], 8], -1., 1.)))
        extrema["min_height_mm"] = min(extrema["min_height_mm"], dz)
        extrema["max_height_mm"] = max(extrema["max_height_mm"], dz)
        extrema["max_tilt_deg"] = max(extrema["max_tilt_deg"], *tilts)
        speeds = data.qvel.reshape(2, 6)
        extrema["max_speed_mm_s"] = max(extrema["max_speed_mm_s"],
                                        np.linalg.norm(speeds[:, :3], axis=1).max() * 1000)
        extrema["max_spin_deg_s"] = max(extrema["max_spin_deg_s"],
                                        math.degrees(np.linalg.norm(speeds[:, 3:], axis=1).max()))
    stable = (supported_steps == window and extrema["max_tilt_deg"] <= 5
              and extrema["min_height_mm"] >= height * 1000 - 2
              and extrema["max_height_mm"] <= height * 1000 + 2
              and extrema["max_speed_mm_s"] <= 2 and extrema["max_spin_deg_s"] <= 3)
    mujoco.mj_forward(model, data)
    xy_mm = (data.xpos[upper, :2] - data.xpos[lower, :2]) * 1000
    return dict(release_xy_offset_mm=offset_mm, release_azimuth_deg=azimuth, release_gap_mm=gap_mm,
                stable_upright=bool(stable), final_xy_offset_mm=float(np.linalg.norm(xy_mm)),
                final_lower_xy_drift_mm=float(1000 * np.linalg.norm(data.xpos[lower, :2])),
                contact_fraction=supported_steps / window, **extrema)


def run(cad_receipt, output):
    dimensions = json.loads(cad_receipt.read_text(encoding="utf8"))
    results = []
    for friction in FRICTIONS:
        model, height = scene(dimensions, friction)
        for gap in HEIGHTS_MM:
            for offset in OFFSETS_MM:
                for azimuth in (AZIMUTHS_DEG if offset else (0,)):
                    trial = probe(model, height, offset, azimuth, gap)
                    results.append({"sliding_friction": friction, **trial})
    by_offset = []
    for offset in OFFSETS_MM:
        rows = [r for r in results if r["release_xy_offset_mm"] == offset]
        by_offset.append(dict(offset_mm=offset, stable=sum(r["stable_upright"] for r in rows),
                              trials=len(rows)))
    report = dict(scope="Ideal vertical release; independent synthetic contact study",
                  cad_sha256=hashlib.sha256(cad_receipt.read_bytes()).hexdigest(),
                  cad_sections_mm=dimensions["geometry"], mujoco_version=mujoco.__version__,
                  assumptions=dict(density_kg_m3=DENSITY, block_mass_kg=float(model.body("upper").mass[0]),
                                   geometry="Uniform-density rigid body: two cylinders and a 64-sided frustum",
                                   friction=list(FRICTIONS), contact_dimension=3, solref=[0.006, 1],
                                   solimp=[0.90, 0.95, 0.0002], initial_tilt_deg=0,
                                   initial_linear_and_angular_velocity=0, level_rigid_floor=True,
                                   timestep_s=TIMESTEP, horizon_s=HORIZON_S, scoring_window_s=CHECK_WINDOW_S),
                  stable_rule="Throughout final 150 ms: inter-block contact, both tilts <=5 degrees, "
                              "bottom-centre height difference within 2 mm of CAD height, "
                              "both linear speeds <=2 mm/s and angular speeds <=3 degrees/s",
                  limitations="Deterministic sensitivity sweep, no jaw slip, disturbances, camera, "
                              "measured inertia, material properties, edge radii or experimental tolerance.",
                  by_offset=by_offset, trials=results)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    return by_offset


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cad-receipt", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    print(json.dumps(run(args.cad_receipt, args.output)))
