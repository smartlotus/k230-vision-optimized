"""Regenerate the annotated views from build_scene.py and mission_vision.py outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw, ImageFont


PAPER = "#FAF9F5"
INK = "#192D36"
MUTED = "#4B6067"
LENS = "#FFB648"
PIVOT = "#40DBDA"
RULE = "#B8C4C1"
GREEN = "#78F2A4"


def font(size, bold=False):
    directory = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    path = os.environ.get("CAR_SIM_FONT") or str(directory / ("msyhbd.ttc" if bold else "msyh.ttc"))
    return ImageFont.truetype(path, size)


def text(draw, xy, label, size, color=INK, bold=False):
    draw.text(xy, label, font=font(size, bold), fill=color, anchor="lt")


def reticle(draw, xy, color, radius):
    x, y = map(round, xy)
    for stroke, width in ((INK, 5), (color, 2)):
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), outline=stroke, width=width)
        for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
            draw.line((x+dx*(radius-3), y+dy*(radius-3),
                       x+dx*(radius+5), y+dy*(radius+5)), fill=stroke, width=width)


def projected_points(blender, scene):
    expression = """
import bpy, json
from bpy_extras.object_utils import world_to_camera_view
s = bpy.context.scene
c = s.objects['Plan / front left, right up']
rows = {}
for frame in (1, 40):
    s.frame_set(frame)
    row = []
    for key in ('ROAD CAMERA | optical centre', 'AZIMUTH | turntable'):
        p = world_to_camera_view(s, c, s.objects[key].matrix_world.translation)
        row.append([p.x, 1-p.y])
    rows[str(frame)] = row
print('POINTS=' + json.dumps(rows))
"""
    command = [str(blender), "-b", str(scene), "--python-exit-code", "1", "--python-expr", expression]
    proc = subprocess.run(command, capture_output=True, check=True, encoding="utf-8", errors="replace")
    return json.loads(next(line[7:] for line in proc.stdout.splitlines() if line.startswith("POINTS=")))


def vehicle_sheet(folder, blender):
    points = projected_points(blender, folder / "vehicle.blend")
    poses = {r["frame"]: r for r in json.loads((folder / "rig_check.json").read_text(encoding="utf8"))}
    canvas = Image.new("RGB", (1264, 854), PAPER)
    draw = ImageDraw.Draw(canvas)
    text(draw, (36,31), "相机如何转到车体右侧", 38, bold=True)
    text(draw, (36,94), "车辆俯视图     ← 车头      ↑ 车右", 22, MUTED)
    reticle(draw, (914,102), LENS, 8)
    text(draw, (935,91), "镜头光心", 19)
    reticle(draw, (1083,102), PIVOT, 8)
    text(draw, (1104,91), "云台转轴", 19)
    for panel, (frame, stem, title) in enumerate(((1,"stowed","停车收拢"), (40,"deployed","行驶展开"))):
        x = 36 + 609*panel
        draw.line((x,141,x+583,141), fill=RULE, width=1)
        text(draw, (x,164), title, 28, bold=True)
        with Image.open(folder / f"plan_{stem}.png") as src:
            w, h = src.size
            left, right = int(w*.20), int(w*.80)
            scale = 583/(right-left)
            panel_image = src.crop((left,0,right,h)).resize((583,round(h*scale)),Image.Resampling.LANCZOS)
            canvas.paste(panel_image, (x,216))
        for (u,v), color in zip(points[str(frame)], (LENS, PIVOT)):
            reticle(draw, (x+(u*w-left)*scale,216+v*h*scale), color, 11)
        height = round(poses[frame]["xyz_mm"][2])
        caption = f"镜头收在底盘轮廓内  ·  停车高度暂取 {height} mm" if panel == 0 else f"镜头伸至右侧  ·  光心离地 {height} mm，垂直拍向地面"
        text(draw, (x,787), caption, 21, MUTED)
    canvas.save(folder / "车体右侧相机.png", optimize=True)


def camera_sheet(folder):
    report = json.loads((folder / "mission_vision.json").read_text(encoding="utf8"))
    rows = report["observations"]
    scenes = [
        ("原料区抓取", "RAW", "pick_top", "指定颜色物块的顶面中心"),
        ("粗加工区放置", "PROCESS", "empty_rings", "目标编号的圆环中心"),
        ("暂存区第二层放置", "BUFFER", "support_top", "第一层同色物块的支撑面中心"),
    ]
    canvas = Image.new("RGB", (1336, 524), PAPER)
    draw = ImageDraw.Draw(canvas)
    text(draw, (32,28), "同一台右侧相机，三个工位", 36, bold=True)
    text(draw, (32,86), "取自 Blender 初赛示例的到站画面，镜头离地 400 mm。", 21, MUTED)
    for index, (title, station, feature, caption) in enumerate(scenes):
        row = next(r for r in rows if r["station"] == station and r["feature"] == feature
                   and r["status"] == "ACQUIRED" and r["offline_score"]["result"] == "ON_TARGET")
        path = folder / "mission_camera" / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Regenerate mission_vision.json for {path}")
        x = 32 + 430*index
        text(draw, (x,149), title, 25, bold=True)
        with Image.open(path) as source:
            image = source.convert("RGB")
        marks = ImageDraw.Draw(image)
        cx, cy = (d/2 for d in image.size)
        for stroke, width in ((INK,5), ("white",2)):
            marks.line((cx-13,cy,cx+13,cy), fill=stroke, width=width)
            marks.line((cx,cy-13,cx,cy+13), fill=stroke, width=width)
        reticle(marks, row["center_xy"], GREEN, 11)
        canvas.paste(image.resize((412,232), Image.Resampling.LANCZOS), (x,201))
        text(draw, (x,452), caption, 19, MUTED)
    text(draw, (894,91), "白十字  画面中心      绿圈  识别中心", 20, INK)
    canvas.save(folder / "工位相机.png", optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--blender", default=shutil.which("blender") or
                        r"C:\Program Files\Blender Foundation\Blender 5.3\blender.exe")
    args = parser.parse_args()
    vehicle_sheet(args.out, args.blender)
    camera_sheet(args.out)
