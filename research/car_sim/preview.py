"""Regenerate the annotated views from build_scene.py and mission_vision.py outputs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont


PAPER = "#FAF9F5"
INK = "#192D36"
MUTED = "#4B6067"
LENS = "#FFB648"
PIVOT = "#40DBDA"
RULE = "#B8C4C1"
GREEN = "#78F2A4"
WHEEL = "#11615F"


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


def vehicle_views(blender, scene, detail_path):
    expression = """
import bpy, json, math, sys
from bpy_extras.object_utils import world_to_camera_view
s = bpy.context.scene
c = s.objects['Plan / front left, right up']
def project(cam, pos):
    p = world_to_camera_view(s, cam, pos)
    return [p.x, 1-p.y]
rows = {}
for frame in (1, 40):
    s.frame_set(frame)
    names = {'lens':'ROAD CAMERA | optical centre', 'pivot':'AZIMUTH | turntable',
             'front':'Mecanum RF', 'rear':'Mecanum RB'}
    pts = {k:s.objects[name].matrix_world.translation for k,name in names.items()}
    mid = (pts['front'] + pts['rear'])/2
    pts['mid'] = mid
    rows[str(frame)] = {k:project(c, pos) for k,pos in pts.items()}
# Orthographic detail looking forward from behind the vehicle. +X right, +Z up.
opt = s.objects[names['lens']].matrix_world.translation
holder = s.objects['Camera holder | vertical'].matrix_world.translation
cam = c.copy()
cam.data = c.data.copy()
s.collection.objects.link(cam)
s.camera = cam
cam.data.ortho_scale = .165
cam.location = (opt.x-.038, opt.y-.80, holder.z)
cam.rotation_euler = (math.pi/2, 0, 0)
s.render.resolution_x, s.render.resolution_y = 660, 240
bpy.context.view_layer.update()
rows['side'] = {'lens':project(cam,opt),'holder':project(cam,holder)}
s.render.filepath = sys.argv[sys.argv.index('--') + 1]
bpy.ops.render.render(write_still=True)
print('POINTS=' + json.dumps(rows))
"""
    command = [str(blender), "-b", str(scene), "--python-exit-code", "1", "--python-expr", expression,
               "--", str(detail_path)]
    proc = subprocess.run(command, capture_output=True, check=True, encoding="utf-8", errors="replace")
    return json.loads(next(line[7:] for line in proc.stdout.splitlines() if line.startswith("POINTS=")))


def wheel_guides(draw, pts, top, bottom):
    front, rear, mid = (pts[name] for name in ("front","rear","mid"))
    # A perpendicular to the actual projected hub-centre line, in this orthographic view.
    dx, dy = rear[0]-front[0], rear[1]-front[1]
    length = math.hypot(dx,dy)
    nx, ny = -dy/length, dx/length
    for step in range(math.ceil((top-mid[1])/ny), math.floor((bottom-mid[1])/ny), 14):
        a = (mid[0]+step*nx, mid[1]+step*ny)
        b = (a[0]+7*nx, a[1]+7*ny)
        draw.line((a,b), fill="white", width=3)
        draw.line((a,b), fill=WHEEL, width=1)
    draw.line((front,rear), fill="white", width=4)
    draw.line((front,rear), fill=WHEEL, width=2)
    for key, label in (("front","右前轮"),("rear","右后轮")):
        x,y = pts[key]
        draw.ellipse((x-4,y-4,x+4,y+4),fill=WHEEL,outline="white",width=2)
        draw.text((x,y-22),label,font=font(16),fill="white",anchor="mb",stroke_width=2,stroke_fill=INK)
    x,y = mid
    draw.polygon([(x,y-5),(x+5,y),(x,y+5),(x-5,y)], fill=WHEEL, outline="white",width=2)


def vehicle_sheet(folder, blender):
    with tempfile.TemporaryDirectory(prefix="k230-mount-preview-") as tmp:
        detail = Path(tmp)/"side.png"
        points = vehicle_views(blender, folder / "vehicle.blend", detail)
        with Image.open(detail) as image:
            side = image.convert("RGB")
    poses = {r["frame"]: r for r in json.loads((folder / "rig_check.json").read_text(encoding="utf8"))}
    canvas = Image.new("RGB", (1264, 1118), PAPER)
    draw = ImageDraw.Draw(canvas)
    text(draw, (36,31), "相机展开至车右，对准前后轮中点", 38, bold=True)
    text(draw, (36,94), "俯视     ← 车头      ↑ 车右", 22, MUTED)
    text(draw, (405,93), "实线连接右轮轮心，虚线穿过中点", 19, WHEEL)
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
        pts = {name:(x+(u*w-left)*scale,216+v*h*scale) for name,(u,v) in points[str(frame)].items()}
        wheel_guides(draw, pts, 240, 216+round(h*scale)-18)
        reticle(draw, pts["pivot"], PIVOT, 9)
        reticle(draw, pts["lens"], LENS, 11)
        height = round(poses[frame]["xyz_mm"][2])
        caption = f"镜头收在底盘轮廓内，停车高度暂取 {height} mm" if panel == 0 else f"镜头在右侧车外，光心离地 {height} mm，垂直向下"
        text(draw, (x,787), caption, 21, MUTED)
    draw.line((36,839,1228,839),fill=RULE,width=1)
    bx,by = 36,868
    bw,bh = 583,212
    canvas.paste(side.resize((bw,bh),Image.Resampling.LANCZOS),(bx,by))
    for name,color,size in (("holder","white",8),("lens",LENS,8)):
        u,v = points["side"][name]
        reticle(draw,(bx+u*bw,by+v*bh),color,size)
    text(draw, (645,883), "展开后，从车尾看支架", 26, bold=True)
    text(draw, (645,940), "白圈标出竖直支架，橙圈标出下方镜头光心。", 21, MUTED)
    delta = poses[40]["lens_minus_midpoint_xy_mm"][1]
    overhang = poses[40]["right_edge_overhang_mm"]
    text(draw, (645,990), f"模型中，光心与两轮中点的纵向偏差为 {abs(delta):.0f} mm。", 21, MUTED)
    text(draw, (645,1032), f"镜头超出底盘右边缘约 {overhang:.0f} mm，外伸量为建模暂估。", 21, MUTED)
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
