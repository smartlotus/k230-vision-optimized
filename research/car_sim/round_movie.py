"""Make a short annotated route replay from the finished Blender round and camera frames."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def render_views(tmp):
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    s = bpy.context.scene
    cam = s.camera
    cam.location.y = .085
    cam.data.ortho_scale = 2.9
    s.render.resolution_x = s.render.resolution_y = 512
    s.render.resolution_percentage = 100
    s.cycles.samples = 2
    s.cycles.use_denoising = True
    s.render.use_persistent_data = True
    root = s.objects["K230 | Vehicle ground origin"]
    frames = json.loads((tmp / "frames.json").read_text())
    points = []
    for i, frame in enumerate(frames):
        s.frame_set(frame)
        p = world_to_camera_view(s, cam, root.matrix_world.translation)
        points.append([p.x, 1-p.y])
        s.render.filepath = str(tmp / f"map{i:03}.png")
        bpy.ops.render.render(write_still=True)
    (tmp / "points.json").write_text(json.dumps(points))
    cam.location = (-.63,-.47,.36)
    cam.rotation_euler = (Vector((-1.125,0,.06))-cam.location).to_track_quat("-Z","Y").to_euler()
    cam.data.ortho_scale = .56
    s.render.resolution_x, s.render.resolution_y = 640,360
    s.cycles.samples = 8
    s.render.filepath = str(tmp / "end_stacks.png")
    bpy.ops.render.render(write_still=True)


def create(folder, tmp, blender):
    from PIL import Image, ImageDraw
    from preview import INK, MUTED, PAPER, font

    data = json.loads((folder / "round.json").read_text(encoding="utf8"))
    records = data["timeline"]
    steps = data["plan"]["steps"]
    end = records[-1]["end"]
    frames = list(range(1,end,23)) + [end]
    (tmp / "frames.json").write_text(json.dumps(frames))
    subprocess.run([blender, "-b", str(folder / "competition_round.blend"), "--python-exit-code", "1",
                    "--python", str(Path(__file__).resolve()), "--", str(tmp)], check=True,
                   stdout=subprocess.DEVNULL)
    positions = json.loads((tmp / "points.json").read_text())
    station_names = {"RAW":"原料区", "PROCESS":"粗加工区", "BUFFER":"暂存区"}
    color_names = {1:"红色", 5:"黑色", 6:"浅蓝色"}
    fps = 6
    style = {size: font(size,bold) for size,bold in ((39,True),(19,False),(23,True),(30,True),(18,False))}
    target = tmp / "movie.mp4"
    command = ["ffmpeg","-hide_banner","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
               "-s","1184x672","-r",str(fps),"-i","pipe:0","-vf","fps=24","-an","-c:v","libx264",
               "-preset","veryfast","-crf","19","-pix_fmt","yuv420p","-movflags","+faststart",str(target)]
    encoder = subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for i,frame in enumerate(frames):
            index = max(j for j,s in enumerate(records) if s["start"]<=frame)
            rec = records[index]
            step = steps[index]
            canvas = Image.new("RGB",(1184,672),PAPER)
            draw = ImageDraw.Draw(canvas)
            def label(x,y,value,size,color=INK):
                draw.text((x,y),value,font=style[size],anchor="lt",fill=color)
            label(26,20,"智能搬运初赛",39)
            label(294,39,"Blender 参考轨迹 · 加速回放",19,MUTED)
            mx,my,side = 27,88,552
            with Image.open(tmp / f"map{i:03}.png") as image:
                canvas.paste(image.resize((side,side),Image.Resampling.LANCZOS),(mx,my))
            trail = [(round(mx+x*side),round(my+y*side)) for x,y in positions[max(0,i-10):i+1]]
            if len(trail)>1:
                draw.line(trail,fill="#58C9D0",width=3)
            u,v = trail[-1]
            draw.ellipse((u-4,v-4,u+4,v+4),fill="#58C9D0",outline=INK,width=2)
            rx = 612
            action = step["action"]
            if action=="QR":
                phase,title,image_path,caption = "出发","前往二维码板读取任务",folder/"qr.png","扫码工位画面"
            elif action=="RETURN":
                phase,title,image_path,caption = "返回","回到出发时的蓝色启停区",tmp/"end_stacks.png","暂存区终态 · 三组同色物块分两层放置"
            elif action=="MOVE":
                phase = "第一批" if step["batch"]==1 else "第二批"
                title = "前往"+station_names[step["dest_station"]]
                image_path,caption = folder/"deployed.png","行驶姿态 · 相机展开至车体右侧"
            else:
                phase = "第一批" if step["batch"]==1 else "第二批"
                station = station_names[step["source_station"] if action=="PICK" else step["dest_station"]]
                if action=="PICK":
                    verb = "取"+color_names[step["target_color"]]+"物块"
                else:
                    verb = f'{step["target_slot"]} 号位' + ("同色叠放" if step["layer"]==2 else "放料")
                title = station+" · "+verb
                image_path,caption = folder/"mission_camera"/f"{index:02}.png","右侧相机 · 本次动作的到站画面"
            label(rx,99,phase,23,MUTED)
            label(rx,146,title,30)
            with Image.open(image_path) as image:
                canvas.paste(image.resize((544,306),Image.Resampling.LANCZOS),(rx,210))
            label(rx,535,caption,19)
            label(rx,595,"任务码   "+data["plan"]["task"],19,MUTED)
            draw.rectangle((rx,631,1156,635),fill="#D8DEDB")
            draw.rectangle((rx,631,rx+round(544*frame/end),635),fill="#267480")
            repeats = fps if i==0 else (fps*2 if i==len(frames)-1 else 1)
            encoder.stdin.write(canvas.tobytes()*repeats)
    finally:
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f"ffmpeg failed ({code})")
    shutil.copyfile(target,folder/"整轮回放.mp4")


if __name__=="__main__":
    if "--" in sys.argv:
        render_views(Path(sys.argv[sys.argv.index("--")+1]))
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out",type=Path,default=Path(__file__).resolve().parent/"output")
        parser.add_argument("--blender",default=shutil.which("blender") or
                            r"C:\Program Files\Blender Foundation\Blender 5.3\blender.exe")
        args = parser.parse_args()
        with tempfile.TemporaryDirectory(prefix="k230-round-") as temp:
            create(args.out.resolve(),Path(temp),args.blender)
