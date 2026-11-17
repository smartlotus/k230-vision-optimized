# 录像与连续 RGB 离线回放

工具读取已交付 a06 ZIP 内的原始浮点 ONNX，以及板端 R2 包的相机锚点和参数。支持 CNN1、GRU4/8、TCN4/8，默认运行 A 和 TCN4。参考坐标、实际起点编号与地标标记只用于回放评分。滤波初始化始终使用两个蓝区等权先验。输出逐帧候选光心 XY、双区概率质量、局部与远场概率质量、缺帧、LOST、全图恢复事件与测量误差。真实坐标有效标记维持当前实验状态。

所需主机依赖为 Python、NumPy、Pillow、ONNX Runtime。视频拆帧额外使用 ffmpeg 和 ffprobe。回放通过读取既有模型 ZIP 运行，模型无需解压到磁盘。

## 从录像生成时间戳清单

~~~powershell
python prepare_recording.py "D:/clips/start1.mp4" --out "D:/clips/start1" --trajectory-id start1 --source ov2735-recording --start-zone 1
~~~

--start-zone 是可选的评分标记。视频逐帧使用原始呈现时间戳，减去第一帧时间，保留帧间隔变化。解码不旋转图像。--max-frames 可以截取片段开头用于短测。输入目录存在时拒绝覆盖。失败时保留已写文件以便检查。

录像只包含实际编码进文件的帧。PTS 间隔参与软先验衰减。缺测次数需要采集端记录，或在 manifest 中用 image_path=null 明确表示。工具由这些显式记录触发连续五次缺测的 LOST 与窗口清空。

## 读取已有 RGB 帧

manifest.jsonl 与其相对图片路径放在同一目录。每行至少包含 trajectory_id、从 0 开始的连续 frame_index、严格递增的 timestamp_s 和 image_path。缺测保留一行，令 image_path 为 null。一次清单可包含多条完整轨迹。

~~~json
{"trajectory_id":"start1","frame_index":0,"timestamp_s":0.0,"image_path":"frames/00000000.png","start_zone":1}
{"trajectory_id":"start1","frame_index":1,"timestamp_s":0.033,"image_path":"frames/00000001.png"}
{"trajectory_id":"start1","frame_index":2,"timestamp_s":0.1,"image_path":null}
~~~

可提供 image_sha256 校验实际输入，首次可见地标那一帧可标为 first_landmark=true。若一帧有直接测量的光心 XY，可同时添加下面两项。只标注实际测量帧即可，其余帧输出候选位置与状态。

~~~json
{
  "camera_xy_target":{"camera_x_mm":1050.0,"camera_y_mm":1000.0},
  "measurement":{
    "reference":"DIRECT_CAMERA_OPTICAL_CENTER_FIELD_MAP_XY",
    "uncertainty_mm":10.0,
    "method":"实际光心投影测量方法",
    "record_id":"实际测量记录编号"
  }
}
~~~

640×360 RGB 按通过历史字节验证的 AI2D 路径预处理。160×90 RGB 直接补边。其他 16:9 图像先在主机上缩放到 640×360，结果明确记录该主机缩放路径。后者仍需与实际 UVC 转换做对照。

## 运行所有路线

~~~powershell
python replay_camera.py --source ov2735-recording --manifest "D:/clips/start1/manifest.jsonl" --models "../K230_Official_Gray_Map_Pilot_A06_Models.zip" --board "../K230_Official_Gray_Map_A_TCN4_Board_Bench_R2.zip" --settings cnn1 gru4 gru8 tcn4 tcn8 --out "D:/clips/start1/replay.json"
~~~

连续五帧缺测触发 LOST，时序窗口随后清空。恢复第一张有效图重复填充初始窗口。每条轨迹单独重置历史，捕获时间间隔参与滤波。新的报告使用独立文件名。

回归已覆盖 48 次输入、38 张有效帧、10 次缺帧和 4 条轨迹，每条路线都观测到两次全图恢复。另用 6 帧可变帧率视频核对了 0.04、0.08、0.12、0.04、0.16 秒间隔。该验证的素材声明为 simulated。真实 OV2735 回放和 K230 为 **NOT RUN**。 [验证记录](map-correction/OFFLINE_REPLAY_WITNESS.json)
