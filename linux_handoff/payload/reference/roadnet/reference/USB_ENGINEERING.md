# USB 摄像头到模型的工程链路检查

本入口对应 2026-11-17 用户明确限定的任务：只确认代码和硬件能完整调用，不进行正确率或真实道路效果测试。它使用既有 RoadNet r3 与 TopFaceNet 模型，不比较标签、不评价道路类别内容、不计算安全走廊，也不调用 CSI、二维码、UART 或运动控制。

运行前读取 OPS 状态、认领 COM3/K230-media，并确认设备空闲。模型和源码继续放在 `/sdcard/rn1117a`、`/sdcard/tfp1116`；录像、图片、KPU 原始输出、运行报告和大日志统一放到 `/data/roadnet-engineering/`。扩展入口把源码直接交给 CanMV 在 RAM 中执行，结果从终端完整回传电脑；需要板端持久化的报告也只写 `/data/`。

测试按下列顺序推进；每轮必须使用新 tag。

```powershell
python reference/native_host.py measure --mode uvc --duration 10 --tag UNIQUE-USB
python reference/native_host.py measure --mode roadnet --duration 30 --tag UNIQUE-ROAD
python reference/run_usb_engineering.py --tag UNIQUE-DISPLAY --seconds 15
python reference/run_usb_engineering.py --tag UNIQUE-BOTH --seconds 120 --topface
python reference/run_usb_engineering.py --tag UNIQUE-RESTART --seconds 30 --topface
```

前两个原版入口保留独立小型 SD 报告；后续 RAM 入口只在电脑保存证据。tag 最多32个字母、数字、下划线或连字符；原版入口最多20字符。报告、意图和执行源码位于 `evidence/board-20261117/usb-engineering-v1/`；旧失败不会被覆盖。测量若失败或清理未确认，保留原件、停止序列并诊断。

当前 `native_host.py measure` 的板端报告目录是 `/data/roadnet-engineering/reports/`；`tensor` 的临时脚本、输入和完整 KPU 输出也位于 `/data/roadnet-engineering/`。代码会在写入前确认目录存在，并拒绝把报告路径传回 `/sdcard/`。`/sdcard` 只承担模型加载和少量程序文件。

在 Windows 资源管理器中查看大文件：`此电脑 → CanMV → data → roadnet-engineering`。不要把模型目录当作录像、图片或日志目录；旧的 `/sdcard` 文件没有自动删除，避免覆盖既有证据。

实际调用链为 USB MJPEG 1280×720 → RGB888 640×360 → 独立 CHW RGB 图像 → AI2D 320×180 与上下6px填充 → KPU → 1×7×192×320 logits → 全部430080值有限检查 → argmax → 裁去填充后的180×320类别图结构检查。这里只确认输出形状和数值范围，不保存或评价道路掩膜内容。若启用 TopFaceNet，两个模型顺序读取同一拥有独立 backing 的帧。TopFace 解码仅检查返回类型，不统计识别正确率。

显示使用 packed RGB888 转 RGB565，保留模型源图像独立所有权。画面仅叠加 `ENGINEERING ONLY` 和已处理帧数，不叠道路预测或识别效果评价。显示通道与 IDE 输出均显式开启；MCP预览的新鲜帧必须在程序运行期间取样。结束后 `Display.deinit()`，旧缓存帧不算实时画面。

每轮核对：attempts/frames/model/postprocess/display 分母、异常、逐段时延、资源采样和 cleanup。每帧释放临时引用并 GC；结束依次关闭UVC、显示、MediaManager与模型。停止后再次成功建立相同链路，才能支持本次范围内的资源复用结论。短测不能替代30分钟长期稳定性。

FPS包含本入口的全部有限值检查、类别图结构检查、GC和显示开销；它是工程诊断配置的实测值。摄像头协商FPS、单独取流FPS、KPU耗时和全链路FPS分别报告。

主机端 `openaicam#openaicam` 是设备枚举名，不能独立证明传感器芯片型号。模型本体和既有接口由板端初始化重新计算SHA，执行源码与结果由主机意图中的tag及SHA绑定。

本轮归档可用 `python reference/verify_usb_engineering.py` 只读复核。检查实际分母、执行脚本文本SHA、结果、清理、预览和最终idle，不重新运行相机或模型。执行脚本SHA以实际发送的UTF-8/LF文本计算，Git/Windows文件换行的字节清单另行保存。

原始报告、执行脚本、终端及清单通过证据目录的局部 `.gitattributes` 禁用文本换行转换。提交后还要按 `FILE_MANIFEST.json` 逐项核对 Git blob 的字节数和SHA；仅验证Windows工作区不足以证明跨系统取回时保留了原始字节。不修改全局Git配置或历史证据的属性。

提交后运行 `python reference/verify_usb_engineering.py --git-blobs`，同时核对工作区与HEAD中的全部清单成员。
