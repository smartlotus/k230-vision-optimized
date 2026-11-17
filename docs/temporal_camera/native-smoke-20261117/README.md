# 滤波定位工程跑通检查

日期：2026-11-17。仅检查硬件调用与代码流程，不进行定位正确率、道路检测或整车测试。

## 当前结果

| 检查 | 证据与结论 |
|---|---|
| K230 通信 | COM3，LCKFB-LSPI-K230-1G-CanMV，固件 v1.8-36-gae2f20d；实际执行唯一标记 `TEMPORAL_PREFLIGHT_20261117_A` 的预检查并返回终端 |
| Runtime | 设备实读 nncase 2.11.0，与 R2 模型编译器 2.11.0 一致 |
| USB 枚举 | `UVC.probe()` 返回 `(True, 'openaicam#openaicam')`；列出 1280×720 / 1920×1080 MJPEG 30FPS 及三种 YUY2 模式。设备名本身不能鉴定传感器芯片 |
| 交付完整性 | [archive-check.json](archive-check.json)：R2 ZIP SHA256 和全部 20 个 ZIP 成员校验通过（19 个内容文件及内置 manifest）；5 个本地板端模块与交付包逐字节一致 |
| 本地逻辑 | 原有 20 项测试通过，包括因果历史窗口、滤波、缺测恢复、时间戳、输入契约与评估标注隔离；见 [HOST_VALIDATION.json](HOST_VALIDATION.json) |
| 本次原生采集 / 推理 / 滤波 | **NOT RUN，0 帧**。准备安装时发现外部脚本正在运行，安装器在任何 SD 写入前拒绝继续 |
| 报告落盘 / 原生资源清理 | **NOT RUN** |
| 定位精度 / 道路实际测试 | 本次不做，**NOT RUN**；原有精度 HOLD 和 `position_valid=false` 保留 |

安装前查询 `running=false`，执行只枚举与读取的预检查后返回 REPL。随后再次查询发现 `running=true`，非本次启动；主机 CanMV 日志也有相应时段外部预览活动。保留现场程序，没有发送 stop、上传模型或切换相机。稍后查询再次为 `running=false`，但独占条件尚待用户确认。OPS 认领只能协调当前 Git common directory，不能证明其他会话没有使用 COM3。

启动文件预检查 SHA：`/sdcard/main.py` 为 `c0b89b0b48864ff51e0e1d9a07082d306dbaf301aa4b6c95432ea6e04dbc640c`，`/sdcard/boot.py` 不存在。本次没有写入启动文件、生产参数、旧 SD 目录或其他线路源码。

## 方案与待执行入口

本线为真实 USB 图像 → RGB888 640×360 → CHW / AI2D 160×96 → CameraXY KModel → 2304 个位置 logits → CPU 概率网格时序滤波 → 候选位置 / 置信度 / 状态 → 独立 JSON 报告。CNN1 为单帧基线，TCN4 为同线的四帧历史对照；不涉及 RoadNet、TopFace 或车辆控制。

[native_smoke.py](../../../research/temporal_camera/native_smoke.py) 复用既有 CanMV MCP 客户端和分片拼合器。它校验 ZIP 与全部成员，262144-byte 分片上传并逐片读回，再重新打开目标文件核对完整 SHA。新目录与新报告名必须不存在，板忙或状态未知时拒绝写入。每次运行限 1–120 次尝试，不自动重试。

用户确认本次独占后，在 `C:/Users/28389/Desktop/k230-temporal-camera-v1` 执行：

```powershell
python -B -X utf8 research/temporal_camera/native_smoke.py install --archive C:/Users/28389/Desktop/k230-project/deliverables/K230_Official_Gray_Map_A_TCN4_Board_Bench_R2.zip --setting cnn1 --remote /sdcard/k230_temporal_v1_c1s1117 --out docs/temporal_camera/native-smoke-20261117/install-cnn1-original.json
python -B -X utf8 research/temporal_camera/native_smoke.py run --remote /sdcard/k230_temporal_v1_c1s1117 --run-id c1_original10_a --frames 10 --out docs/temporal_camera/native-smoke-20261117/c1-original10-intent.json
```

首个 install 调用在 idle 门处退出，未创建上述板端目录或安装收据。恢复时仍应核对当前板端状态、目录与报告是否存在；不得以本文件为当前设备空闲证明。运行结束后用 `fetch` 子命令读取报告和 entry 记录。真机错误保留原报告；仅在确认根因并修复后以新 ID 复验。准确率 HOLD 不因工程跑通而解除。
