# K230 时序定位 V1 离线模型包

- `gray_frozen_fallback/` 含 A 路线的权重、用于恢复 PyTorch 模型的地图参考权重、浮点和 uint8 ONNX，以及同一张正式灰道修正版地图上的验证记录。沿用以前训练的 RGB 编码器，修正参考地图后优化更新为 0。
- `board/` 含对应的 int16 激活、uint8 权重 KModel、相机位置滤波参数和独立测试入口。滤波输出相机光心候选坐标。真实置信度校准前 `position_valid` 为 false，`camera_x_mm` 与 `camera_y_mm` 为空。
- `earlier_kaggle_a05/` 含旧版地图 Kaggle 私有笔记本第 1 版的同轨迹比较、CNN 和两种窗口各自的 GRU/TCN 权重、ONNX、历史与指标。一律按旧地图 Pilot 理解。修正版地图训练使用独立的 a06 记录。
- `SHA256_MANIFEST.json` 列出压缩包内其余每个文件的长度和 SHA256。外部收据记录整个 ZIP 的长度和 SHA256。

## 在 K230 上运行独立测试

真机测试状态为 **NOT RUN**。当前包的 KModel 已在 nncase 2.11.0 Simulator 上回放完整的修正版验证集。精度处于 HOLD。数据和真实像素预处理的等价性仍需实测。

板卡可用后，把 `board/` 内容放入独立的 `/sdcard/k230_temporal_v1_gray_a01`。在空闲 CanMV 设备上确认 OV2735 设备 ID 后，从 REPL 执行下面命令。每次使用新的报告 ID。入口只启动 UVC 采集、AI2D、KPU 和 CPU 后处理。

```python
import sys
root = '/sdcard/k230_temporal_v1_gray_a01'
sys.path.insert(0, root)
from board_temporal import run
usb = '<OV2735 ID from UVC.probe()>'
run(root, 'raw_smoke_01', usb, frames=120, mode='raw')
run(root, 'filtered_smoke_01', usb, frames=120, mode='filtered')
run(root, 'filtered_endurance_01', usb, frames=3000, mode='filtered')
```

每次运行保存原始报告和 `.filter.json`。记录从摄像头到结果的 FPS，各阶段延迟，堆、系统内存和 MMZ 内存，丢帧、无结果数和资源释放异常。UVC 超时由采集层计数。滤波器 LOST 与全图恢复还需在离线回放或后续真机故障注入中独立验证。

报告保留实验候选坐标；实际比赛用的位置有效性需要真实光心坐标和标定阈值。软先验覆盖两个启停区，时间间隔降低旧位置权重，未知实车速度时启用全局概率底。LOST 后首个可用观测全图重定位。
