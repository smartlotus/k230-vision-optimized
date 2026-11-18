# 模型与 K230 运行接口

## CameraXYNet 候选输出

- `location_logits`: \[1,1,24,24\] 或经测试确定的网格。
- `xy_refinement`: \[1,2\] 或 cell residual。
- `visual_quality`: \[1,1\] 可选。

训练只监督 camera X/Y。Yaw 不进入 loss。

## Filter 接口

``` python
result = filter.update(
    visual_likelihood=location_probability_grid,
    timestamp_s=monotonic_time,
    v_camera_max_mm_s=configured_upper_bound,
    process_margin_mm=configured_margin
)
```

输出包含
camera_x_mm、camera_y_mm、confidence、state、top_k、entropy、relocalized。

## K230

Temporal Filter 在 CPU 上运行；不需要 KPU。复用 buffer，不要每帧
load/unload 模型。分开测模型延迟和 filter 延迟，并测完整 USB capture +
AI2D + KPU + postprocess + filter。 PyTorch→ONNX Runtime→nncase
Simulator→KPU 必须对齐 heatmap、XY、坐标归一化和 top-k decode。
