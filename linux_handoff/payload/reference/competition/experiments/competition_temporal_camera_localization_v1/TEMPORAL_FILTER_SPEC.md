# Temporal Camera Filter 规范

## 输入

- `visual_likelihood[H,W]`：CameraXYNet 的地图位置似然。
- `timestamp_s`：monotonic 时间戳。
- `map_valid_mask`：camera optical center 的可配置有效区域。
- 上一帧 belief。
- `v_camera_max_mm_s` 和 `process_margin_mm`。
- 可选真实运动反馈。

## 预测和更新

`r_mm = v_camera_max_mm_s * dt_s + process_margin_mm`

将上一 belief 扩散到 r_mm 内可达网格，形成 motion prior。再执行：
`posterior ∝ motion_prior * visual_likelihood * map_valid_mask`
归一化后返回 top-k、confidence、entropy、state。

## 状态机

- UNINITIALIZED：没有历史可信位置，使用已知起点区域先验或全局视觉重定位。
- TRACKING：视觉和运动先验一致，正常输出。
- DEGRADED：视觉多峰、低置信度、运动冲突；保留多候选。
- LOST：连续失败；停止强行沿用旧位置，注入全局恢复概率/重新初始化。

## 安全规则

1.  移动门限只排除明确不可能的位置。
2.  `v_camera_max` 未测量前必须保守。
3.  未知 camera-to-robot lever arm 会影响原地旋转时 camera XY
    的运动范围。
4.  dt 过大/丢帧时扩大可达范围，必要时 LOST。
5.  保留 global recovery，防止错误 prior 永久锁死。
6.  map valid mask 是 camera path envelope，不是 robot centerline。
7.  未校准的模型 score 不可直接当概率。

## 第一版为什么不训练时序网络

当前需求只需要历史位置和最大移动范围约束。确定性滤波可直接实现，CPU
运行、不增加 KModel、易调试。连续轨迹用于测试滤波，不等于训练 GRU。


## 7. 网格坐标解码
地图原点为 (-1200,-1200) mm 时，网格 (row,col) 的中心坐标为 `x=-1200+(col+0.5)*cell_mm`、`y=-1200+(row+0.5)*cell_mm`。不得把 0..2400 的栅格索引误当作有符号地图坐标。top cell posterior mass 是离散后验质量，不自动等于校准后的真实置信度；阈值必须通过验证集校准。
