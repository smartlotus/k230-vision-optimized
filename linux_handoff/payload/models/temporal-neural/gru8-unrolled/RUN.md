# K230 时序神经网络试验

当前为精度 HOLD。试验程序输出候选，已验收的位置字段保持空值。输入为最近 8 张按时间顺序采样的 RGB 帧。窗口节奏绑定 RUNTIME_PROFILE.json 的仿真训练时间戳。

解压到独立 SD 卡目录，确认设备空闲。从 CanMV 解释器运行以下代码。

```python
import sys
sys.path.insert(0, "/sdcard/temporal_neural_trial")
import neural_board_probe
neural_board_probe.run("/sdcard/temporal_neural_trial", "/sdcard/neural_bench_a01.json", "实际 OV2735 USB ID", frames=1200)
```

`frames` 限制采集尝试次数。结果中的 `frames` 是完成了神经网络推理的样本数。报告记录采集与推理、窗口打包、跳采和长间隔重置、内存及完整周期。同名 .frames.jsonl 文件逐帧记录采集时间、窗口内时间及候选，可与实测 XY 对齐。
