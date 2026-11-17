# K230 官方灰道 A 与 TCN4 独立板端测试包

两套目录分别绑定 Kaggle a06 的 CameraXYNet 单帧实验 fallback 与 TCN 4 帧实验候选。输入为 uint8 RGB NCHW。TCN 历史顺序从旧帧到当前帧。量化模型只用训练轨迹校准，分别通过 25 帧 nncase Simulator 回放。主机运行了因果窗口和连续掉帧后的全图恢复验证。K230 实测、设备端 AI2D 像素等价性和真实精度为 **NOT RUN**。输出坐标处于实验精度 HOLD，`position_valid` 保持 false。

板卡可用后，将两套目录各自复制到 `/sdcard`，从 CanMV REPL 运行。使用 `media.uvc.UVC.probe()` 确认 OV2735 的确切 ID。入口只访问选定的 UVC、AI2D、KPU 与 CPU 滤波，使用独立的结果文件。

~~~python
import sys
root = '/sdcard/k230_temporal_v1_gray_cnn1'
sys.path.insert(0, root)
from board_multiroute import run
usb = '<verified OV2735 identifier>'
run(root, 'cnn1_raw_01', usb, frames=120, mode='raw')
run(root, 'cnn1_filter_01', usb, frames=120, mode='filtered')
run(root, 'cnn1_3000_01', usb, frames=3000, mode='filtered')
~~~

完成 A 组后，清理解释器的实验会话。将 `root` 换为 `/sdcard/k230_temporal_v1_gray_tcn4`，用不同的报告 ID 重复相同测试。比较两套报告内 `camera_to_result_fps`、分阶段延迟 P50/P95/P99、空帧与超时计数、退出时资源释放、堆和系统内存。`decode` 包含滤波，`filter` 单独记录其中的 CPU 滤波开销。

首帧填满 TCN 历史窗口。滤波时间戳取采集返回时刻。掉帧会报告缺测，连续五次超时后清空旧历史，恢复时以新首帧填充窗口。滤波保留双启停区软先验、全局概率底和 LOST 后全图恢复。当前部署参数和 a06 实验一致。时差扩窗保留为单独的仿真参数。报告同时记录固件版本、设备查询的 nncase runtime 版本和模型使用的编译器版本。

UVC、AI2D 和 Tensor 入口沿用 CanMV API。TCN 每次从最新打包好的数组创建输入 Tensor。 [CanMV 官方 nncase 使用文档](https://www.kendryte.com/k230_canmv/en/main/example/ai/nncase_usage.html)
