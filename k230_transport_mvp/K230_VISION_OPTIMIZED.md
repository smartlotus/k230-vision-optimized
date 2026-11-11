# 视觉跟踪独立优化版

本轮新增文件，保留上一轮 UART 优化与全部旧代码。
工作区 `C:/Users/28389/Desktop/k230worker-board-optimize`，分支 `ops/k230-vision`。
基线 `cadb12fe8f652e0897539d18511fdc12b8fc0a67`。

## 入口与代码

- `vision/block_detector_optimized.py`：沿用原生 find_blobs、颜色/形状/裁边/重叠过滤，优化跟踪匹配。
- `k230/vision_adapter_optimized.py`：为原适配器的 block_parameters 路径选择新跟踪器，复用原观测、标定与错误处理。
- `k230/main_vision_optimized.py`：组合新视觉适配器与上一轮 UART 环形缓冲，复用原 run_loop。
- `run_vision_service_optimized.py`：新增的完整组合入口；旧 run_block_service.py 和上一轮 run_block_service_optimized.py 保留。

本入口面向 block_profile 观测服务；没有替换现有 live 调参面板、材料多目标生产者或 KPU 推理路径。
QR/RING/LANE 和非 block_parameters 的 TOP 路径继续使用旧实现。
未来获准部署时，完整依赖树放在 `/sdcard/transport_mvp_optimized`，从新解释器显式运行新入口。
不能只复制新入口到空目录；不能把旧版模块缓存与新目录混用。旧 SD 目录和 main.py 不需覆盖。
本轮没有连接、上传或启动板卡，默认 UART 启用条件保持原值。

## 修改与等价性

匹配阶段从“收集全部匹配并排序”改为“单次扫描保留最近两个距离”。
只需最近两个距离即可保持原差值大于16的歧义规则；相同距离仍按原遍历顺序处理。
二维距离改为直接算术，距离平方和稳定窗口配置每次 update 读取，不跨调用缓存可变参数。
历史窗口保留原表达式，不修改旧轨迹对象；目标丢失、重帧、倒序、超时及不合格候选语义不变。

10 个固定随机种子共3000帧差分覆盖候选数量0至12、多个颜色、无效候选、时钟/帧序变化和运行时窗口变化。
另有等距、差值16边界、位移边界、旧检测与服务回归，以及新适配器实际选用新类的检查。
全套 **461项主机测试通过，无跳过**。原有1106个跟踪文件逐项比对基线，改动数0。
证据：`evidence/k230-vision/tests.json`、`.log`、`preservation.json`。

## 本机基准与限制

每组1500次 update，五次交替顺序测量，以下为中位数；不计 native find_blobs、图像采集或显示。

| 候选数 | 旧版耗时 | 新版耗时 | 耗时减少 |
|---|---:|---:|---:|
| 3 | 17.114 ms | 8.873 ms | 48.2% |
| 6 | 34.039 ms | 16.954 ms | 50.2% |
| 12 | 75.683 ms | 38.623 ms | 49.0% |

速度收益来自减少 Python 排序和生成器开销，不代表 K230 整帧 FPS 提高同等比例。
内存没有取得净下降：12候选的 tracemalloc 峰值从15544增至16192字节（约+4.2%），因此只声明主机跟踪耗时收益。
首版历史窗口追加导致的额外列表容量已回退，最终保留旧窗口表达式。
可复现命令：`python -B -m tools.benchmark_block_optimized --output evidence/k230-vision/benchmark.json`。
硬件帧率、MicroPython兼容性、识别精度、串口消费和整车效果均待独立实测。
