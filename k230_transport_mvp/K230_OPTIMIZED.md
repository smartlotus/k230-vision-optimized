# K230 独立优化版

本轮仅新增文件，旧代码、旧入口、参数和用户数据全部保留。
工作区：`C:/Users/28389/Desktop/k230worker-board-optimize`；分支：`ops/k230-buffer`。
基线 `5f1622d5622d89820f2a7d0c3db596c8ecdfb6c2`，包含上一轮主机 SHA256 优化。

## 新文件与使用关系

- `k230/uart_adapter_optimized.py`：继承旧适配器的协议编码、接收、序号、心跳和硬件启用条件，仅替换发送队列与排空实现。
- `k230/main_optimized.py`：复用旧 `run_loop`，显式选用新适配器；摄像头构造失败时关闭已创建的 UART。
- `run_block_service_optimized.py`：单独启动入口，使用 `/sdcard/transport_mvp_optimized`，不修改旧安装位置。
- `tools/benchmark_uart_optimized.py`：可重复运行的纯主机内存对照工具。
- `tests/test_uart_optimized.py`：新版本的队列和兼容性测试。

未来若获准部署，应把本工作区的完整模块树放入独立目录
`/sdcard/transport_mvp_optimized`，从全新解释器显式运行新入口。
新模块依赖保留的旧模块，不能只复制三个新文件到空目录。
入口检测到已缓存的 `k230` 模块会拒绝启动，避免混用旧安装。
现有旧入口仍选择旧适配器；本轮未生成覆盖式板端包，也未安装为 main.py。

## 优化效果与代价

旧发送队列使用不可变 bytes，每次部分发送都复制剩余队列。
新版本首次入队才分配固定环形缓冲，容量沿用配置（最多 3072 字节）；
部分发送只移动游标。每次驱动写入仍传递最多 256 字节的 bytes，避免引入原生缓冲生命周期假设。
队列绕回处可能多一次驱动调用，每次 drain 仍限制在最多 16 次。
诊断属性 `tx_buffer` 按需生成 bytes 快照；正常发送和排空不读取它。

代价是启用并发送后保留固定容量直到 close；空闲时不比旧版省内存。
默认 UART 关闭，因此不会改善当前视觉预览 FPS。图像算法与识别门槛不变。
返回发送成功仍仅表示本地排队接受，不表示 STM32 已消费数据。

主机合成对照：预先排队 3072 字节，每次驱动只接受 1 字节；
旧版排空阶段 Python 峰值分配 6528 字节，新版 633 字节。
这是 tracemalloc 临时分配，排除了初始化队列存储，不是总 RSS、板端实测内存或速度。
证据见 `evidence/k230-buffer/benchmark.json`。

## 验证边界

原有测试与新增测试一起运行，命令、结果和源指纹见 `evidence/k230-buffer/tests.json` 及 `.log`。
测试涵盖绕回、3000 步固定种子随机分段、完整/部分/零写入、错误、背压、协议字节对照、
STOP 接收、心跳、发送预算、资源关闭和主机入口拒绝。
旧文件逐项核验见 `evidence/k230-buffer/preservation.json`。
首次测试中宿主解释器替身缺少 cache_tag，已修复测试夹具；不是板端试验失败。

CanMV 本地缓存 `machine.pyi`（revision f6ca68a0ed02f8d29d27bcf9681581e9f71209bb）
确认 UART.write / any / read / deinit 接口；没有连接板卡。
MicroPython 实跑、物理 UART、STM32 消费和识别精度均未验证。
