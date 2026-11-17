# K230 Vision Optimized

本仓库维护 K230 视觉算法、资源优化、主机验证与板端接入。各任务分支保存相应的原生测试与离线评估。分支和证据入口见 [项目地图](.ops/PROJECT.md)。

来源提交、311份原样导入文件及SHA256见 [docs/UPSTREAM_IMPORT.json](docs/UPSTREAM_IMPORT.json)。
本项目使用独立Git历史与OPS项目ID，原仓库保留。

## 入口

| 版本 | 入口或模块 | 用途 |
|---|---|---|
| 实时算法优化 | `k230_transport_mvp/run_live_workbench_optimized.py` | 每帧完整识别，默认每3帧生成一次预览 |
| 跟踪与通信优化 | `k230_transport_mvp/run_vision_service_optimized.py` | 单次扫描匹配物块，组合UART环形缓冲 |
| 仅通信优化 | `k230_transport_mvp/run_block_service_optimized.py` | 保留原识别实现，选择新UART队列 |
| 主机文件校验 | `k230_transport_mvp/tools/file_integrity.py` | 流式SHA256，降低大文件校验临时内存 |
| 原始实现 | 同目录下不带`_optimized`的模块和入口 | 对照和兼容依赖 |

新增板端入口面向独立目录`/sdcard/transport_mvp_optimized`，需要完整依赖树和经核验的配置。
首次导入未迁移操作者的本机草稿、SD配置或标注工具环境。

## 电脑端验证

Python 3.11环境：

```powershell
python -m pip install -r requirements-host.txt
python scripts/verify_host.py
```

完整模型工具测试另需 [requirements-research.txt](requirements-research.txt) 中的可选依赖。
缺少可选ONNX/PyTorch环境或C编译器时，报告会保留实际跳过数量。
首轮导入验证使用现有隔离环境，但从本仓库运行全部测试；结果见`docs/validation/`。

基准：

```powershell
cd k230_transport_mvp
python -m tools.benchmark_live_optimized --output evidence/local/live-benchmark.json
python -m tools.benchmark_block_optimized --output evidence/local/block-benchmark.json
python -m tools.benchmark_uart_optimized --output evidence/local/uart-benchmark.json
```

主机基准不代表K230帧率、识别准确率或STM32消费成功。
历史优化说明保留在`k230_transport_mvp/*OPTIMIZ*.md`，其中旧工作区路径属于来源快照。

## OPS交接

接手先读 [AGENTS.md](AGENTS.md)、[.ops/PROJECT.md](.ops/PROJECT.md) 和 [.ops/PROTOCOL.md](.ops/PROTOCOL.md)，然后：

```powershell
python .ops/tools/ops.py status
python .ops/tools/ops.py verify --task <任务ID>
```

从 status 查询任务 ID 和对应工作树。在该工作树核验交接并认领，模块完成后保存 checkpoint 和释放认领。
自动备份使用当前私有远端。任务完成、推送成功和主线集成分别记录。
使用、来源与数据边界见 [docs/PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md)。
