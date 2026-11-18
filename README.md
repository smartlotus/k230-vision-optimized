# K230 Vision Optimized

**Linux 迁移线**：立创庐山派 K230 1G 使用官方 Linux SDK 的交接包、模型契约和推进路线见 [Linux 交接入口](docs/linux-handoff/README.md) 与 [执行提示词](docs/linux-handoff/EXECUTION_PROMPT.md)。

接手先读 [最新项目状态](docs/STATUS.md) 和 [便携交接文档](docs/HANDOFF.md)。它们记录本次现场时间、当前USB运行、已完成证据和下一步；历史任务README中的“当前”只对其原快照时间有效。

项目经验与故障预防统一记录在 [经验总账](docs/LESSONS_LEARNED.md)。涉及板端、传输或部署前先读相关条目；确认新问题或验证修复后，随证据和 OPS checkpoint 更新。

K230视觉算法与资源优化的独立项目。新旧实现并存，可在电脑端比较结果，后续再做板端验收。
**约25FPS仍是待实机验证目标。2026-11-16 用户认可模型的人工识别效果后，已接入原TOP接口并完成双相机1000/1000帧、10.82FPS测试；USB模型600/600帧为8.42FPS。真实定量精度、完整数值配对与长期稳定性分别验收。**

当前显式板端入口为 `k230_transport_mvp/run_topface_service.py`，部署目录 `/sdcard/tfp1116`；资源、源码校验及USB结果见 [本轮记录](docs/validation/topface-production-fps-20261116/README.md)。不替换main.py/boot.py，也不写生产profile。

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
python .ops/tools/ops.py verify --task import-snapshot
```

任务分支`ops/任务ID`；写入前认领，模块完成后checkpoint并释放。
按项目授权范围同步到已核对的远端。任务完成、推送成功和主线集成分别记录。
使用、来源与数据边界见 [docs/PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md)。
