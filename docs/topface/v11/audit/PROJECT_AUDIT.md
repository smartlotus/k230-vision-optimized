# V1.1 项目审计（G0）

当前实验根为 `C:\Users\28389\Desktop\k230-vision-optimized`；原始工程 `C:\Users\28389\Desktop\k230worker-live` 保持只读。两者是独立 Git 历史，非同一个 worktree。当前 hub README:10 指定 optimized 为开发入口，因此继续在其 `research/topface` 与 `docs/topface/v11` 隔离实验；不改生产源码。精确 HEAD、分支、remote、工作区与10条历史见 `REPOSITORY_IDENTITIES.json`。

optimized 的 `docs/UPSTREAM_IMPORT.json:2` 记录从旧仓库 `ops/k230-fps` 的 `0c3b8499e2877bc05df786921e199ab0533685d2` 导入，实际清单 332 项；live 当前 HEAD 为 `c4fcbf2675eb45d6eb74241fe28f1bb12c072039`。旧 README 的311项统计不能替代实际清单。

## 相机与输入链路

- `k230_transport_mvp/k230/camera_roles.py:1-5`：COLOR/TOP/RING→CSI2，QR/LANE→CSI1，CSI0隔离。
- `k230/camera_manager.py:65-76` 初始化 Sensor、默认 RGB565；`:106-119` 有界 snapshot、禁止自动重启和实际尺寸校验；`:126-145` 清理 sensor/display/media。
- 通用 `k230/config.py:13-14` 仍是640×480；审核真图 metadata 实际为640×360。不能修改旧配置来消除差异。
- `tools/prepare_frozen_model_fps.py:83-88` 从真实 RGB565 经主机转换后 transpose 生成691200字节RGB888P冻结文件。因此冻结文件 planar布局已验证，实时CSI2 RGB888P channel/stride/DMA寿命仍UNKNOWN。
- 历史固件传感器枚举为GC2093 CSI1/2，native1920×1080@30/60、1280×960/720@90；初始化帧率不是完整处理FPS。

## AI2D / KPU

`tools/kpu_pipeline_probe.py:137` 显式绑定存活的输入 backing；`:179-185` 配置NCHW、pad49/49、114、tf_bilinear/half_pixel；`:194-215` readinto691200→AI2D→KPU→输出拷贝；`:245-257` 清理对象与memory pool。69.29FPS历史证据是冻结文件循环且 camera/display=false，不能充当新模型端到端FPS。

历史runtime原始收据为nncase2.11.0，编译器及plugin也为2.11.0。具体证据见 CURRENT_MODEL_CONTRACT.json。新模型必须重新比较PyTorch/ORT/Simulator/KPU，不能把历史可运行性当识别通过。

## 数据与保护

r4 SHA256 `6d21a2e9d0a251c1161ee3b63345909d64ad1f8cda6d3f9be0e0c0454843e198` 在两仓一致；重新核验 36 个capture文件均匹配。12图/33顶面，7批准图/17框，5困难图；实际uncertain黑框为5个（含四目标困难图），历史文字“4个”不完整。3个批准scene组无跨split。旧test已在2026-11-13被评估，不能称全新盲测。

颜色映射以 `tools/material_annotation_schema.py:3`、`data/spec/human.md:86`、`data/spec/ai01.md:75` 核对：1红、2黄、3蓝、4绿、5黑、6浅蓝。无真实椭圆真值。

UART、运动、生产Save保持关闭；CSI0隔离；CSI1不改；main.py/boot.py、r4、历史模型及用户草稿不覆盖。本轮无新的板端调用。
