# 六个仿真仓库核查

2026-11-14 使用官方 GitHub API 确认默认分支 HEAD，再读取固定 commit 的 README、许可证、依赖与入口。未安装这些仓库依赖，也未从它们复制实现。许可证 UNKNOWN 不等于允许复制。

| 仓库固定快照 | 许可证 | 依赖和多目标限制 | Windows/Kaggle 证据 |
|---|---|---|---|
| [altaf7740/blender-synthetic-data, 15fc9f4](https://github.com/altaf7740/blender-synthetic-data/tree/15fc9f4efa86c723d09862aa5dd4c418741573ca)（2026-09-30） | UNKNOWN；API为空且 tree 无许可文件 | Python3.13、bpy≥5、cascadio、OpenCV/Torch/Ultralytics；每类0–3实例，整物体颜色 mask，非顶面。[pyproject](https://github.com/altaf7740/blender-synthetic-data/blob/15fc9f4efa86c723d09862aa5dd4c418741573ca/pyproject.toml#L5) | 作者称 Windows/Linux/macOS 支持；本项目未测，Kaggle UNKNOWN |
| [DLR-RM/BlenderProc, 76bd8b8](https://github.com/DLR-RM/BlenderProc/tree/76bd8b82ee19489843578ff045298642197bf883)（2026-01-07） | [GPL-3.0](https://github.com/DLR-RM/BlenderProc/blob/76bd8b82ee19489843578ff045298642197bf883/LICENSE) | Blender独立Python、NumPy/Matplotlib/Pillow/h5py等；多物体/instance segmentation成熟，顶面仍需面级 pass | 官方安装器支持三平台；Colab 示例不等于 Kaggle 验证。本项目旧 Kaggle v8 在 quickstart backend 阶段失败 |
| [LogqnTyler/DataGen, 36f3a69](https://github.com/LogqnTyler/DataGen/tree/36f3a6998feb5b68d5f6cbe371fce32312546fa7)（2026-08-31） | UNKNOWN；API为空且 tree 无许可文件 | Python≥3.13、BlenderProc≥2.8；[README:80](https://github.com/LogqnTyler/DataGen/blob/36f3a6998feb5b68d5f6cbe371fce32312546fa7/README.md#L80) 明确每job单target，targets_per_scene未消费；并行job不等于同帧多目标 | Windows/Kaggle运行验证均 UNKNOWN |
| [Janga786/robot_vision, 1fc15fb](https://github.com/Janga786/robot_vision/tree/1fc15fb65e48e2ed723d88b6458779b86be17d16)（2026-11-10） | [MIT](https://github.com/Janga786/robot_vision/blob/1fc15fb65e48e2ed723d88b6458779b86be17d16/LICENSE) | Blender≥2.80；每帧随机选一个模型，顶点投影 bbox、背景合成，无可见顶面 mask | 有headless命令，无明确Windows/Kaggle验收；作者无真实照片验证，val复用train，不能借用精度 |
| [mandyxiaomeng/SynMfg, d191f76](https://github.com/mandyxiaomeng/SynMfg/tree/d191f76fb5afc9286060a6736fae12f392f6708c)（2026-02-25） | [GPL-3.0-or-later](https://github.com/mandyxiaomeng/SynMfg/blob/d191f76fb5afc9286060a6736fae12f392f6708c/LICENSE#L6)；API NOASSERTION，但许可正文明确 | README Blender3.4，环境文件另列bpy4/BlenderProc2.7.1/NumPy2；多实例和空场，object_index mask非顶面 | Windows/Linux安装示例；Kaggle链接只是数据集，无执行验证。[launcher](https://github.com/mandyxiaomeng/SynMfg/blob/d191f76fb5afc9286060a6736fae12f392f6708c/generation_main.py#L23) 崩溃自动重启，不能直接继承本项目禁止科学任务自动重跑的策略 |
| [denicprotopopov/blender-dataset-generation, 5e07f25](https://github.com/denicprotopopov/blender-dataset-generation/tree/5e07f25e9b6d28bb9e8c263a1962940e810c4db2)（2026-08-10） | UNKNOWN；API为空且 tree 无许可文件 | Blender4.x/Cycles内置模块；每帧1..N目标，整mesh投影bbox、6D pose/内参，无顶面mask | 作者称Windows内置编辑器曾崩溃，headless更稳定；5.x和Kaggle UNKNOWN |

六个快照都不能直接满足“原始融合CAD不变、Ø30顶面语义mask、物理圆心与conic center分离、完整0–5标签合同”。因此本项目继续使用已本地验证的独立原生 bpy 管线。以上兼容性中的作者声明、代码能力和本项目实测明确区分。
