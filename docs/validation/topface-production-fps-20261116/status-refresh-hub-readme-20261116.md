# K230 项目统一入口

此目录是 K230 项目的统一访问入口。它使用 Windows 目录链接聚合现有仓库、Git worktree 和快照，没有搬动原文件，也没有修改 Git worktree 关联。

- `repos/`：主仓库和当前优化仓库
- `worktrees/`：按 OPS 分支归类的 linked worktree
- `archives/`：GitHub 导出和统一快照
- `PROJECT_MAP.json`：原路径、角色和整理边界

当前开发主入口：`repos/k230-vision-optimized`
当前原始项目入口：`repos/k230worker`

当前TopFace实现与交接入口：`worktrees/ops-topface-production`。

- [最新项目状态](worktrees/ops-topface-production/docs/STATUS.md)
- [最新交接文档](worktrees/ops-topface-production/docs/HANDOFF.md)

## 当前板端验收与经验入口（2026-11-16）

- [经验与故障总账](worktrees/ops-topface-production/docs/LESSONS_LEARNED.md)
- [本轮显示修复与板端证据](worktrees/ops-camera-diagnosis/docs/validation/camera-diagnosis-20261116/)

历史显示修复对应分支 ops/camera-diagnosis-20261116；当前TopFace接入分支见下方，原训练交付与其他工作区保留。

- [正式接入、FPS与USB](worktrees/ops-topface-production/docs/validation/topface-production-fps-20261116/README.md)

当前接入分支：ops/topface-production-fps-20261116，模型与源码已部署至/sdcard/tfp1116。
