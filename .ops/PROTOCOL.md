# OPS 开发协议 v2

本文件随项目保存，换模型、换工具、换电脑后都能读取。它是协作约定，不是操作系统访问控制。Python 工具能拒绝违规登记和提交；直接编辑文件仍靠 Agent 遵守协议和 worktree 隔离。

## 文件与权威范围

| 位置 | 用途 / 写入职责 |
|---|---|
| `AGENTS.md` | 项目开工入口；保留已有规则，协调者维护 |
| `.ops/PROJECT.md` | 目标、范围、目录地图、关键接口、命令、约束、阶段；协调者维护 |
| `.ops/config.json` | 项目标识、协调者、主线、同步目标与授权记录；协调者维护 |
| `.ops/tasks/ID.json` | 每任务一个持久快照；通过工具由该任务认领会话更新 |
| `.ops/handoffs/ID/时间-随机值.{md,json}` | 不覆盖的交接记录；任务 Agent 写入 |
| `.ops/PROTOCOL.md`、`.ops/tools/ops.py` | 可携带的协议和工具；串行维护 |
| Git common dir 的 `ops-runtime/项目ID/` | 同机 linked worktrees 共用的事务任务登记、认领、同步回执；不提交、不迁移为远端锁 |
| `.ops/local/` | 临时报告、草稿；忽略，不应是交接唯一证据位置 |

**项目真相以代码、Git 与实际证据为准。** 同机实时任务登记以共享 SQLite 为当前协调来源；Git 中的 JSON 是持久快照，用于远端恢复。`status` 同时显示快照矛盾和 `sync_current`，不能把旧回执当成当前备份。`PROJECT.md` 记录稳定背景，不再手写第二份任务状态列表。

在不同工作分支上，快照可能不同。不要看到自己分支上的旧快照就覆盖另一个 worktree 的当前记录。首次在新的 clone 建运行库时，导入本分支的快照；旧 `active` 转为需协调者确认的 `paused`，不继承远端的活跃锁。其他分支未合并的任务不会凭空出现：接手者必须取得对应任务分支和交接记录。

## 开始项目

1. 先查项目根、现有协议、Git 状态、远端、活跃 Agent/进程。已有编排系统时沿用其权威任务状态，仅补交接或备份缺口，不重复初始化。只读工作不写 `.ops`。
2. 新项目运行技能中的 `ops.py --root 项目根 init`。已有仓库必须指定真正的 worktree 根；不能把上层杂物目录当成一个新项目。脚本保留旧文件，追加带标记入口；新项目会初始化 Git，但不会提交全部文件。
3. 填好 `.ops/PROJECT.md`。初始化提示里的“待填写”不是已完成的上下文；不填就无法做到可靠无聊天交接。重要架构决策放既有 ADR 或在 PROJECT 中链接实际文件，交接只记录本次决定及理由。
4. 检查忽略规则、文件、历史和授权后，精确提交基础协议及属于本项目的初始代码。不要 `git add -A` 吞入其他人的改动。第一次并行之前，必须有可用 baseline commit，且新 worktree 能读到 `.ops/config.json` 与工具。
5. 协调者登记任务。小任务可单个串行任务，无需虚构子任务、审核者或后台进程。

```text
python .ops/tools/ops.py task-add --agent coordinator --task login --title "登录模块" --paths src/auth tests/auth --lane parallel --acceptance "登录成功/失败路径通过现有验收"
python .ops/tools/ops.py task-add --agent coordinator --task navigation --title "导航模块" --paths src/navigation tests/navigation --lane parallel --acceptance "导航与键盘访问验证通过"
```

先把这两份任务计划精确提交到 baseline。再用宿主的 managed worktree 工具（可用时优先）或 Git 创建各自工作目录。例如 `git worktree add -b ops/login ../project-login HEAD`。已有任务分支就复用，不 reset、不覆盖现有 worktree。任务 `--branch` 可适配项目既有命名规则。

## 认领与工作

在对应工作目录执行：

```text
python .ops/tools/ops.py status
python .ops/tools/ops.py claim --task login --agent codex --session session-20261009-a
```

会话 ID 必须区分重新启动的会话。任务 ID 和 Agent 会话标识不是安全身份认证，目的是防误操作。认领事务检查依赖已集成、路径/资源不重叠、串行互斥、工作目录和分支隔离。路径支持字面相对文件/目录，不支持 glob；父目录与子目录、Windows 大小写差异都视为冲突。新增/删除/重命名都属于写入，重命名两端均要在认领范围。

协调者登记公共接口、依赖清单/锁文件、迁移或集成任务时选 `serial`。耗用同一设备/端口/测试库的任务填相同 `--resources`，并行区也因此互斥。一个串行任务与所有活跃写任务互斥；只读工作不占写认领。不要为了避免等待而把共享依赖文件复制成第二份。

任务范围变更：先交接并释放旧任务，由协调者创建新的明确范围任务；不要工作到一半静默扩大权限。工作前发现其他人的改动时保留并确认归属。不要无条件 pull、重置、清理、删除锁或杀他人进程。

## Checkpoint 与交接

报告样例，先保存在 `.ops/local/login-report.json`。证据如果需要跨机器接续，应放任务拥有范围内、随代码提交的合适位置；只有本机路径的证据必须明确标记为本机可用。

```json
{
  "completed": ["已实现凭据校验与登录错误提示，具体代码见 src/auth"],
  "remaining": ["需要主线集成后验证导航与登录的联合流程"],
  "decisions": ["复用现有认证接口，理由：避免新增重复的会话状态"],
  "risks": ["尚未运行生产部署验证"],
  "next_steps": ["协调者先集成 ops/login，再运行 PROJECT.md 中的联合验收命令"],
  "verification": [
    {"command": "填写实际运行的命令", "exit_code": 0, "summary": "填写实际结果与适用范围", "evidence": ["tests/auth/result.txt"]}
  ]
}
```

```text
python .ops/tools/ops.py checkpoint --task login --agent codex --session session-20261009-a --state ready --files src/auth/login.py tests/auth/test_login.py tests/auth/result.txt --report .ops/local/login-report.json
python .ops/tools/ops.py release --task login --agent codex --session session-20261009-a
```

命令是格式示例，不表示项目有这些路径或已通过测试。`--files` 精确列出本模块改动，包含删除和重命名两端。工具会拒绝遗漏的未提交任务文件、超范围文件、缺失证据，以及无成功验证的 `ready`。工具只能检查报告结构，无法证明 Agent 诚实运行了命令；必须实际执行，并在交接中说明证据覆盖什么。

工具同时记录指定文件和整个 Git 可见任务范围的指纹，后续新增或改动其他范围文件也会让 `verify` 失败。Git 忽略的生成物不会自动纳入任务范围；重要生成物、测试配置和外部数据输入应显式指认，或采用项目既有 artifact manifest。失败/暂停使用 `blocked` / `paused`，保留部分工作；不要为了成功提交伪造验证。

自动同步启用时，checkpoint 和 release 都触发同步。先写本地记录、再提交与推送。网络失败命令非零退出，本地 checkpoint/释放动作可能已经成功：先读 `status`，不要重复制造交接或重新抢占。按相同任务执行 `sync --execute` 可恢复待同步提交。

## 接手与恢复

1. 读 PROJECT、协议、status、自身任务 JSON 和最新 handoff；不遍历全部旧聊天。
2. 确认拿到正确任务分支，运行 `verify --task ID`。工具检查前次 HEAD 是当前 HEAD 的祖先、任务文件和范围指纹一致。HEAD 若向前移动，仍需阅读新增提交和不在任务范围内的变更；祖先检查不会证明全部环境一致。
3. 无差异就认领新会话。若上一会话没 release，确认其进程已停，再由协调者 `recover --task ID --agent coordinator --owner-stopped --reason "实际确认方法与证据"`。不会按年龄自动解锁。
4. SQLite 与 JSON 不一致（例如写快照后进程崩溃），doctor 返回非零。保留两份事实；recover 归档旧运行登记和磁盘快照、标 paused。它不自动选择“较新就更正确”，也不恢复成 ready。
5. 交接指纹已变化且需继续修复时，协调者先定位变化并确认正确分支，再用 `claim ... --agent coordinator --reconcile --reason "明确差异及重新验证计划"`。随后实际修复/验证并写新 checkpoint。原交接保留，不能回填成它早已验证过。
6. 不复制 SQLite、活跃锁或机器绝对路径到另一个 clone；跨机器并行须单一协调者预分配范围/分支，或使用中央认领服务。GitHub push 不能作为跨机器的分布式文件锁。

## 主线集成

`ready` 的任务备份到其分支后，释放认领。协调者按项目已经授权的 PR/审核/验证流程集成，并保存合并后验收证据。工具不自动 merge；主线发布沿用用户实际授权。

```text
python .ops/tools/ops.py finish --task login --agent coordinator --source-commit 实际包含该模块的提交SHA --evidence docs/validation/login-integration.txt
```

finish 只登记结果：当前必须是配置主线，任务已 ready 且释放，指定 source commit 已包含在当前主线历史，source 中 checkpoint 的文件指纹一致，验收证据存在。普通 merge 用任务交付 commit；squash 后应明确指定实际包含该工作的新提交并核对结果。登记不替代真正执行的集成验证。依赖此时才允许开工。完成登记和新的证据由协调者按既有主线授权精确提交/推送；自动 checkpoint 只更新任务分支。

## 避免文档负担

只在可独立验收模块完成、交接、暂停和重要阻塞时写 checkpoint。实现细节以代码和 commit 为索引；交接写当前事实、理由、未完成和可执行下一步。历史记录不覆盖；当数量增长时保留索引与 Git 历史，只按需阅读相关记录。敏感信息不进入报告，较大日志/数据/模型用外部工件引用。
