# ACT 自动路线与 K230 发布器准备

ACT056–062 的实际链路是：API 启动私有离线 Kaggle compute → API 启动 CPU packer 挂精确输出、核验并打包 → 私有 GitHub Actions 获取签名源 archive、上传 Google Drive、全字节读回 → host 验收 compact receipt → 下一批。ACT055 曾同样遇到 API Save Secret 失败，使用编辑器 Save&RunAll 才通过；ACT056 起的独立发布器减少了逐批人工。

证据位于用户已有工程：`C:/Users/28389/Desktop/doastraforgpt/reports/three_lines_completion_20261104/act/cloud_first/{auto_api.py,pack_compute_cloud.py,transfer_compute_cloud.py,README.md}`、`research/act_cloud_first/ci_transfer.py`、`.github/workflows/act_cloud_transfer.yml`。实际 ACT056 workflow run42799289664 /31,210,610字节，ACT062 run43084957514 /19,683,116字节，均有完整DriveSHA读回。未重启或改动旧ACT/A/B研究。

K230此前采用每Notebook的 runtime Secret 直写Drive；Kaggle Save API无Secret关联字段且不会可靠继承，导致人工编辑器操作反复出现。这是执行路线选择造成的成本，计算API本身可自动启动。后续准备切换到ACT同类分离发布路线。

## 可审查实现

- `research/topface/publisher/ci_transfer.py`：从已有ACT实现适配，逐文件来源SHA及变化见 `SOURCE.json`。固定K230 Drive目录 `1MwltNmaI24HjvcxnG1w608qui42RAPKf`，1GiB归档上限、8MiB分块、600s传输限时、15min workflow。先验证签名源全部字节size/SHA，再同一resumable session上传，再验证目标id/name/parents/size/SHA并读回全部字节。
- `research/topface/publisher/workflow.yml`：审查模板，尚未安装或触发。拟目标为现有私有仓库 `smartlotus/k230-vision-optimized` 的独立 `ops/topface-transfer-*` 分支，权限仅contents:read；checkout固定官方commit、不持久化Git凭据、仅首次run_attempt。无main合并或公开发布。
- 输入严格限定字段，拒绝refresh_token等多余项、非Kaggle源域、非Google上传session、其他Drive目录、过期token和身份不匹配。错误不输出签名URL、sessionURL或凭据。
- host将通过gh secret set的stdin提交临时信封，不在源码、Git、命令参数或日志放值。终态后删除该临时Secret；传输不确定时保留原file ID/session/branch，核对原操作，不重算科学任务。

## 需要一次明确授权的新增范围

允许上述**K230私有GitHub Actions发布器**临时接收：短时Google Drive **access token**及expiry、Kaggle签名archive下载地址、单文件resumable session URL、非秘密archive身份。用途限定为K230目录的归档和全字节读回，后续同范围可自动重复，终态清理Secret。长期refresh token/client secret/Kaggle API token留在现有本机凭据源，不发送Actions。

ACT令牌实测为drive.file scope；这不是技术上的单文件授权，单文件限制来自upload session及发布器代码。K230 sender必须核对实际既有OAuth scope，按实际范围告知，不能虚称token只访问一个文件。代码备份授权不自动覆盖把Google访问令牌发给Actions；旧ACT成功也不证明K230新目的地已获授权。

当前状态：CODE PREPARED，offline测试后可审查；尚未向GitHub发送任何Google凭据、未触发publisher。授权后先用既有63KiB审计归档做一次云到云技术探测（不重算数据/训练），通过后03训练及后续归档可由API控制。
