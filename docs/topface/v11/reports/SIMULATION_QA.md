# V1.1 Smoke QA

本地原生 Blender 4.2.9 CPU 实际生成 100 个独立场景、246 个实例，24 samples，共 442.281 s。0–5 实例场景分别为 17/17/17/17/16/16；六色实例分别为 32/49/33/50/33/49。自动 QA 无问题；用户查看 30 张分层选图后明确回复“抽查通过，继续 Pilot”。

检查包括：原始融合 CAD 的 Ø30 顶面材质 pass、可见 bbox、物理 3D 圆心独立投影、完整边界与 conic ellipse、center_valid/visible、遮挡与出框、空场、全部五个槽、重复 RGB/seed/scene group、相机配置组跨 split。最大物理圆心重投影差为 1.2711×10⁻¹³ px。75 个相机配置组无跨 split；split 在训练前按 height/pitch/HFOV 配置组确定。原始 labels_raw 保留生成提示，最终 labels 记录派生分组，不为 split 变更重渲染。

范围为镜头到顶面 100–400 mm、光轴与顶面 80–100°，物块直立。超出该真实使用范围的强透视组明确排除，未声称覆盖。真实内参/畸变、失焦、ISP、真实中心和椭圆精度仍未验收。

完整 Smoke 归档 30,052,798 字节，SHA-256 `271857352a7dc8abb7d861436ddac66f41722055e02d6f0951f99f6c54ed65d4`。Drive file ID `1cn3RMsr0nryMGoWE_wCgw4BBlXMoBQCo`，已全字节读回匹配。该数据由本机生成后迁移归档，不冒充云端生成。原始 Smoke 没有生成前完整源码冻结；原始逐场景位姿/相机/seed、实际渲染收据与此限制均保留在归档。新 Pilot 增加了输入源码冻结及 manifest 验证。

新版 worker 本地端到端验证为 6 场景/15 实例，见 `native_worker_qualification.json`。14 项主机回归检查通过，见 `host_all_tests.log`。覆盖新增的跨会话准入拒绝、过期预算停止、渲染及归档都失败时仍保存 compact recovery。网络及 Linux/GPU 的最终结果以实际 Kaggle v2 回执为准。
