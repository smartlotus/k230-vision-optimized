# K230 时序神经网络定位

本线使用连续 OV2735 RGB 帧预测末帧的相机光心 `camera_x_mm` 和 `camera_y_mm`。主线比较 CNN+GRU 与 CNN+因果 TCN。单帧 CNN 是共享数据和评估端点的对照。CPU 网格滤波属于独立工程。

离线实现与逐项证据见 [OFFLINE_STATUS.json](OFFLINE_STATUS.json)。当前已提供数据审计、Kaggle 训练与完整归档读回、多帧对照、算子失败记录、TCN KModel 逐帧验证、部署环形缓冲与故障回放、真实图像回放入口和有界 CanMV 试验包。实物条件汇总在 [统一清单](HUMAN_INPUTS.md)。整体项目尚未通过定位质量和原生验收。

使用官方图 1 对应的田字形修正版场地。两个蓝区分别作为随机起点和另一端终点。数据包含 Smoke 100 帧、Pilot 1,000 帧，以及单独渲染的 300 帧空间与外观挑战。Pilot 的 40 条轨迹完整分组，500/250/250 帧分作训练、验证和测试。

## Kaggle 试训

先对已保存的 A06 五个图复验。每个设置完成 192 次更新，单帧为对照，GRU/TCN 比较 4 和 8 帧。再运行本线独立的联合训练 A01，更新共享 RGB 编码器和 XY 概率图头，比较 CNN-1 与 TCN-8。两个设置各训练 24 轮和 384 次参数更新，验证集均选择第 9 轮。精确保存版本为 [Kaggle Joint A01 v1](https://www.kaggle.com/code/lotus30/k230-neural-camera-xy-joint-a01)，使用 Tesla T4。保存输出已经从该版本完整读取，核对归档 SHA256 和 11 个成员。证据见 [读回回执](cloud/joint-a01/ARCHIVE_READBACK.json)。

下表为相同 250 帧、10 条轨迹、48×48 网格的直接神经网络输出。误差单位为 mm。

| 试验 | 中位误差 | P95 | 超过 500 mm 跳变 |
| --- | ---: | ---: | ---: |
| A06 CNN-1 | 913 | 2118 | 84 |
| A06 GRU-4 | 922 | 2118 | 63 |
| A06 GRU-8 | 926 | 2118 | 75 |
| A06 TCN-4 | 851 | 2041 | 65 |
| A06 TCN-8 | 926 | 2033 | 66 |
| 联合 A01 CNN-1 | 909 | 1964 | 105 |
| 联合 A01 TCN-8 | 754 | 1991 | 48 |
| 联合 A01 TCN-8 KModel Simulator | 768 | 2004 | 55 |

A01 TCN-8 重复末帧填满窗口时，中位误差升至 888 mm。只反转历史帧、保留末帧时升至 862 mm。对完整 8 帧窗口的 180 个端点也单独计分。丢失连续 6 帧后，10 条轨迹中 3 条达到连续 3 帧误差 ≤200 mm。验证集未找到能同时覆盖 25 帧且达到 95% 的 200 mm 内正确率的置信度阈值。双蓝区初始歧义需要后续真实视野和不对称地标来验证。

原始矩阵见 [A06](A06_NEURAL_ONNX_REPLAY.json)、[联合 A01](JOINT_A01_NEURAL_ONNX_REPLAY.json)、[故障与置信度](JOINT_A01_NEURAL_FAULTS.json)和 [KModel 同帧对照](JOINT_A01_TCN8_KMODEL_TEST.json)。

## 空间与外观挑战

先冻结路线再运行模型。100 帧空间组覆盖四条新轨迹。其中 55 个完整窗口的末帧距原训练轨迹的分段连续折线至少 100 mm。有 7 个窗口的全部 8 帧满足这项距离条件。另有 100 对外观帧覆盖四条新路线，全部保证相机、XY 和语义掩膜相同，分别渲染基准和暖光或冷光。数据用于离线评估，Blender QA 与 ZIP 回读见 [渲染回执](NEURAL_CHALLENGE_RENDER_QA.json)。

外观训练 A02 沿用原始 1,000 帧 Pilot，两个设置各 24 轮、384 次更新。训练对每个 RGB 窗口施加整段一致的亮度、RGB 增益和 gamma 随机扰动，验证和挑战图像保持原样。[私有 Kaggle 保存版本](https://www.kaggle.com/code/lotus30/k230-neural-camera-xy-appearance-a02)的归档按原版本完整回读，11 个成员的大小与 SHA256 均一致。A02 在原 Pilot 留出集上的 TCN-8 中位误差为 544 mm、P95 为 1,919 mm，超过 500 mm 的跳变为 76 次。A01 在相同数据上分别为 754 mm、1,991 mm 和 48 次。

| TCN-8 挑战切片 | 帧数 | A01 P95 | A02 P95 |
| --- | ---: | ---: | ---: |
| 空间挑战全量 | 100 | 1,959 mm | 1,939 mm |
| 完整窗口且末帧距训练路线 ≥100 mm | 55 | 1,716 mm | 1,744 mm |
| 基准光照 | 100 | 1,924 mm | 1,998 mm |
| 暖光 | 50 | 2,130 mm | 2,122 mm |
| 冷光 | 50 | 2,086 mm | 1,893 mm |

逐帧与基准光照配对后，误差增量的 P95 从 1,635 mm 降到 1,266 mm。55 帧空间组的尾部误差略升，说明光照增强仍不足以解决陌生路段定位。A06 的 GRU 和 TCN 同样跑了这批挑战。本次实验用于诊断，后续模型选定后须另用新的空间与外观样本验收。

证据见 [A06 挑战](A06_INDEPENDENT_HOLDOUT.json)、[A01 挑战](JOINT_A01_INDEPENDENT_HOLDOUT.json)、[A02 挑战](APPEARANCE_A02_INDEPENDENT_HOLDOUT.json)、[A02 原 Pilot 对照](APPEARANCE_A02_NEURAL_ONNX_REPLAY.json)和 [A02 云端读回](cloud/appearance-a02/ARCHIVE_READBACK.json)。

## 板端试验与下一轮

联合 A01 试验包含已校验的 KModel、浮点与 uint8 ONNX、权重、固定窗口配置和 CanMV 入口。nncase 2.11.0 使用训练集内 100 个窗口做 A16 PTQ。输入为 `[1,24,96,160]` 的 uint8 RGB，按从旧到新的顺序排 8 帧。测试集完整窗口的中位跨度为 2.44 秒。试验以训练集中位帧间隔 388 ms 采样，间隔超过 700 ms 后重新建立窗口。这两个值来自仿真，用于试验。包内 `RUN.md` 给出独立 SD 目录入口和参数，打包事实见 [试验包回执](JOINT_A01_BOARD_TRIAL.json)。

已新增几何 A03 Pilot。与原 Pilot 相比，训练集中固定地标占画面至少 1% 的帧从 95/500 增至 140/500。250 帧本地测试中有 36 个窗口的全部 8 帧距新版训练轨迹至少 100 mm。这批测试图像未进入几何训练输入数据集。单帧和 TCN-8 各训练 24 轮、384 次更新，使用相同验证选择。结果见 [几何 A03](GEOMETRY_A03_NEURAL_REPLAY.json)。历史帧 XY 辅助监督 A04 使用同一训练输入、同一时序网络，辅助权重 0.35，在 [A04](HISTORY_A04_NEURAL_REPLAY.json) 中单独计分。

| 新版相同 250 帧本地测试 | 中位误差 | P95 | 超过 500 mm 跳变 |
| --- | ---: | ---: | ---: |
| 原外观 A02 TCN-8 | 1,360 mm | 2,291 mm | 109 |
| 几何 A03 单帧 CNN | 1,129 mm | 2,074 mm | 87 |
| 几何 A03 TCN-8 | 985 mm | 2,205 mm | 65 |
| 历史辅助 A04 TCN-8 | 1,097 mm | 2,159 mm | 见原始报告 |
| 几何 A03 TCN-8 KModel | 971 mm | 2,205 mm | 65 |

几何 A03 KModel 与浮点模型的首选锚点一致率为 94%。原生 GRU4/8 算子在实际已训练图导入时触发 nncase 2.11.0 的 VisitTranspose/CheckedType 失败，日志保存在 [GRU_IMPORT_SMOKE.json](GRU_IMPORT_SMOKE.json)。当前试验使用 TCN-8。

A03 的常规外观测试属于同光照类型的新随机场景。另补了 100 对相同相机位姿的严格外观测试，`warm_reserved` 和 `cool_reserved` 两种灯光参数在训练中从未出现。A03 的暖光和冷光 P95 分别为 2,038 mm 和 1,917 mm，A04 分别为 2,078 mm 和 1,968 mm。辅助历史 XY 监督在这项测试上也没有改善尾部误差。[A03 严格外观](GEOMETRY_A03_STRICT_APPEARANCE.json) · [A04 严格外观](HISTORY_A04_STRICT_APPEARANCE.json)

新的试验包绑定 [A03 时间配置](GEOMETRY_A03_RUNTIME_PROFILE.json)，采样下限约 230 ms，长间隔重置约 730 ms。配置来自训练时间戳分布，仍需真实相机校验。主机与板端使用同一个 FrameWindow 和候选解码函数。实际窗口生命周期下的丢帧、黑屏、不规则间隔和远处重定位见 [运行回放](GEOMETRY_A03_RUNTIME_REPLAY.json)。板端会逐帧保存时间戳、窗口内时间和候选，便于之后与光心 XY 真值同步。

正式使用仍处于 HOLD。试训 P95 门限为 500 mm。几何数据改善了中位误差和跳变，远距离误判仍多，置信度校准也未通过。已完成本轮有界离线试验和软件验证。原计划的 10k–20k 阶段以 Pilot 通过为前提，目前尚未进入。需要实物阶段验证安装视野、起步条件和真实外观，再决定后续数据与模型修订。

TCN-16 延长窗口实验 A05 复用几何 A03 的训练与验证，24 轮、384 次更新，以验证 NLL 选第 16 轮。实际 ONNX 采用四层 kernel=2、dilation=1/2/4/8 的因果卷积，输入 `[1,48,96,160]`。250 帧原测试的中位误差为 1,050 mm，P95 为 2,146 mm。KModel 完整同帧测试的 P95 相同，首选锚点一致率为 96%。[云归档读回](cloud/long-a05/ARCHIVE_READBACK.json) · [普通窗口消融](LONG_A05_WINDOW_ABLATIONS.json) · [量化对照](LONG_A05_TCN16_KMODEL_TEST.json)

另行冻结并渲染了六条西侧路线，共 240 帧。原几何测试只有 8 帧全历史空间留出，新的长路线提供 90 个整段 16 帧均距训练折线至少 100 mm 的末帧。单帧、TCN-8、TCN-16 用相同的 150 个 index≥15 端点计分，完整空间留出切片上的 P95 分别为 2,194、2,216 和 2,006 mm。TCN-16 在该切片上降低尾部误差，但仍未通过 500 mm 门限。90 个窗口来自六条相近路线且相互重叠，本次结果限于该单种子试验。[完整比较](LONG_A05_SPATIAL_REPLAY.json) · [长轨迹数据回执](LONG_SPATIAL_DELIVERY.json) · [TCN-16 试验包回执](LONG_A05_BOARD_TRIAL.json)

地面语义可观测性审计使用相同相机矩阵，把视野旋转到另一端蓝区。40 条几何轨迹的首帧都存在地面语义完全相同的替代起点，12 条轨迹的前 16 帧仍无 1% 以上的语义差异。这是地面类别层面的分析，RGB 纹理和真实相机是否可区分仍需单独验证。[审计记录](GEOMETRY_OBSERVABILITY.json)

连续空间输出头 A06 使用八个学习到的 XY 高斯分量，保持 A03 的 TCN8、750 帧训练/验证输入、seed 117、24 轮和 384 次更新。云版本 1 的模型文件与完整归档已经回读校验。250 帧本地测试中位误差为 1,037 mm、P95 为 2,189 mm；完整空间留出的 90 个末帧分别为 1,838 mm 和 2,179 mm。相同切片的 A03 TCN8 中位误差为 1,584 mm，本轮未升级为主模型。严格暖光与冷光 P95 为 2,088 mm 和 2,018 mm。测试和恢复继续 HOLD。[冻结评测规则](MIXTURE_EVALUATION_PLAN.json) · [普通测试](MIXTURE_A06_GEOMETRY_REPLAY.json) · [长空间测试](MIXTURE_A06_SPATIAL_REPLAY.json) · [严格外观](MIXTURE_A06_STRICT_APPEARANCE.json) · [运行故障回放](MIXTURE_A06_RUNTIME_REPLAY.json)

GRU 的原生循环节点导入失败仍保留原始记录。新增固定四/八帧展开将同一已训练 ONNX 的门运算展开成基础算子，复用原权重和零初始状态。两个设置均在全部 250 个原测试帧上与原始 ONNX 对照，首选锚点一致率为 100%，最大差异不超过 5.73e-6；另以随机输入和独立复核检查了门公式及输出顺序。后续量化与板端结论以各自报告为准。[浮点等价记录](GRU_UNROLLED_FLOAT_PARITY.json) · [转换源码](lower_fixed_gru.py)

A06 的实际训练权重直接转换后，在第 12 个测试窗口出现 3 个 NaN。改用显式减最大值的 log-sum-exp 等价图后，浮点 250 帧首选锚点仍全部一致；KModel 的 250 帧模拟输出均有限，P95 为 2,165 mm，但首选锚点与浮点只在 26.8% 的帧上相同。99.6% 的帧位置偏移不超过 100 mm，另有一帧偏移约 2.20 m。数值稳定恢复与任务精度分别记录，该候选继续 HOLD。[原失败](MIXTURE_A06_DIRECT_COMPILER_DIAGNOSTIC.json) · [稳定图浮点校验](MIXTURE_A06_STABLE_FLOAT_PARITY.json) · [量化全帧对照](MIXTURE_A06_TCN8_KMODEL_TEST.json) · [位置偏移审计](MIXTURE_A06_QUANTIZATION_DISPLACEMENT.json)

原 GRU Pilot 的部分留出帧间隔短于训练侧采样下限。运行回放按真实时间戳产生 12 帧正常跳采，clean 计分为 238 帧；这与固定窗口图的完整 250 帧对照属于两个计分条件。回放审计按采样下限与长间隔阈值独立核对采样数、跳采数及重置次数。[GRU4 运行回放](GRU4_UNROLLED_RUNTIME_REPLAY.json) · [GRU8 运行回放](GRU8_UNROLLED_RUNTIME_REPLAY.json)

固定展开后的 GRU4 与 GRU8 均完成原始 A06 Pilot 相同 250 帧的量化对照，P95 分别为 2,129 mm 与 2,118 mm，首选锚点一致率分别为 66.4% 与 63.6%。两份独立板端试验 ZIP 均已完整回读，真实设备验收继续待人工接入。[GRU4 量化](GRU4_UNROLLED_KMODEL_TEST.json) · [GRU8 量化](GRU8_UNROLLED_KMODEL_TEST.json) · [GRU4 试验包](GRU4_UNROLLED_BOARD_TRIAL.json) · [GRU8 试验包](GRU8_UNROLLED_BOARD_TRIAL.json)

长轨迹生成器所需的两个渲染器已按原 QA 的 SHA 冻结，配套 sim、场地配置和 contract 一同保存。用冻结目录新渲染了首条完整 40 帧轨迹，图像、掩膜、光心 XY、时间戳和元数据逐项相同。[依赖清单](LONG_RENDERER_SOURCE.json) · [新目录复验](LONG_RENDERER_PORTABILITY.json)
