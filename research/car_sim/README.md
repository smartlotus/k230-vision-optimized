# K230 右侧相机与两批搬运仿真 R2

## R3 连续仿真

R3 的验收、套件和预算见 [执行计划](../../docs/plans/k230-vision-sim-iterations.md)。
从 R3 工作树根目录运行首批 R3.0 用例。

```powershell
$hub = 'C:/Users/28389/Desktop/k230-project'
$out = Join-Path $hub 'experiments/continuous-sim'
python -B research/car_sim/run_campaign.py --hub $hub --out $out --campaign r30-smoke --profile r30 --resume
python -B research/car_sim/run_campaign.py --hub $hub --out $out --campaign r30-smoke --status
```

`r30` 跑冻结 R2 工位观测、6 个合法任务、2 个扫码负例、主机 C/V/A 回放和队列恢复探针。扫码从 Blender 前视机位的 PNG 解码，随后使用现有离散工作流。计划中的视觉行驶和抓放闭环由后续套件验收。
`r31` 增加夹爪台架、右侧道路观测和匹配扰动的直道闭环。直道用新的本地二进制校验帧传递底盘速度，积分后从新位姿重新渲染。裁判从独立场地掩码检查底盘足迹。直道通过后仍需跑转弯、路口、工位停靠和返航。
`r32` 增加找料对齐、下探并用新帧确认、闭爪与试提，以及独立 MuJoCo 释放对照。整轮还要完成转盘和车载仓、平面及二层放置。
`full` 会报告当前实现可执行的用例和因缺失实现或依赖而受阻的用例。报告按套件记录范围。

运行目录包含 `manifest.json`、`queue.sqlite`、`status.json`、`attempts/` 和 `episodes/`。在该目录下创建 `STOP` 文件会在一致断点退出。移除 STOP 后用同一条 `--resume` 命令接续。输入或源码改变后使用新的 campaign ID。报告在 `episodes/delivery/report.md`。

压缩包内的车模、短片和验证记录都在 `research/car_sim/output/`。以下缩写 `output/` 均指这个目录。

打开 `output/vehicle.blend` 检查机械结构，按空格播放。第 1 帧收拢，第 40 帧展开到右侧最高点，第 85 帧在右侧降到最低，第 110 帧回到行驶高度。在 `output/competition_round.blend` 播放整轮参考轨迹，可在时间轴按工序跳转。中文预览在 `output/车体右侧相机.png`、`output/工位相机.png` 和 `output/整轮回放.mp4`。完整的图像决策记录见 `output/live_profile_round.json`。

## 车与镜头的关系

把底盘中心向地面投影作为原点，+Y 向车头，+X 向车右，+Z 向上。对应手绘俯视图，车头在左，车右在上。水平臂随云台绕竖轴转动，臂端固定一段垂直于地面的相机支架，镜头装在支架下端，始终朝正下方。停车收拢时光心投影在底盘内。行驶时，臂端转出右侧车身，光心在前后方向对准右前、右后两个轮心的中点，看到这两个车轮外侧的路面。参考动画从起点先按收拢外形行驶 40 mm，再展开。

| 项目 | 数值 | 来源与解释 |
| --- | --- | --- |
| 底盘长、宽、上表面高 | 295、280、79 mm | 手绘尺寸，立柱顶面离地 370 mm |
| 云台轴至光心的水平距离 | 180 mm | 用户确认。图中另标的 100 mm 用于末端条形臂段 |
| 光心工作高度和朝向 | 100–400 mm，行驶时 400 mm，竖直向下 | 用户确认 |
| 展开后光心的前后位置 | Y = 0 mm | 用户确认纵向对准右轮中点。模型两右轮的中点为 (122, 0) mm |
| 展开后光心的横向位置 | X = 256 mm | 暂取光心超出名义车右边界 116 mm。边界 X = 140 mm，光心在右轮中心线外侧 134 mm |
| 云台轴位置、臂端竖直支架高度 | (79.05, 33) mm、25 mm | 暂取转轴在车体中心前方 33 mm，结合 180 mm 半径与展开终点反算 X。支架高度为简模假设 |
| 收拢光心、云台转程 | (−97.9, 66.0, 330.0) mm、半圈 | 收拢高度按图纸基准推算，角度端点暂取 +90° 和 −90° |

`vehicle_dimensions.json` 记录图纸数值、实车确认项和假设，`vehicle.py` 根据模型上的两个右轮轮心确定纵向中点。打开 `output/rig_check.json` 可核对光心、支架轴向和整车网格包围尺寸。收拢时宽、长、高约为 277.9 × 295.0 × 370.0 mm，展开到行驶高度时约为 406.0 × 295.0 × 439.5 mm。第 85 帧只降下臂端，光心仍在右侧 (256, 0) mm，离地 100 mm。

道路相机使用对角视场 96°、640×360 的针孔近似。二维码使用单独设置的虚拟前视安装位。升降连接件和线束采用简化形体，夹持和入仓采用动作代理。实车镜头参数、读码视角、关节限位、全行程间隙、夹指几何及驱动响应仍待测量。

## 整轮运行结果

场地复用 2.4 m 灰图，物料复用原始融合 CAD，物块底面直径 50 mm、顶面直径 30 mm、高 60 mm。粗加工区沿图从左至右编号 3、2、1，暂存区从下至上编号 1、2、3。样例任务码为 `156+123+516+231`。

- **参考动画**复用任务状态机，呈现扫码、转盘依次取三件、粗加工放入与取回、暂存第一层、第二批处理、同色叠放和返航。两种起点的离散车体扫掠分别检查 13,330 和 15,088 个位姿，名义底盘侵入禁区的像素数均为 0。此检查覆盖底盘矩形与固定布局。
- **视觉决策**在每次取放前按当前物料位置渲染图像。解码确定任务，转盘分次呈现物块。取料检查颜色、中心和可见度，首层放置检查空槽的编号和环心，叠放检查槽位中的同色底块。程序用曝光时的相机位姿和工作平面把像素换算成 XY，用图像给出的落点更新场景，使放置偏差传递到下一次抓取和叠放。夹持与释放由几何距离模拟，无法取点或无法确认接触时会停止。

三轮均完成 12 次抓取、12 次放置、三个同色叠层并返回出发点。表中偏差是结束时上下块中心轴线的水平距离，单位为毫米。

| 运行配置及记录 | 1 号位 | 2 号位 | 3 号位 |
| --- | ---: | ---: | ---: |
| 冻结顶面模型取物块中心，`output/live_round.json` | 1.81 | 2.86 | 1.79 |
| 浅色工位的黑块增加 CAD 轮廓复核，`output/live_profile_round.json` | 1.86 | 1.34 | 1.76 |
| 同上，曝光降低 0.40 EV、沿工位偏移 5 mm、改变转盘初相位，`output/live_challenge_round.json` | 1.64 | 1.62 | 1.89 |

黑块轮廓复核要求两个观察位置给出相差小于 4 mm 的平面估计。上述三轮共用同一场地、任务码和预设底盘路径。道路模型单独评测，轮速、打滑、真实夹爪反馈、其他颜色组合和随机障碍仍待验证。

另用参考动画中的 24 张操作前图像检查单帧取点。按约定工位和目标类型限定搜索窗口，24 张均在独立参考中心的 8 px 半径内命中，误选 0 次、重拍 0 次。每张图的搜索窗口、检测中心和参考误差见 `output/mission_vision.json`。

## 抓取、叠放和下位机接口

1. **行驶与到站。** 按工序使用前视扫码或右侧俯视道路。进入工位后锁定批次、颜色、动作、目标槽、云台角和升降高度。可在收臂前锁存末次可靠观测。若底盘、转盘、物料或相机继续运动，后续动作需使用更新后的位姿与观测。原料盘目标坐标只在对应的停歇窗口内有效。
2. **输出坐标。** 对 640×360 源图取画面中心 (320, 180)，调试时可显示向右、向上的误差 `(u−320,180−v)`。下位机接口沿用源图 `u_px,v_px` 定义。要获得车体坐标系的毫米值，需使用曝光时的相机位姿和对应工作面高度，将像素射线与该平面求交。图像坐标轴会随云台转动。实车可在固定工作高度示教夹爪落点，再做小幅已知 XY 移动，验证像素变化的方向和比例。
3. **夹取。** 对齐目标是顶面中心与夹具工作点在同一平面的投影。R2 行驶姿态下，原料抓取点投影为 (320, 232.7) px，工位物块抓取点为 (320, 200.6) px。用两者各自的投影作参考，比统一追画面中心更符合夹具位置。`output/servo_trials.json` 的 6 个扰动工况均达到连续 3 帧偏差不超过 2.5 px，按仿真场景独立核对的最终水平误差为 0.69–3.11 mm。整轮另用 8 mm 模拟夹取及释放回执门槛。实车还需执行接近、闭夹、提起和入仓，并由下位机返回动作结果。
4. **第二次放置。** 先检查第一层的槽位和颜色，求出支撑顶面中心在车体坐标系的 XY。本场景工位面离地 3 mm，第一层顶面再高 60 mm。夹持上层的中心需按自身高度计算 XY，或使用已示教的夹爪偏置。根据同一坐标系里的水平差微调，夹具遮住底块后保持末次可靠参考，底盘或底块移动后重新观测。实车释放后还需撤爪并复核稳定状态。`output/stack_contact.json` 包含 52 组独立的 MuJoCo 理想竖直释放试验。在给定几何和摩擦条件下，偏心 0、5、10、14 mm 的 28 组稳定，16、20、24 mm 的 24 组失稳。实际放置容差取决于摩擦、夹持扰动和释放姿态。

## 模型与训练

在 30 张渲染图上运行冻结的 RoadNet 和 TopFaceNet ONNX。图像覆盖行驶、原料、粗加工、暂存和收拢状态，含曝光、停车偏移和朝向变化。场景射线投射生成道路类别和可见顶面中心真值。冻结 TopFace 在 12 px 同色匹配半径内命中 19 个可见顶面、漏 4 个，另有 7 个未匹配预测。逐帧结果见 `output/perception.json`。这里统计整幅图中的所有顶面，24 张工位图的取点试验只评价本步指定的一个位置。

道路实验按固定种子分成 18 张训练、6 张验证、6 张测试。保持 RoadNet 权重固定，用 RGB 和模型输出训练灰路二分类适配层。测试组灰路 IoU 从 0.7034 提高到 0.8636，同时非道路像素误判可走的比例从 7.96% 升至 8.24%。按预设选用条件，R2 继续使用冻结基线，权重 `output/adapter/candidate.npz` 仅用于复查候选，详见 `output/adapter/report.json`。以上图像来自同一地图与名义路线，板端和现场验收仍暂停。

## 复现

压缩包根目录含本轮所需地图、原 CAD、任务码、工作流依赖、两个 ONNX 和结果。已有项目工作树也可运行，其中 `$hub` 指向桌面的 k230-project 聚合目录。示例在压缩包根目录的 PowerShell 执行。Blender 5.3、Python、NumPy、Pillow、OpenCV、ONNX Runtime、SciPy 用于主流程，MuJoCo 3.3.7 用于接触测试。先选新的输出目录，以保留随包验证记录。

```powershell
$hub = (Get-Location).Path
$sim = Join-Path $hub 'research/car_sim'
$out = Join-Path $hub 'my-run'
$blender = 'C:/Program Files/Blender Foundation/Blender 5.3/blender.exe'
$py = (Get-Command python).Source
& $blender -b --python-exit-code 1 --python "$sim/build_scene.py" -- --hub $hub --out $out
python -B "$sim/perception.py" --hub $hub --frames "$out/rgb" --truth "$out/truth.json" --output "$out/perception.json"
python -B "$sim/fit_adapter.py" --hub $hub --manifest "$out/splits.json" --out "$out/adapter"
python -B "$sim/mission_vision.py" --hub $hub --dir $out
& $blender -b "$out/competition_round.blend" --python-exit-code 1 --python "$sim/servo_trial.py" -- --hub $hub --out $out --python $py
python -B "$sim/mission_vision.py" --hub $hub --dir $out
& $blender -b "$out/competition_round.blend" --python-exit-code 1 --python "$sim/live_round.py" -- --hub $hub --out $out --python $py
& $blender -b "$out/competition_round.blend" --python-exit-code 1 --python "$sim/live_round.py" -- --hub $hub --out $out --python $py --profile-review
& $blender -b "$out/competition_round.blend" --python-exit-code 1 --python "$sim/live_round.py" -- --hub $hub --out $out --python $py --profile-review --challenge
$cad = Join-Path $hub 'webhelp/camera_xy_v2_audit_20261117/k230_material_block_model_bundle/k230_material_block_model/model_validation.json'
python -B "$sim/stack_contact.py" --cad-receipt $cad --output "$out/stack_contact.json"
```

实车下一次最有价值的补测，是把云台转轴、镜头中心、夹爪闭合中心标在同一张俯视照片上，附一把直尺，并记录抓取、第一层、第二层的工作高度。由此可以把仿真落点替换成真实夹具落点。之后分别使用真实转盘停歇时长、读码视角、板端推理延迟和释放后反馈复验。
