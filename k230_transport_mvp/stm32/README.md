# STM32 接入接口（硬件无关粗版本）

2026-11-09新增材料整帧S协议及自持数值缓存，见
[材料通信规范](../MATERIAL_COMMUNICATION_V1_20261109.md)。
`material_receiver.c/.h`可离线编译测试；当前未接入实时生产者或Transport_car/UART，
完整性/逐物块稳定性和active/draft参数绑定仍须先贯通。下文C/A/V及运动接口为既有边界。

`protocol.c/.h` 不依赖 HAL、RTOS、UART 外设或电机。PC 编译和双语言协议向量已验证；真实 STM32 UART、中断/DMA、超时看门狗、急停和电机状态机尚未实测。视觉帧只提供观测，不能授权电机。

在 UART RX 回调把收到的字节存入项目现有的有界队列。主循环逐字节调用 `k230_feed(&parser, byte, local_monotonic_ms)`。返回 1 时读取 `parser.frame`；`fields` 指针只在下次 feed 前有效，需要保留的内容必须有界复制。空闲主循环调用 `k230_expire`。解析器占约 1 KB RAM，无动态分配。

`fields[0]` 是 C/V/A/S；`fields[1]` 是发送端序号。完整索引见 `../contracts/protocol_samples.txt`。V 中 `fields[20]` 为 image_profile 的 canonical JSON 百分号编码，C 端保留为不透明 ASCII 绑定标识，必须与当前配置的完整 canonical token 逐字比较。不要只比较分辨率，也不要丢弃 crop、翻转、旋转或停车 pose_id。reason/raw_payload 同样可能百分号编码；不要把这些诊断字段当控制指令。C 解析器验证帧、字段数、整数、有限数、布尔、坐标语义、转义语法、长度和序号；它不执行 JSON 标定语义验证或运动安全门。S索引不同，需按新规范处理。

建议上层处理规则：

1. 上电、断线、会话耗尽及 STOP 后保持驱动失能。只有电控自己的显式授权能晋升使能；MODE、PING、V、ACK 都不能使能电机。
2. C 的 MODE 仅切换视觉任务；STOP 请求停视觉并要求电控走独立停车/急停路径。PING 只查询存活。
3. V 收到时记录 MCU 本地接收时刻；发送方 timestamp_ms 只用于该发送方时间域，不能与 MCU ticks 直接相减。接收帧新鲜不等于标定、可达、通道净空或电机授权成立。时间有效性与具体命令请求/观测身份仍由电控独立验收。
4. 对 IMAGE_PX、SIM、缺 H、旧观测、未知停车姿态、错 camera/plane/station/profile/calibration_id、未知目标身份或通道净空，一律拒绝动作。RING 的 target_id 是槽号，TOP/COLOR 的 target_id 是颜色号；二层放置要匹配颜色。
5. ACK/BUSY 只代表接收/在执行；仅同 command_id 的 DONE 更新物料账本。运动超时或传输异常保留 UNKNOWN，不自动重发。重连或重置 UART 序号不能清除运动不确定性。
6. 本会话每个方向共享一条严格递增 0..65535 序列，禁止环绕。A 的 command_seq 关联被应答命令，A 自身 tx_seq 仍递增。只有电机确认失能后才能显式 `k230_reset_session(..., 1)`；此函数只重置协议接收状态。

PC 验证（工程目录）：`python -B -m unittest discover -s tests -p test_protocol.py -v`。此测试调用 gcc，以 `-std=c99 -Wall -Wextra -Werror` 编译临时可执行文件，运行超时/序号/溢出/时钟环绕 C 自测，并将同一正负流向量交给 Python 和 C，比较实际接收帧。临时产物自动清理，不需要把 `.exe` 部署到板端。

`ports.motion_port.LiveMotionPort` 是 Python 侧额外边界，默认失能；其标定要求除视觉工具输出的 `validated=True` 外，还需要电控/人工签验后的 `status="VERIFIED"`、真实来源、有限可逆 H、完整绑定、停车 pose_id 和明确运动授权。它不是 STM32 电机控制器的替代品。此交付没有建立任何实际电机 transport。

LiveMotionPort 每个 camera 的 frame_id 在提交 transport 前消费，DONE、ACK、错误或重新 update 同一旧帧都不会使其恢复；后续动作必须有更大的 frame_id。PROCESS 抓取必须提供槽号，BUFFER 不允许抓取，PROCESS 不允许第二层放置。

导航（drive_to/return_to_start）单独走 `SafetyGate.check_navigation`，允许 QR/NONE 观测，但 QR payload 不参与运动授权。配置必须同时具有总体 `motion_enabled=True`、`motion_authorization="EXPLICIT_ENABLED"`，以及 `navigation_authorization="EXPLICIT_ENABLED"`、`current_pose_verified=True`、`current_station` 与观测/binding 的 station 一致、`route_verified=True`、`route_destination` 与本次目标完全一致、`navigation_timestamp_ms` 为本地近期安全状态时刻、`lane_safe=True`、`obstacle_clear=True`。真实稳定观测、本地 capture/receive/nav 三时间新鲜性、current_observation_identity、camera/station/完整 profile/非空停车 pose_id 绑定同样必需。任何缺失字段保持拒绝；不从 QR 推导姿态、H 或路线批准。
