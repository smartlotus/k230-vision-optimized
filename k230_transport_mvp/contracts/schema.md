# Transport MVP contract v1

当前新增统一视觉调试入口和可选非阻塞会话，不改变UART v1字段或权限。VisualInspector只导入像素配置，五模式相机绑定CSI1/2，CALIBRATIONS始终为空；切换/取帧失败清空候选和稳定历史。TOP需要独立surface_shape及相机/成像配置绑定，合成文件参数不作为实拍参数。WorkflowSession只保存意图和回执，不发送命令；单请求在途、ACK/BUSY不推进、DONE证明后推进、超时UNKNOWN不重发、STOP保留未决、恢复重验。它尚未连接TaskPreview或真实控制器，详见FULL_WORKFLOW_PLAN.md及VISUAL_COMPLETION.md。

来源原工作区 docx/ai01.md；未发现 starter 协议实现，v1 为本工程首个实现。

## TaskPlan / Step

parse_task(payload) raises ValidationError(ValueError) for invalid text, returning TaskPlan with payload, batch1_colors, batch1_slots, batch2_colors, batch2_slots, color_to_buffer_slot.
All groups are exactly three ASCII digits; colors in 1..6, slots are a permutation of 1..3. Colors in both batches are the same set with no duplicates.
build_execution_steps(task, capacity=3) returns Step objects with action, batch, phase, source_station, dest_station, target_color, target_slot, pick_index, layer, requires_carrier_unload, preconditions. capacity is integer 1..3. Steps never carry an object in the gripper across stations; pick means confirmed placement onboard.
A complete phase of three PROCESS placements precedes any PROCESS pick; BUFFER first layer complete before batch2 raw pickup. For 156+123+516+231, batch2 buffer slots = 2,1,3.
MotionPort drive_to(station), pick_to_carrier(color_id,source,slot), place_from_carrier(color_id,dest,slot,layer), return_to_start(start_id). Receipt status DONE/BUSY/ERROR/UNKNOWN plus command_id; accepted ACK alone is not completion. Never retry uncertain motion automatically.
Stats record confirmed object picks and floor/stack placements separately from onboard transfers: complete mission expected picks=12, placements=12 (SIM operational counters, not official score). Final location original START_1 or START_2.

## Observation

Dictionary keys: status, kind, camera_id, frame_id, timestamp_ms, target_id, coord_frame, u_px,v_px,x_mm,y_mm,angle_deg,score_hint,stable,is_simulated,calibration_id,raw_payload.
status in OK,NOT_FOUND,UNCALIBRATED,UNAVAILABLE,ERROR; reason contains details (including ID_UNKNOWN). kind in QR,COLOR,RING,LANE,TOP.
coord_frame IMAGE_PX,ROBOT_MM,WORLD_MM,NONE. Non-OK never has x_mm/y_mm. Detection may be OK IMAGE_PX yet unusable for motion. score_hint is heuristic, not probability.
Extra fields may include station, plane, image_profile, target_pose_valid, stale_after_ms, ring_support, offset_px. Calibration is tied to camera_id, plane, station, image_profile (resolution/crop/flip/rotation), real source and validation. H stays null until measured. QR has no mechanical pose.
All timestamps use local monotonic milliseconds. UART consumers evaluate local receive age; sender clock and MCU clock are not assumed synchronized. A new session must reset replay state explicitly while motors remain disabled.

## UART

ASCII LF-terminated $BODY*HH\n, XOR of ASCII BODY bytes, two uppercase hex digits. Max line 768 bytes (including delimiters). C=control, V=vision, A=ack.
Precise field order and examples in protocol_samples.txt. No CSV quoting: commas, stars, dollars and line breaks are rejected in fields.
Sequence numbers 0..65535, strictly increasing in one explicit session. Reset/reconnect does not clear physical action uncertainty. BUSY/ACK must not increment completed counters. Invalid checksum/type/field count/range discarded.
Only mode/heartbeat/stop controls are exposed on K230. Board UART never sends motor commands. LIVE motion boundary independently validates real source, fresh timestamp, valid H binding, stable target, reachable work envelope, safe lane/known obstacle clearance, explicit enable and current observation identity.

## Rules reference

### Optional RING center references

Per-station `center_gates` may supplement ring ID recognition at an explicitly verified parking pose. Keys are known ring IDs; each value has measured `center=[u,v]`, positive `radius_px`, integer `min_support>=1`, and positive `min_distance_gap_px<=radius_px`. Numbered circular regions must not overlap. Within the requested known region, the nearest qualified candidate must be separated from the next candidate by the configured distance margin; otherwise ID remains unknown. The existing camera/image-profile binding remains mandatory. Missing geometry never falls back to guessing an ID. Fixture references are explicitly simulated and never installed as real parameters. This optional configuration adds no motor authority or millimeter calibration.

Official first-round example frozen from user specification; references:
https://www.shiep.edu.cn/_upload/article/files/61/d5/340105744e3c9ee39242687d8129/0958691b-a187-48e7-a642-8440ce5b65c8.pdf
https://www.shiep.edu.cn/_upload/article/files/61/d5/340105744e3c9ee39242687d8129/b2c2cdc8-2595-4b35-9d10-bf6c10cdd220.pdf
Board API grounded in local cached examples and stubs, plus
https://wiki.lckfb.com/zh-hans/lushan-pi-k230/image-recog/use-sensor.html
https://wiki.lckfb.com/zh-hans/lushan-pi-k230/basic/uart.html

