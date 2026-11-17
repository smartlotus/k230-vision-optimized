# Temporal V1 package audit — 2026-11-17

Read-only extraction and evidence review; no Git worktree, board, cloud workload, original archive or original extracted source changed.

## Integrity and extraction

- Original ZIP: 6,043,252 bytes; SHA256 `ce4b8e342209b5948435635974a34feeebcfdac52586dd79adb600f0e118e691`; 28 members.
- Recursively extracted to this audit directory, including all nested CameraXY V2, previous Temporal, RoadNet, TopFaceNet and CAD archives (166 extracted files).
- All 13 SHA256 manifests pass, including 27 top-level members. The user's existing unpacked directory matches all 27 top-level hashes.
- 77 unique UTF-8 text source files (234,015 bytes) were inventoried. Core contracts/code and relevant reuse requirements were reviewed. Historical task instructions are source documents, not new authorization.
- Package includes plans, starter code, field PDF/JPEG and CAD; no trained CameraXY checkpoint, ONNX, KModel or Blender .blend is included.
- Detailed hashes and actual checks: `AUDIT_PACKAGE_INVENTORY.json`.

## Frozen contract

Only camera_x_mm and camera_y_mm are supervised. Metadata yaw, height, pitch, roll, intrinsics, timestamps and world transforms do not become extra labels. The XY target is the optical-center translation projected vertically into the map plane, independent of optical-axis tilt. No chassis target or camera-to-robot offset is inferred.

FIELD_MAP origin is field center; x right on official figure, y up, z up, mm; XY bounds [-1200,1200]. Blender scene uses meters; target conversion is exactly 1000. Blender downward camera local -Z points to ground at Euler (0,0,yaw); the V2 sample showing pitch=-90 and an identity transform is explicitly illustrative and cannot be copied into Blender.

Current packet camera 400mm height is USER_ESTIMATE; 2.88mm lens/96deg diagonal FOV are manufacturer references, not calibration. Historical 640x360 is not proof of native negotiated UVC mode. Camera intrinsics, distortion and mounting transform remain UNKNOWN. Speed/margin UNKNOWN -> no strict hard gate. Recover globally after LOST.

## Reuse entrypoints

Use the already advanced local experiment rather than the old embedded starter:
`C:/Users/28389/Desktop/k230-project/webhelp/K230_RoadNet_Boundary_Sim_Execution_Package/experiments/roadnet_boundary_sim_v1/`

- `blender_scene.py`: build(size), apply_record(parts, rec, size), render(parts, rec, out, size).
- `sim.py`: layout/appearance/obstacle metadata templates. Do not use capture_records unchanged as physically continuous temporal GT: it has per-frame independent +/-16mm jitter and no timestamp.
- `configs/field_layout.json`: advanced inferred blue centers (1.05,+/-1.05)m and turntable (0,1.27)m. Old embedded starter uses (.9,+/-.9)m and script turntable 1.12m, while old config says 1.2m. Pin and use one chosen source identity.
- Native material CAD: STL measures ~50x50x60mm, 19,518 triangles; GLB position bounds measure ~.05x.05x.06m. They are the same three-section geometry; scale explicitly at import. Detailed measured bounds in JSON.
- Original package `assets/...RoadNet.../blender/generate_field_scene.py` only renders 3 demonstration views, without labels/masks/randomized dataset/calibrated camera; it is not a continuous-data pipeline.
- CameraXY V2 `scripts/sample_camera_xy.py` only produces POSE_ONLY_NO_RENDER, random independent positions, image=None, no timestamp/trajectory/split. Its pose labels can inform a new sampler, but outputs cannot train temporal models.

## Actual defects and gaps

1. Temporal filter implementation rejects UNKNOWN motion bound after initialization although config specifies hard_gate_enabled=false. Add soft prediction separately and retain measured-bound gated behavior.
2. Sequence builder groups by (split,trajectory), so it silently accepts the same trajectory in train and val and produces duplicate sequence_id. Reproduced with 8 rows, exit=0, 2 sequences both named same_0000000. Evidence: split-leak-probe.jsonl and split-leak-probe-sequences.jsonl. Need trajectory/pose-group split exclusivity, unique IDs, finite strictly monotonic time, file checks and no forbidden nested targets.
3. `CameraXYHead.xy_refinement` sigmoid output has no defined residual-vs-map coordinate decoder or loss; define before training. Quality head has no training or calibration implementation. Derive uncertainty only from XY supervision/validation rather than inventing a new supervised task.
4. CNN single-frame class, trainer, calibration, compile runner and comparison evaluator are absent. GRU/TCN are reference structures, not known supported KModel graphs. All final route model metrics and board timings are NOT RUN.
5. Main package supersedes previous Temporal archive text which says not to train GRU. Current user requires B/C attempts. Historical TopFace restrictions about one model apply to TopFace task, not the present localization comparison.
6. Map interpretation risk: official PDF p3 / figure1 visually shades extensive outer regions gray, while existing scene implements gray central cross on light background. Old docs call non-cross remainder light background. Record this ambiguity and audit against physical/map evidence before claiming scene is official ground truth; do not quietly redraw a new map.
7. No exact current CanMV v1.8-36 compiler/runtime evidence in the packet itself. Local RoadNet/TopFace evidence must be checked; a listed ONNX GRU operator does not establish compiler/runtime support.

## Minimum 100-position temporal adaptation

Freeze one deterministic set of trajectory records with world-space camera positions, timestamps, yaw/appearance metadata and whole-trajectory split. Reuse one built RoadNet scene and apply_record to camera and moving turntable at each real timestamp. Obtain camera XY from actual camera.matrix_world translation, not separately resampled labels. Use the same frames/windows for all six comparisons. Keep pose/appearance variants in one split and reserve spatial/appearance holdouts explicitly. New rendering and all training remain NOT RUN in this audit.
