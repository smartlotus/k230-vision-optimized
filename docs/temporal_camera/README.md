# K230 competition CameraXY temporal V1

Current implementation branch: `ops/temporal-camera-v1-20261117`. Read `.ops/PROJECT.md`, `.ops/PROTOCOL.md`, and the task handoff before writing. This module is isolated from accepted production, UART and motion code.

## Verified starting state

- [G0 package audit](AUDIT_READOUT.md), [full archive inventory](AUDIT_PACKAGE_INVENTORY.json): original ZIP and all embedded packages extracted; 13 manifests, 169 hashes pass. Reuse the advanced RoadNet scene, not the older inconsistent starter.
- [Tiny operator smoke](g1-smoke-a01/SUMMARY.json), [actual Pilot input resolution smoke](g1-input90x160-a01/SUMMARY.json): CNN, GRU4/8, TCN4/8 all pass PyTorch/ONNX/ORT. CNN and TCN4/8 compile and execute in nncase 2.11.0 Simulator. GRU4/8 fail in the 2.11.0 importer (`VisitTranspose`, missing CheckedType). Logs and failed attempts retained. Smoke models have random weights and are **not competition weights**.
- Historical physical CanMV evidence identifies v1.8-36-gae2f20d / ae2f20da0467a8e70659f032b240aae50272c49a and `nn.version() = 2.11.0`: `docs/validation/camera-diagnosis-20261116/fixed600.json`, `gc1000.json`. Current physical readback: **NOT RUN**.
- [Smoke](data/smoke-a01/QA.json): 100 rendered frames / 10 trajectories. [Accepted Pilot input](data/pilot-a03/QA.json): 1,000 frames / 100 trajectories, 970 distinct sampled XY groups, split 600/200/200 frames. Camera XY is read from the same rendered camera's world translation, in mm. Images are 160x90 RGB. Metadata height=400mm/FOV=96 degrees are assumptions/manufacturer references, not calibration.
- a01 Pilot and a02 Pilot remain local diagnostic predecessors, not training inputs. a03 fixes actual appearance provenance, road/intersection paths and evaluation splits.
- CPU filter and dataset regression checks: `python -m unittest discover -s research/temporal_camera -p test_contracts.py -v`, 7 pass. Unknown speed defaults to soft full-support prior. Optional hard support requires explicit conservative camera bounds. LOST recovery is tested.

## Frozen Pilot and fair comparison

[Submission identity](cloud/pilot-a02/SUBMISSION_INTENT.json), [publication admission](cloud/pilot-a02/ADMISSION.json), [input member hashes](cloud/pilot-a02/INPUT_MANIFEST.json). Follow the original returned saved version; do not create a replacement because output is not yet available.

Kaggle input: `lotus30/k230-camera-xy-pilot-v1-20261117-a01/1`, private. Producer run: `k230-camera-xy-multiroute-pilot-20261117-a01`. Source Notebook: `lotus30/k230-camera-xy-multiroute-pilot-a01`.

Compare CNN1, GRU4, GRU8, TCN4, TCN8, each with and without the identical filter. Every setting trains/evaluates the same T8 endpoint IDs (300 total, 180 train / 60 val / 60 test). Seed117, AdamW, up to60 epochs each, patience12, total workload wall budget2400s. No local optimizer training. Only camera_x_mm/camera_y_mm supervise CE map cell + subcell XY residual. No yaw/robot/extrinsic head; no untrained quality head is exported. Validation selects best weights and temperature.

Main trajectory holdout has all ten scenarios in each split. Spatial challenge (50 frames /15 scored endpoints) and unseen reflection appearance challenge (50/15) are test-only, reported separately. Actual endpoint support is sparse: only26/60 val endpoints occupy cells positively supervised by train endpoints (43.33%; 11/30 unique val cells). This is a diagnostic Pilot limitation, not a localization success claim. Do not use challenge outputs to tune the model or filter.

All stress routes consume the same available observations: no dropped frame remains in neural history; blackout changes actual shared neural inputs and quality; filters warm up from frame0; kidnapped-camera stress changes image and GT without resetting the filter. Grid-only raw/filter metrics share the same100mm decoding; subcell-refined raw metrics are separate. These short10-frame clips qualify pipeline behavior, not long-run tracking.

## Commands and continuation

Host compilation Python: `C:/Users/28389/Desktop/k230worker-live/.ops/local/venvs/kpu-train-cpu/Scripts/python.exe`. Cloud control Python: `C:/Users/28389/Desktop/k230-vision-optimized/.ops/local/topface-env/Scripts/python.exe`. These local paths are convenience only; pinned environment/input records are the portable identity.

From repository root:

```powershell
python research/temporal_camera/cloud.py observe --record docs/temporal_camera/cloud/pilot-a02
python research/temporal_camera/cloud.py collect --record docs/temporal_camera/cloud/pilot-a02
python research/temporal_camera/cloud.py publish --record docs/temporal_camera/cloud/pilot-a02
```

Use the cloud-control Python for these commands. Collector binds exact original version and bounded compact results. Publisher reuses the explicitly authorized private K230 cloud-to-cloud archive route, original Drive folder and full byte SHA256 verification. No full training archive is relayed through the host. Named small deployment files may be collected separately.

## Remaining acceptance

Training results pending actual saved execution.10k–20k expansion: **NOT RUN**, blocked on Pilot qualification, not automatically admitted by training completion. Final trained KModel/parity, native full OV2735→preprocessing→KPU→CPU filter loop, memory pools, coexistence budget with RoadNet/TopFaceNet and long-run stability: **NOT RUN**. Real optical-center camera XY ground truth, intrinsics, camera speed bound and installation offset: **UNKNOWN**. Mechanical offset remains outside the model and is not required for camera-only inference.

Competition selection remains provisional: A is the required fallback architecture, C is the currently compile-qualified temporal candidate, B remains a host comparison until its compiler issue is resolved. No route is claimed competition-ready before accuracy and physical acceptance.

## Actual first Pilot outcome and revision

Original saved version145509345/v1 trained CNN16 epochs/96updates, selected epoch4, then stopped at missing cloud `onnxruntime`. Its original1,411,552-byte archive is accepted on Drive with full-byte SHA256 and temporary-Secret cleanup. CNN checkpoint, ONNX and history were recovered by original saved member hashes, without replaying optimization.

[Recovery a02](cloud/pilot-recovery-a02/compact/PILOT_GATE.json) finished all five settings on TeslaT4, Python3.13.16 / torch2.11.0+cu130 / ORT1.24.4. Original CNN training reused; GRU4/8 and TCN4/8 completed. All five models' main test mean errors are around644–645mm and P95 around1247–1254mm: **NEEDS_IMPROVEMENT**, no full-scale admission. These are actual poor results, not placeholders. The original CNN also compiled with real train-only calibration and ran all60 test endpoints in Simulator; [report](trained-cnn-pilot-a01/KMODEL_REPORT.json). Quantization kept every grid argmax; it did not cause the collapse.

Independent CPU forward diagnosis found both best and last CNN predict only class324, the training majority class. Training CE4.0355 is near label-prior entropy4.0293; most input feature variation is lost before GAP. This supports a bounded architecture repair, not merely more epochs.

[Spatial revision a03](cloud/spatial-pilot-a03/SUBMISSION_INTENT.json) uses Conv/BN/ReLU,3x5 spatial pooling and all600/200/200 causal endpoints (repeat the first frame only during initial warmup). The five settings remain fair within this revision. A separate32-sample train-only learning gate performs160 updates and discards its weights; formal training is admitted only if accuracy>=80% and CE<0.7. [Recompiled spatial model smoke](g1-spatial-v2-a01/SUMMARY.json): CNN/TCN pass, GRU importer failure retained. Do not mix a03 metrics with the old180-training-endpoint experiment.

User steering on2026-11-17 explicitly defers board connection: continue cloud/offline delivery. Native board tests stay NOT RUN; do not repeatedly request board access during this phase.

The reused scene's gray outer-region interpretation differs from one reading of the official figure. Keep its exact map hash and label this simulation scope; do not rebuild the map or call it a calibrated physical replica.
