# K230 competition CameraXY temporal V1

[Offline delivery and capture inputs](OFFLINE_HANDOVER.md) maps requirements to evidence. [Offline replay](OFFLINE_REPLAY.md) runs timestamped RGB or video through the delivered models. [K230 native report](native-smoke-20261117/NATIVE_REPORT.md) records the a06 CNN1 and CPU filter USB smoke, including 100 processed frames and repeated MMZ cleanup. OV2735 positioning accuracy, native temporal routes and sustained board acceptance remain **NOT RUN**.

Current map revision follows the provided official PDF Figure 1 and the user's two-start procedure. [Corrected scene, Smoke/Pilot manifests, and updated evaluation contract](map-correction/README.md). Earlier runs retain their original map and must be reported under that map identity.

The task branch is `ops/temporal-camera-v1-20261117`. Follow the [OPS project entry](../../.ops/PROJECT.md) and [protocol](../../.ops/PROTOCOL.md). Select `temporal-camera-v1-20261117` for offline work or `temporal-native-smoke-20261117` for the native smoke evidence, then read the handoff listed in OPS status.

## Verified starting state

- [G0 package audit](AUDIT_READOUT.md), [full archive inventory](AUDIT_PACKAGE_INVENTORY.json): original ZIP and all embedded packages extracted; 13 manifests, 169 hashes pass. Reuse the advanced RoadNet scene, not the older inconsistent starter.
- [Tiny operator smoke](g1-smoke-a01/SUMMARY.json), [actual Pilot input resolution smoke](g1-input90x160-a01/SUMMARY.json): CNN, GRU4/8, TCN4/8 all pass PyTorch/ONNX/ORT. CNN and TCN4/8 compile and execute in nncase 2.11.0 Simulator. GRU4/8 fail in the 2.11.0 importer (`VisitTranspose`, missing CheckedType). Logs and failed attempts retained. Smoke models have random weights and are **not competition weights**.
- Historical physical CanMV evidence identifies v1.8-36-gae2f20d / ae2f20da0467a8e70659f032b240aae50272c49a and `nn.version() = 2.11.0`: `docs/validation/camera-diagnosis-20261116/fixed600.json`, `gc1000.json`. Current physical readback also confirmed both versions in the [native smoke](native-smoke-20261117/README.md).
- [Smoke](data/smoke-a01/QA.json): 100 rendered frames / 10 trajectories. [Accepted Pilot input](data/pilot-a03/QA.json): 1,000 frames / 100 trajectories, 970 distinct sampled XY groups, split 600/200/200 frames. Camera XY is read from the same rendered camera's world translation, in mm. Images are 160x90 RGB. Metadata height=400mm/FOV=96 degrees are assumptions/manufacturer references, not calibration.
- a01 Pilot and a02 Pilot remain local diagnostic predecessors, not training inputs. a03 fixes actual appearance provenance, road/intersection paths and evaluation splits.
- CPU filter and dataset regression checks: `python -m unittest discover -s research/temporal_camera -p test_contracts.py -v`, 7 pass. Unknown speed defaults to soft full-support prior. Optional hard support requires explicit conservative camera bounds. LOST recovery is tested.

## Original Pilot and fair comparison

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

The [sensor-aligned a08 Pilot](cloud/gray-map-pilot-a08/REPORT.md) completed five Kaggle training settings and recorded `NEEDS_IMPROVEMENT`. Expansion to 10,000–20,000 positions is **NOT RUN**. CNN and TCN KModels were compiled and tested in Simulator. The [fixed-window GRU export](map-correction/GRU_FIXED_WINDOW.md) passed compilation and Simulator with an equivalent static recurrent computation.

A06 CNN1 plus CPU filtering is the experimental fallback with a K230 USB smoke. The temporal models still require native measurement. Competition selection needs camera XY accuracy and recovery evidence with real images, plus end-to-end latency, memory and stability tests. The [handover](OFFLINE_HANDOVER.md) tracks remaining acceptance.

## Actual first Pilot outcome and revision

Original saved version145509345/v1 trained CNN16 epochs/96updates, selected epoch4, then stopped at missing cloud `onnxruntime`. Its original1,411,552-byte archive is accepted on Drive with full-byte SHA256 and temporary-Secret cleanup. CNN checkpoint, ONNX and history were recovered by original saved member hashes, without replaying optimization.

[Recovery a02](cloud/pilot-recovery-a02/compact/PILOT_GATE.json) finished all five settings on TeslaT4, Python3.13.16 / torch2.11.0+cu130 / ORT1.24.4. Original CNN training reused; GRU4/8 and TCN4/8 completed. All five models' main test mean errors are around644–645mm and P95 around1247–1254mm: **NEEDS_IMPROVEMENT**, no full-scale admission. These are actual poor results, not placeholders. The original CNN also compiled with real train-only calibration and ran all60 test endpoints in Simulator; [report](trained-cnn-pilot-a01/KMODEL_REPORT.json). Quantization kept every grid argmax; it did not cause the collapse.

Independent CPU forward diagnosis found both best and last CNN predict only class324, the training majority class. Training CE4.0355 is near label-prior entropy4.0293; most input feature variation is lost before GAP. This supports a bounded architecture repair, not merely more epochs.

[Spatial revision a03](cloud/spatial-pilot-a03/SUBMISSION_INTENT.json) uses Conv/BN/ReLU,3x5 spatial pooling and all600/200/200 causal endpoints (repeat the first frame only during initial warmup). The five settings remain fair within this revision. A separate32-sample train-only learning gate performs160 updates and discards its weights; formal training is admitted only if accuracy>=80% and CE<0.7. [Recompiled spatial model smoke](g1-spatial-v2-a01/SUMMARY.json): CNN/TCN pass, GRU importer failure retained. Do not mix a03 metrics with the old180-training-endpoint experiment.

During the offline phase on 2026-11-17 the user deferred board access. The later [isolated USB smoke](native-smoke-20261117/NATIVE_REPORT.md) records the approved device run.

[Official gray-map correction](map-correction/README.md) records the map geometry and 1,100 rendered frames. [Offline model delivery](map-correction/OFFLINE_MODEL_DELIVERY.md) describes the frozen experimental A route, native bench and prior Kaggle comparisons. [Corrected-map a06 report](cloud/gray-map-pilot-a06/REPORT.md) records the actual five-setting training, the held-out comparisons, and the Simulator checks for the trained CNN1 and TCN4 KModels. The corrected Pilot finished with `NEEDS_IMPROVEMENT`. [Model archive identity](cloud/gray-map-pilot-a06/MODEL_DELIVERY_SHA256.json)

[Start-area observability and paired motion replay](map-correction/OBSERVABILITY_AND_MOTION.md) records the independent static mask check and the held-out CPU replay. The optional time-scaled prior is available in the isolated runtime.

[RGB projection follow-up a07](cloud/gray-map-pilot-a07/REPORT.md) finished on Tesla T4 with the same trajectories and all five original route heads. It finished with `NEEDS_IMPROVEMENT`. The a06 single-frame and TCN4 builds remain the two experimental benchmark models. The [isolated board procedure](map-correction/BOARD_MULTIROUTE.md) binds both models and documents fresh timers, timeout behavior, and report collection. [Model archive](cloud/gray-map-pilot-a07/MODEL_DELIVERY_SHA256.json) · [Board archive](map-correction/BOARD_DELIVERY_SHA256.json) · [Host integration witness](map-correction/BOARD_HOST_WITNESS.json)

[640×360 sensor and AI2D evidence](map-correction/AI2D_SENSOR.md) records five historical fixture comparisons, the effect on exact compiled models, and a newly rendered 100-frame Smoke with aligned preprocessing. The later USB/KPU smoke is linked above. Full-scale generation remains NOT RUN.

[a08 sensor-aligned Pilot](cloud/gray-map-pilot-a08/REPORT.md) rerendered the identical 1,000 poses and finished five-setting training on private Kaggle. Accuracy remained `NEEDS_IMPROVEMENT`. [Fixed-window GRU report](map-correction/GRU_FIXED_WINDOW.md) records successful GRU4 and GRU8 compilation using an equivalent static graph, as well as the residual quantization differences.
