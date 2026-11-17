# D reserve: uncertainty fusion

Reviewed the [original KFNet repository](https://github.com/zlthinker/KFNet) on2026-11-17. It regresses scene coordinates, models optical-flow transitions and predicts measurement/process uncertainties; its documented tested stack is Python2.7/TensorFlow1.10–1.13. No original code or full6DoF/PnP system is transplanted here.

Engineering inference for this project: keep a multimodal camera-XY belief. Calibrate visual likelihood temperature on validation trajectories, increase process uncertainty with elapsed time and observed visual degradation, and retain a global probability floor when camera optical-center velocity is unknown. A Gaussian mean/covariance alone can merge two repeated-road modes into a physically wrong midpoint, so do not replace the grid simply because Kalman fusion is familiar.

The current soft grid filter implements the deterministic measurement/prior combination and global support, with configurable uncalibrated memory time. A learned process covariance, optical flow and a D-specific training run are **NOT RUN**. Admit a bounded D experiment only if A/B/C results expose unresolved ambiguity that this addition can address. Targets remain camera XY only.
