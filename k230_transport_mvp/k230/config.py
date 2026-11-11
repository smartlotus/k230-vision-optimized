"""LIVE configuration. None means a measurement is still required.

Phone photographs are references, never automatically installed calibration.
DESKTOP_SINGLE_CAMERA is an explicit bench mode, not a competition installation.
"""
from .camera_roles import OBJECT_CAMERA_ID,FRONT_CAMERA_ID,MODE_CAMERA
CAMERA_IDS = (OBJECT_CAMERA_ID,FRONT_CAMERA_ID)
CAMERA_PRIORITY = (OBJECT_CAMERA_ID,FRONT_CAMERA_ID,0)
CAMERA_INIT_ORDER = (2,1,0)
CAMERA_QUARANTINE = (0,)  # CSI0 detection failed repeatedly; root cause unresolved.
# Clearing quarantine requires explicit diagnostic capture and fresh acceptance.
DESKTOP_SINGLE_CAMERA = False
WIDTH = 640
HEIGHT = 480
SENSOR_WIDTH = 1920
SENSOR_HEIGHT = 1080
SENSOR_FPS = 30
CROP = False
HMIRROR = False
VFLIP = False
ROTATION = 0
POSE_ID = None
PREVIEW = False
DEFAULT_MODE = "QR"
DEFAULT_TARGET_ID = None
DEFAULT_STATION = None
# Each CSI may wait at most 2 seconds until its first successful frame.
FIRST_FRAME_TIMEOUT_MS = 2000
SNAPSHOT_TIMEOUT_MS = 100
UART_ENABLED = False
UART_VERIFIED = False
UART = {"enabled": False, "pins_verified": False, "tx_pin": None,
        "rx_pin": None, "uart_id": 2, "baudrate": 115200}

QR_ROI = None
QR_CONFIRMATIONS = 3
QR_TIMEOUT_MS = 5000
QR_MAX_GAP_MS = 500
STABLE_FRAMES = 3
STABLE_MAX_MOTION_PX = 4
STABLE_MAX_GAP_MS = 300
STALE_AFTER_MS = 250

# Per-station entries can override these defaults with measured settings.
COLOR_THRESHOLDS = {1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
COLOR_ROI = None
COLOR_REFERENCE = None
COLOR_MIN_AREA = None
COLOR_MAX_AREA = None
COLOR_CONFIG = {}
TOP_CONFIG = {}
RING_CONFIG = {}
# Optional per-station center_gates requires measured pixel centers, nonoverlapping
# radius_px regions, min_support and min_distance_gap_px. Never copy fixture geometry.
RING_CENTER_TOLERANCE_PX = 8
RING_PARAMETERS = {"threshold": 2000, "x_margin": 10, "y_margin": 10,
                   "r_margin": 10, "r_min": 2, "r_max": 100, "r_step": 2}
LANE_ROI = None
LANE_THRESHOLD = None
LANE_MIN_WIDTH_PX = None
LANE_MAX_WIDTH_PX = None
LANE_SCAN_ROWS = 3
LANE_SAMPLE_STEP = 2
LANE_MAX_CENTER_SPREAD_PX = 20

# Calibration keys are (station, plane), each with exact runtime image_profile.
CALIBRATIONS = {}
WORKSPACE_MM = None
TOP_HEIGHT_MM = None
PLANE_BY_MODE_STATION = {("COLOR", "RAW"): "RAW_TOP",
                         ("COLOR", "PROCESS"): "PROCESS_TOP",
                         ("COLOR", "BUFFER"): "BUFFER_TOP",
                         ("RING", "PROCESS"): "PROCESS_PLANE",
                         ("RING", "BUFFER"): "BUFFER_PLANE",
                         ("TOP", "BUFFER"): "BUFFER_TOP"}
