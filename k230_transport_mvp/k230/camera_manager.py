"""Staged frame acquisition. No board is accessed by importing this module."""
from vision.common import now_ms, setting


class CameraManager:
    def __init__(self, config=None, sensor_factory=None, display=None, media=None, clock_ms=None):
        if config is None:
            from . import config as defaults
            config = defaults
        self.config = config
        self.sensor_factory = sensor_factory
        self.display = display
        self.media = media
        self.clock_ms = clock_ms or now_ms
        self.sensors = {}
        self.frame_id = 0
        self.started = False
        self.media_started = False
        self.display_started = False
        self.close_errors = []
        self.successful_cameras = set()

    def camera_for_mode(self, mode):
        camera_id = setting(self.config, "MODE_CAMERA", {}).get(mode)
        if camera_id is None:
            return None
        ids = tuple(setting(self.config, "CAMERA_IDS", (2,)))
        if camera_id not in ids and setting(self.config, "DESKTOP_SINGLE_CAMERA", False) and len(ids) == 1:
            return ids[0]
        return camera_id

    def start(self):
        if self.started:
            return self
        ids = tuple(setting(self.config, "CAMERA_IDS", (2,)))
        if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or i not in (0, 1, 2) for i in ids):
            raise ValueError("CAMERA_IDS must contain distinct CSI IDs")
        quarantine = tuple(setting(self.config, "CAMERA_QUARANTINE", ()))
        if any(i in quarantine for i in ids):
            raise ValueError("Quarantined camera requested; use explicit diagnostic configuration")
        priority = tuple(setting(self.config, "CAMERA_PRIORITY", (2, 1, 0)))
        if len(priority) != 3 or set(priority) != {0, 1, 2}:
            raise ValueError("CAMERA_PRIORITY must contain each CSI ID once")
        initialization=tuple(setting(self.config,'CAMERA_INIT_ORDER',priority))
        if len(initialization)!=3 or set(initialization)!={0,1,2}:
            raise ValueError('CAMERA_INIT_ORDER must contain each CSI ID once')
        ids = tuple(i for i in initialization if i in ids)
        if setting(self.config, "ROTATION", 0) != 0:
            raise ValueError("rotation requires an explicit verified image transform")
        for key, default in (("FIRST_FRAME_TIMEOUT_MS", 2000), ("SNAPSHOT_TIMEOUT_MS", 100)):
            timeout = setting(self.config, key, default)
            if type(timeout) is not int or not 1 <= timeout <= 2000:
                raise ValueError(key + " must be an integer in 1..2000 milliseconds")
        if self.sensor_factory is None:
            from media.sensor import Sensor
            self.sensor_factory = Sensor
        if self.media is None:
            from media.media import MediaManager
            self.media = MediaManager
        if setting(self.config, "PREVIEW", False) and self.display is None:
            from media.display import Display
            self.display = Display
        try:
            for camera_id in ids:
                sensor = self.sensor_factory(id=camera_id,
                    width=setting(self.config, "SENSOR_WIDTH", 1920),
                    height=setting(self.config, "SENSOR_HEIGHT", 1080),
                    fps=setting(self.config, "SENSOR_FPS", 30))
                # Register before reset so a reset failure still releases this sensor.
                self.sensors[camera_id] = sensor
                sensor.reset()
                sensor.set_framesize(width=setting(self.config, "WIDTH", 640),
                    height=setting(self.config, "HEIGHT", 480), crop=setting(self.config, "CROP", False))
                sensor.set_pixformat(self.sensor_factory.RGB565)
                if camera_id==2 and setting(self.config,'TOPFACE_MODEL_ROOT'):
                    sensor.set_framesize(width=640,height=360,crop=False,chn=2)
                    sensor.set_pixformat(self.sensor_factory.RGBP888,chn=2)
                sensor.set_hmirror(setting(self.config, "HMIRROR", False))
                sensor.set_vflip(setting(self.config, "VFLIP", False))
            if setting(self.config, "PREVIEW", False):
                self.display_started = True
                self.display.init(self.display.VIRT, width=setting(self.config, "WIDTH", 640),
                                  height=setting(self.config, "HEIGHT", 480), to_ide=True)
            self.media_started = True
            self.media.init()
            # The cached multi-sensor example starts the shared capture once.
            self.sensors[ids[0]].run()
            self.started = True
            return self
        except Exception:
            self.close()
            raise

    def get_frame(self, camera_id, mode=None):
        if not self.started:
            return None
        if camera_id not in (0, 1, 2):
            return None
        if camera_id not in self.sensors:
            ids = tuple(self.sensors)
            mode_sources = setting(self.config, "MODE_CAMERA", {}).values()
            if setting(self.config, "DESKTOP_SINGLE_CAMERA", False) and len(ids) == 1 and camera_id in mode_sources:
                camera_id = ids[0]
            else:
                return None
        timeout_key = "SNAPSHOT_TIMEOUT_MS" if camera_id in self.successful_cameras else "FIRST_FRAME_TIMEOUT_MS"
        timeout_default = 100 if camera_id in self.successful_cameras else 2000
        # Give sensor startup a bounded first-frame window; never enable driver reboot.
        kwargs={'timeout':setting(self.config,timeout_key,timeout_default),'auto_reboot':False}
        if camera_id==2 and mode=='TOP' and setting(self.config,'TOPFACE_MODEL_ROOT'):kwargs['chn']=2
        image = self.sensors[camera_id].snapshot(**kwargs)
        if image is None:
            return None
        width, height = image.width(), image.height()
        if (width, height) != (setting(self.config, "WIDTH", 640), setting(self.config, "HEIGHT", 480)):
            raise RuntimeError("actual frame dimensions differ from configured image profile")
        self.successful_cameras.add(camera_id)
        self.frame_id += 1
        profile = {"width": width, "height": height, "crop": setting(self.config, "CROP", False),
                   "hmirror": setting(self.config, "HMIRROR", False), "vflip": setting(self.config, "VFLIP", False),
                   "rotation": 0, "pose_id": setting(self.config, "POSE_ID")}
        return {"image": image, "camera_id": camera_id, "frame_id": self.frame_id,
                "timestamp_ms": self.clock_ms(), "width": width, "height": height,
                "image_profile": profile, "is_simulated": False}

    def show_frame(self, frame):
        """Optional raw preview; do not rewrite frame identity or image geometry."""
        if setting(self.config, "PREVIEW", False) and self.display_started and frame and frame.get("image") is not None:
            self.display.show_image(frame["image"])

    def close(self):
        self.close_errors = []
        for sensor in self.sensors.values():
            try:
                sensor.stop()
            except Exception as error:
                self.close_errors.append(str(error))
        self.sensors = {}
        self.successful_cameras = set()
        if self.display_started:
            try:
                self.display.deinit()
            except Exception as error:
                self.close_errors.append(str(error))
        if self.media_started:
            try:
                self.media.deinit()
            except Exception as error:
                self.close_errors.append(str(error))
        self.started = self.display_started = self.media_started = False
