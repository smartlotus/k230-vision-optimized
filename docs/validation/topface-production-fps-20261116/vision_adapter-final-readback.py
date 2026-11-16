"""Convert CanMV image results into the frozen Observation contract."""
from vision.common import observation, pixels, setting, finite
from vision.qr import QRTracker
from vision.tracker import StableTracker
from vision.color import select_candidate
from vision.ring import cluster_circles, identify_ring
from vision.lane import estimate_lane
from vision.mapping import apply_calibration
from .camera_roles import MODE_CAMERA


class VisionAdapter:
    def __init__(self, config=None, topface_backend=None):
        if config is None:
            from . import config as defaults
            config = defaults
        self.config = config
        self.qr = QRTracker(setting(config, "QR_CONFIRMATIONS", 3), setting(config, "QR_TIMEOUT_MS", 5000),
                            setting(config, "QR_MAX_GAP_MS", 500))
        self.tracker = StableTracker(setting(config, "STABLE_FRAMES", 3),
            setting(config, "STABLE_MAX_MOTION_PX", 4), setting(config, "STABLE_MAX_GAP_MS", 300))
        self.context = None
        self.excluded = {}
        self.blocks = {}
        self.last_block_candidates = []
        self.last_top_circles = []
        self.topface_backend=topface_backend
        self.last_topface=None

    def reset_session(self):
        self.qr.reset_session()
        self.tracker.reset()
        self.excluded = {}
        self.context = None
        self.blocks = {}
        self.last_block_candidates = []
        self.last_topface=None

    def mark_consumed(self, station, target_id, u_px, v_px, radius_px):
        """Call only after a confirmed physical pickup, not after command acceptance."""
        self.excluded.setdefault((station, target_id), []).append((u_px, v_px, radius_px))
        self.tracker.reset()

    def clear_consumed(self, station, target_id=None):
        """Explicitly begin a newly confirmed physical scene/batch."""
        for key in list(self.excluded):
            if key[0] == station and (target_id is None or key[1] == target_id):
                del self.excluded[key]
        self.tracker.reset()

    def _camera_id(self, mode):
        camera_id = setting(self.config, "MODE_CAMERA", MODE_CAMERA).get(mode)
        ids = tuple(setting(self.config, "CAMERA_IDS", (2, 1)))
        if camera_id not in ids and setting(self.config, "DESKTOP_SINGLE_CAMERA", False) and len(ids) == 1:
            return ids[0]
        return camera_id

    def detect(self, mode, frame, target_id=None, station=None):
        self.last_block_candidates = []
        self.last_top_circles = []
        result = observation(mode, frame if isinstance(frame, dict) else None, target_id, station)
        result["stale_after_ms"] = setting(self.config, "STALE_AFTER_MS", 250)
        if mode not in ("QR", "COLOR", "RING", "LANE", "TOP"):
            result.update({"status": "ERROR", "reason": "INVALID_MODE"})
            self.tracker.reset()
            return result
        if not isinstance(frame, dict) or frame.get("image") is None:
            result.update({"status": "UNAVAILABLE", "reason": "NO_FRAME"})
            self.tracker.reset()
            self.qr.reset_scan()
            return result
        image = frame["image"]
        if (type(frame.get("camera_id")) is not int or frame.get("camera_id") != self._camera_id(mode)
                or type(frame.get("frame_id")) is not int or frame["frame_id"] < 1
                or type(frame.get("timestamp_ms")) is not int or frame["timestamp_ms"] < 0
                or type(frame.get("is_simulated")) is not bool):
            result.update({"status": "ERROR", "reason": "FRAME_SOURCE_MISMATCH"})
            self.tracker.reset()
            self.qr.reset_scan()
            return result
        profile = frame.get("image_profile")
        profile_keys = ("width", "height", "crop", "hmirror", "vflip", "rotation", "pose_id")
        try:
            actual_size = (image.width(), image.height())
        except Exception:
            actual_size = None
        if (not isinstance(profile, dict) or any(k not in profile for k in profile_keys)
                or (profile.get("width"), profile.get("height")) != actual_size
                or any(type(profile[k]) is not int or profile[k] <= 0 for k in ("width", "height"))
                or any(type(profile[k]) is not bool for k in ("crop", "hmirror", "vflip"))
                or type(profile["rotation"]) is not int or profile["rotation"] not in (0, 90, 180, 270)
                or (frame.get("width", actual_size[0]), frame.get("height", actual_size[1])) != actual_size):
            result.update({"status": "ERROR", "reason": "FRAME_PROFILE_MISMATCH"})
            self.tracker.reset()
            self.qr.reset_scan()
            return result
        # Copy a scalar signature rather than retaining a caller-mutable dict.
        context = (mode, frame["camera_id"], station, target_id, tuple((key, repr(profile[key])) for key in sorted(profile)))
        if context != self.context:
            self.tracker.reset()
            self.qr.reset_scan()
            self.context = context
        try:
            if mode == "QR":
                # The cached QR example uses grayscale; do not rely on RGB565 decoding.
                gray = image.to_grayscale()
                kwargs = {} if setting(self.config, "QR_ROI") is None else {"roi": setting(self.config, "QR_ROI")}
                codes = [code.payload() for code in gray.find_qrcodes(**kwargs)]
                update = self.qr.update(codes, frame["frame_id"], frame["timestamp_ms"], context)
                result.update({"status": update["status"], "reason": update["reason"],
                               "raw_payload": update["payload"], "stable": update["submitted"],
                               "score_hint": 100 if update["submitted"] else 0})
                return result
            if mode == 'TOP' and self.topface_backend is not None:
                result=self._topface(result,image,target_id,station)
            elif mode in ("COLOR", "TOP"):
                result = self._color(result, image, target_id, station, mode)
            elif mode == "RING":
                result = self._ring(result, image, target_id, station, profile)
            else:
                return self._lane(result, image)
            if result["u_px"] is not None and result["status"] == "OK":
                result["stable"] = self.tracker.update(context, frame["frame_id"], frame["timestamp_ms"],
                                                        (result["u_px"], result["v_px"]))
                planes = setting(self.config, "PLANE_BY_MODE_STATION", {})
                plane = planes.get((mode, station))
                calibration = setting(self.config, "CALIBRATIONS", {}).get((station, plane))
                if plane is None:
                    result.update({"status": "UNCALIBRATED", "reason": "TARGET_PLANE_REQUIRED"})
                else:
                    result = apply_calibration(result, calibration, plane)
            else:
                self.tracker.reset()
            return result
        except Exception as error:
            self.tracker.reset()
            self.qr.reset_scan()
            result.update({"status": "ERROR", "reason": "DETECTOR_ERROR: " + str(error), "stable": False,
                           "x_mm": None, "y_mm": None, "calibration_id": None, "target_pose_valid": False})
            return result

    def close(self):
        if self.topface_backend is not None:self.topface_backend.close()

    def _topface(self,result,image,target_id,station):
        if type(target_id) is not int or target_id not in (1,2,3,4,5,6):
            result.update(status='ERROR',reason='COLOR_ID_REQUIRED');return result
        decoded=self.topface_backend.infer(image);self.last_topface=decoded
        result.update(detector_backend='TOPFACE_KPU',model_sha256=self.topface_backend.identity['kmodel_sha256'],
                      topface_detections=decoded['detections'],capacity_state=decoded['capacity_state'])
        if decoded['capacity_state'] in ('OVER_CAPACITY','UNCERTAIN'):
            result.update(status='UNAVAILABLE',reason='TOPFACE_'+decoded['capacity_state']);return result
        cfg=setting(self.config,'TOP_CONFIG',{}).get(station)
        if not cfg or cfg.get('roi') is None or cfg.get('reference') is None:
            result.update(status='UNCALIBRATED',reason='TOP_PARAMETERS_REQUIRED');return result
        roi=cfg['roi'];ref=cfg['reference'];eligible=[]
        for target in decoded['detections']:
            if target['color_id']!=target_id or target['boundary_candidate']:continue
            if target['center_valid_probability']<.5 or target['center_visible_probability']<.5:continue
            if target['ellipse'] is None or target['ellipse']['geometry_confidence']<.5:continue
            x1,y1,x2,y2=target['bbox_xyxy_visible'];cx,cy=target['center_xy_projected_3d']
            if not all(finite(v) for v in (x1,y1,x2,y2,cx,cy)):continue
            if not (roi[0]<x1<x2<roi[0]+roi[2] and roi[1]<y1<y2<roi[1]+roi[3]):continue
            if not (roi[0]<=cx<roi[0]+roi[2] and roi[1]<=cy<roi[1]+roi[3]):continue
            area=(x2-x1)*(y2-y1)
            if not cfg.get('min_area',0)<=area<=cfg.get('max_area',640*360):continue
            if any((cx-p[0])**2+(cy-p[1])**2<=p[2]**2 for p in self.excluded.get((station,target_id),[])):continue
            eligible.append(((cx-ref[0])**2+(cy-ref[1])**2,target))
        if not eligible:
            result.update(status='NOT_FOUND',reason='NO_VISIBLE_COMPLETE_TOPFACE');return result
        eligible.sort(key=lambda item:item[0]);chosen=eligible[0][1]
        x1,y1,x2,y2=chosen['bbox_xyxy_visible']
        result.update(status='OK',bbox=[x1,y1,x2-x1,y2-y1],ellipse=chosen['ellipse'],
                      model_confidence=chosen['score'],reason=None)
        return pixels(result,*chosen['center_xy_projected_3d'],score=round(chosen['score']*100))

    def _color(self, result, image, target_id, station, mode):
        if type(target_id) is not int or target_id not in (1, 2, 3, 4, 5, 6):
            result.update({"status": "ERROR", "reason": "COLOR_ID_REQUIRED"})
            return result
        configs = setting(self.config, "TOP_CONFIG" if mode == "TOP" else "COLOR_CONFIG", {})
        if mode == "TOP" and station not in configs:
            result.update({"status": "UNCALIBRATED", "reason": "TOP_PARAMETERS_REQUIRED"})
            return result
        cfg = configs.get(station, {})
        block_parameters = cfg.get('block_parameters')
        if block_parameters is not None:
            if block_parameters['camera_id'] != result['camera_id']:
                result.update(status='ERROR',reason='BLOCK_CAMERA_MISMATCH')
                return result
            if block_parameters['thresholds'].get(target_id) is None:
                result.update(status='UNCALIBRATED',reason='COLOR_SAMPLE_REQUIRED')
                return result
            from vision.block_detector import BlockDetector
            key=(mode,station,block_parameters['profile_id'],repr(result['image_profile']))
            if key not in self.blocks:
                self.blocks={key:BlockDetector(block_parameters)}
            detector=self.blocks[key]
            candidates=detector.process(image,result['frame_id'],result['timestamp_ms'])
            self.last_block_candidates=[dict(c) for c in candidates]
            eligible=[c for c in candidates if c['eligible']]
            selected=select_candidate(eligible,target_id,cfg.get('roi'),cfg.get('reference'),
                cfg.get('min_area'),cfg.get('max_area'),self.excluded.get((station,target_id)))
            if selected is None:
                result.update(status='NOT_FOUND',reason='NO_COMPLETE_COLOR_TARGET')
                return result
            chosen=next(c for c in eligible if c['target_id']==target_id and c['bbox']==selected['bbox'])
            result.update(status='OK',bbox=chosen['bbox'],track_id=chosen['track_id'],
                color_profile_id=block_parameters['profile_id'],
                color_profile_sampling_verified=block_parameters.get('sampling_verified',True),
                color_profile_measurement_camera_id=block_parameters.get('measurement_camera_id',block_parameters['camera_id']))
            return pixels(result,*selected['center'],score=selected['score_hint'])
        thresholds = cfg.get("thresholds", setting(self.config, "COLOR_THRESHOLDS", {}))
        threshold = thresholds.get(target_id)
        roi = cfg.get("roi", setting(self.config, "COLOR_ROI"))
        reference = cfg.get("reference", setting(self.config, "COLOR_REFERENCE"))
        min_area = cfg.get("min_area", setting(self.config, "COLOR_MIN_AREA"))
        max_area = cfg.get("max_area", setting(self.config, "COLOR_MAX_AREA"))
        if any(v is None for v in (threshold, roi, reference, min_area, max_area)):
            result.update({"status": "UNCALIBRATED", "reason": "COLOR_PARAMETERS_REQUIRED"})
            return result
        if mode=='TOP' and cfg.get('surface_shape') is not None:
            if cfg.get('camera_id')!=result['camera_id'] or cfg.get('image_profile')!=result['image_profile']:
                result.update(status='UNCALIBRATED',reason='TOP_PROFILE_BINDING_MISMATCH')
                return result
        if mode=='TOP':
            from vision.top import validate_circle_config
            validate_circle_config(cfg)
        circle_geometry=mode=='TOP' and target_id in cfg.get('circle_refine_ids',())
        if circle_geometry:
            import image as native_image
            from vision.top import circle_surfaces,validate_circle_parameters,circle_boundary_support
            parameters=validate_circle_parameters(cfg.get('circle_parameters'))
            circles=[]
            width,height=image.width(),image.height()
            rx,ry,rw,rh=roi
            for circle in image.find_circles(roi=roi,**parameters):
                x,y,radius=circle.x(),circle.y(),circle.r()
                if any(type(v) is not int for v in (x,y,radius)):continue
                if not parameters['r_min']<=radius<=parameters['r_max']:continue
                left,top=x-radius,y-radius
                right,bottom=x+radius,y+radius
                if not (0<left and 0<top and right<width and bottom<height):continue
                if not (rx<left and ry<top and right<rx+rw and bottom<ry+rh):continue
                if not min_area<=4*radius*radius<=max_area:continue
                step=max(2,int(radius*.4))
                if not (step<=x<image.width()-step and step<=y<image.height()-step):continue
                samples=[native_image.rgb_to_lab(*image.get_pixel(x+dx,y+dy)) for dy in (-step,0,step) for dx in (-step,0,step)]
                medians=[sorted(s[axis] for s in samples)[len(samples)//2] for axis in (0,1,2)]
                matched=sum(all(threshold[axis*2]<=sample[axis]<=threshold[axis*2+1] for axis in (0,1,2)) for sample in samples)
                verified=matched>=7 and all(threshold[axis*2]<=medians[axis]<=threshold[axis*2+1] for axis in (0,1,2))
                boundary=None
                if verified and cfg.get('circle_boundary') is not None:
                    boundary=circle_boundary_support(image,native_image.rgb_to_lab,x,y,radius,cfg['circle_boundary'])
                    verified=boundary['verified']
                if len(self.last_top_circles)<40:
                    self.last_top_circles.append({'x':x,'y':y,'r':radius,'matched':matched,
                        'medians':medians,'boundary':boundary,'verified':verified})
                circles.append({'x':x,'y':y,'r':radius,'interior_verified':verified})
            candidates=circle_surfaces(circles,target_id,cfg.get('circle_center_tolerance_px',6))
        else:
            blobs=image.find_blobs([threshold], roi=roi, area_threshold=int(min_area), pixels_threshold=1, merge=False)
        if not circle_geometry and mode=='TOP' and cfg.get('surface_shape') is not None:
            from vision.top import filter_surfaces
            candidates=filter_surfaces([{'bbox':blob.rect(),'pixels':blob.pixels(),'target_id':target_id} for blob in blobs],cfg['surface_shape'])
        elif not circle_geometry:
            candidates=[{'bbox':blob.rect(),'target_id':target_id} for blob in blobs]
        selected = select_candidate(candidates, target_id, roi, reference, min_area, max_area,
                                    self.excluded.get((station, target_id)))
        if selected is None:
            result.update({"status": "NOT_FOUND", "reason": "NO_COMPLETE_COLOR_TARGET"})
            return result
        result.update({"status": "OK", "bbox": selected["bbox"]})
        if circle_geometry:result['top_geometry']='LAB_INTERIOR_CONFIRMED_CIRCLE'
        return pixels(result, selected["center"][0], selected["center"][1], selected["score_hint"])

    def _ring(self, result, image, target_id, station, profile):
        if type(target_id) is not int or target_id not in (1, 2, 3):
            result.update({"status": "ERROR", "reason": "RING_ID_REQUIRED"})
            return result
        cfg = setting(self.config, "RING_CONFIG", {}).get(station)
        if cfg is None or cfg.get("roi") is None:
            result.update({"status": "UNCALIBRATED", "target_id": None, "reason": "RING_PARAMETERS_REQUIRED"})
            return result
        kwargs = dict(setting(self.config, "RING_PARAMETERS", {}))
        kwargs.update(cfg.get("parameters", {}))
        kwargs["roi"] = cfg["roi"]
        circles = [{"x": c.x(), "y": c.y(), "r": c.r()} for c in image.find_circles(**kwargs)]
        groups = cluster_circles(circles, cfg.get('center_tolerance_px',setting(self.config, "RING_CENTER_TOLERANCE_PX", 8)))
        if not groups:
            result.update({"status": "NOT_FOUND", "reason": "NO_RING"})
            return result
        # Ring identity is itself calibrated to the exact observation pose/profile.
        bound_cfg = cfg if (cfg.get("camera_id") == result["camera_id"] and cfg.get("image_profile") == profile) else None
        selected, reason = identify_ring(groups, target_id, bound_cfg)
        if selected is None:
            result.update({"status": "UNCALIBRATED", "target_id": None, "reason": reason,
                           "ring_candidates": groups})
            if len(groups) == 1:
                pixels(result, groups[0]["center"][0], groups[0]["center"][1])
            return result
        result.update({"status": "OK", "ring_support": selected["support"]})
        if "selection_method" in selected:
            result["ring_selection_method"] = selected["selection_method"]
            result["ring_reference_distance_px"] = selected["reference_distance_px"]
        return pixels(result, selected["center"][0], selected["center"][1], min(100, 25*selected["support"]))

    def _lane(self, result, image):
        roi = setting(self.config, "LANE_ROI")
        threshold = setting(self.config, "LANE_THRESHOLD")
        min_width = setting(self.config, "LANE_MIN_WIDTH_PX")
        max_width = setting(self.config, "LANE_MAX_WIDTH_PX")
        if any(v is None for v in (roi, threshold, min_width, max_width)):
            result.update({"status": "UNCALIBRATED", "reason": "LANE_PARAMETERS_REQUIRED"})
            return result
        gray = image.to_grayscale()
        count = max(3, setting(self.config, "LANE_SCAN_ROWS", 3))
        step = max(1, setting(self.config, "LANE_SAMPLE_STEP", 2))
        sampled_y=[roi[1]+int((i+1)*roi[3]/(count+1)) for i in range(count)]
        if len(set(sampled_y))<3:
            result.update(status='UNCALIBRATED',reason='LANE_DISTINCT_ROWS_REQUIRED')
            return result
        rows = []
        for y in sampled_y:
            rows.append({"x": roi[0], "step": step, "values": [gray.get_pixel(x, y) for x in range(roi[0], roi[0]+roi[2], step)]})
        selected = estimate_lane(rows, threshold, min_width/step, max_width/step,
                                 setting(self.config, "LANE_MAX_CENTER_SPREAD_PX", 20))
        if selected is None:
            result.update({"status": "NOT_FOUND", "reason": "LANE_BOUNDARIES_UNCERTAIN"})
            return result
        result.update({"status": "OK", "offset_px": selected["center_x"]-image.width()/2,
                       "auxiliary_only": True, "lane_bounds": selected["bounds"]})
        return pixels(result, selected["center_x"], roi[1]+roi[3]/2, selected["score_hint"])
