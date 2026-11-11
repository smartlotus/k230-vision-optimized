"""Opt-in block tracking inside the unchanged observation contract."""
from vision.common import setting
from vision.block_detector_optimized import BlockDetector
from .vision_adapter import VisionAdapter as LegacyVisionAdapter


class VisionAdapter(LegacyVisionAdapter):
    def _color(self, result, image, target_id, station, mode):
        # Seed only the exact detector cache used by the inherited block path.
        # Other modes, source gates, calibration and candidate selection stay there.
        if type(target_id) is int and target_id in (1, 2, 3, 4, 5, 6):
            configs = setting(self.config, 'TOP_CONFIG' if mode == 'TOP' else 'COLOR_CONFIG', {})
            parameters = configs.get(station, {}).get('block_parameters')
            if (parameters is not None and parameters['camera_id'] == result['camera_id']
                    and parameters['thresholds'].get(target_id) is not None):
                key = (mode, station, parameters['profile_id'], repr(result['image_profile']))
                if key not in self.blocks:
                    self.blocks = {key: BlockDetector(parameters)}
        return super()._color(result, image, target_id, station, mode)
