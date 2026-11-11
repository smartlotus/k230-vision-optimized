"""Explicit service combining opt-in block tracking and UART buffering."""
from .main import run_loop, _setting


def main(config_override=None):
    import sys
    if sys.implementation.name != 'micropython':
        raise SystemExit('Use CanMV on K230; PC simulation: python -m sim.run_mission.')
    import os
    import time
    from . import config
    from .camera_manager import CameraManager
    from .vision_adapter_optimized import VisionAdapter
    from .uart_adapter_optimized import UartAdapter
    config = config_override if config_override is not None else config
    vision = VisionAdapter(config)
    uart = UartAdapter(_setting(config, 'UART'))
    try:
        camera = CameraManager(config)
    except BaseException:
        uart.close()
        raise
    return run_loop(camera, vision, uart, config,
                    exitpoint=os.exitpoint, sleep_ms=time.sleep_ms)


if __name__ == '__main__':
    main()
