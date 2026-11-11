"""Explicitly started K230 observation service. This module never drives motors."""
import gc
import json
import time


def _clock_ms():
    return time.ticks_ms() if hasattr(time, 'ticks_ms') else int(time.monotonic()*1000)


def _elapsed(now, before):
    return time.ticks_diff(now, before) if hasattr(time, 'ticks_diff') else now-before


def _setting(config, key, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def run_loop(camera, vision, uart, config, clock_ms=None, exitpoint=None,
             sleep_ms=None, emit=None, max_iterations=None):
    """Inject adapters for host tests; only main() constructs physical hardware."""
    clock_ms=clock_ms or _clock_ms
    exitpoint=exitpoint or (lambda: None)
    sleep_ms=sleep_ms or (lambda ms: time.sleep(ms/1000))
    emit=emit or (lambda value: print(json.dumps(value)))
    modes=('QR','COLOR','RING','LANE','TOP')
    mode=_setting(config,'DEFAULT_MODE','QR')
    if mode not in modes+('IDLE',):
        raise ValueError('Unsupported default mode')
    target_id=_setting(config,'DEFAULT_TARGET_ID')
    station=_setting(config,'DEFAULT_STATION')
    iterations=frames=detect_total=send_accepted=0
    last_emit=None
    consecutive_errors=0
    uart_unsent_bytes=0
    started=clock_ms()
    stopped='ITERATION_LIMIT'
    try:
        camera.start()
        emit({'event':'START','mode':mode,'motors_enabled':False,
              'uart_enabled':getattr(uart,'enabled',False),
              'uart_reason':getattr(uart,'reason','UNKNOWN'),
              'camera_ids':list(_setting(config,'CAMERA_IDS',(2,)))})
        while max_iterations is None or iterations<max_iterations:
            exitpoint()
            iterations+=1
            stop=False
            for command in uart.poll():
                seq=command['seq']
                action=command['command']
                if action=='STOP':
                    uart.send_ack(seq,'STOPPED','OBSERVATION_SERVICE_STOPPED')
                    stopped='CONTROL_STOP'
                    stop=True
                    break
                if action=='PING':
                    uart.send_ack(seq,'PONG','OBSERVATION_ONLY')
                elif action=='MODE':
                    requested=command.get('mode')
                    if requested not in modes+('IDLE',):
                        uart.send_ack(seq,'ERROR','UNSUPPORTED_MODE')
                        continue
                    mode=requested
                    target_id=command.get('target_id')
                    station=command.get('station')
                    # Restart a bounded scan while preserving QR task de-duplication.
                    if hasattr(vision,'tracker'):
                        vision.tracker.reset()
                    if hasattr(vision,'qr'):
                        vision.qr.reset_scan()
                    uart.send_ack(seq,'ACK','MODE_SET')
                else:
                    uart.send_ack(seq,'ERROR','UNSUPPORTED_CONTROL')
            if stop:
                break
            uart.heartbeat(clock_ms())
            if mode=='IDLE':
                sleep_ms(5)
                continue
            before=clock_ms()
            try:
                frame=camera.get_frame(camera.camera_for_mode(mode))
                result=vision.detect(mode,frame,target_id=target_id,station=station)
                result.setdefault('target_pose_valid',False)
                if frame is not None and hasattr(camera,'show_frame'):
                    camera.show_frame(frame)
                consecutive_errors=0
            except Exception as error:
                from vision.common import observation
                result=observation(mode,None,target_id,station)
                result.update(status='ERROR',reason='CAPTURE_ERROR: '+str(error),target_pose_valid=False)
                consecutive_errors+=1
            detect_total+=max(0,_elapsed(clock_ms(),before))
            frames+=1
            try:
                if uart.send_observation(result):
                    send_accepted+=1
            except (ValueError,TypeError) as error:
                # A malformed observation cannot terminate stop/heartbeat handling.
                emit({'event':'SERIAL_ENCODE_ERROR','reason':str(error),'motors_enabled':False})
            now=clock_ms()
            if last_emit is None or _elapsed(now,last_emit)>=_setting(config,'REPORT_INTERVAL_MS',1000):
                emit({'event':'OBSERVATION','observation':result,'motors_enabled':False})
                last_emit=now
                gc.collect()
            if consecutive_errors>=_setting(config,'MAX_CAPTURE_ERRORS',3):
                mode='IDLE'
                emit({'event':'CAPTURE_UNAVAILABLE','action':'IDLE','motors_enabled':False})
            sleep_ms(_setting(config,'LOOP_YIELD_MS',2))
    finally:
        try:
            camera.close()
        finally:
            try:
                if hasattr(uart,'drain'):
                    uart.drain(max_writes=16)
                uart_unsent_bytes=len(getattr(uart,'tx_buffer',b''))
            finally:
                uart.close()
    metrics={'event':'STOP','reason':stopped,'iterations':iterations,'frames':frames,
             'mean_processing_ms':detect_total/frames if frames else None,
             'elapsed_ms':_elapsed(clock_ms(),started),'uart_queued_observations':send_accepted,
             'uart_unsent_bytes_on_close':uart_unsent_bytes,
             'motors_enabled':False,'heap_free_bytes':gc.mem_free() if hasattr(gc,'mem_free') else None,
             'camera_cleanup_errors':getattr(camera,'close_errors',[])}
    emit(metrics)
    return metrics


def main(config_override=None):
    import sys
    if sys.implementation.name!='micropython':
        raise SystemExit('Use CanMV on K230; PC simulation: python -m sim.run_mission.')
    import os
    from . import config
    config=config_override if config_override is not None else config
    from .camera_manager import CameraManager
    from .vision_adapter import VisionAdapter
    from .uart_adapter import UartAdapter
    uart=UartAdapter(_setting(config,'UART'))
    return run_loop(CameraManager(config),VisionAdapter(config),uart,config,
                    exitpoint=os.exitpoint,sleep_ms=time.sleep_ms)


if __name__=='__main__':
    main()
