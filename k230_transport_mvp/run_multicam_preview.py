"""Three-camera live preview with isolated capture recovery; CanMV only."""
import sys
import time

CAMERA_IDS = (2, 1, 0)  # Operational priority; layout remains fixed by physical CSI ID.
QUARANTINED_CAMERA_IDS = (0,)  # Repeated detection failure; explicit diagnostics only.
POSITIONS = {2: (40, 0), 0: (440, 0), 1: (40, 240)}
FIRST_FRAME_TIMEOUT_MS = 2000
FRAME_TIMEOUT_MS = 500


def new_states():
    return {cid: {'attempts': 0, 'frames': 0, 'errors': 0,
                  'consecutive_errors': 0, 'last_error': None,
                  'status': 'INITIALIZING'} for cid in CAMERA_IDS}


def render_round(sensors, canvas, states, exitpoint=None, quarantined_ids=()):
    """One failed source gets a retry tile; other sources continue to update."""
    for cid in CAMERA_IDS:
        if exitpoint is not None:
            exitpoint()
        state = states[cid]
        x, y = POSITIONS[cid]
        if cid in quarantined_ids:
            state['status'] = 'QUARANTINED'
            canvas.draw_string_advanced(x+4, y+4, 24, 'CSI'+str(cid)+' FLAGGED', color=(255,80,80))
            canvas.draw_string_advanced(x+4, y+40, 18, 'lowest priority / skipped', color=(255,255,255))
            continue
        if cid not in sensors:
            state['status'] = 'OFFLINE'
            canvas.draw_string_advanced(x+4, y+4, 24, 'CSI'+str(cid)+' OFFLINE', color=(255,80,80))
            canvas.draw_string_advanced(x+4, y+40, 18, 'not detected / init failed', color=(255,255,255))
            continue
        timeout = FIRST_FRAME_TIMEOUT_MS if state['attempts'] == 0 else FRAME_TIMEOUT_MS
        state['attempts'] += 1
        try:
            frame = sensors[cid].snapshot(timeout=timeout, auto_reboot=False)
            if frame is None or (frame.width(), frame.height()) != (640, 480):
                raise RuntimeError('No valid 640x480 frame')
        except Exception as error:
            if 'IDE interrupt' in str(error):
                raise
            state['errors'] += 1
            state['consecutive_errors'] += 1
            state['last_error'] = str(error)
            state['status'] = 'RETRY'
            canvas.draw_string_advanced(x+4, y+4, 24, 'CSI'+str(cid)+' RETRY', color=(255,80,80))
            canvas.draw_string_advanced(x+4, y+40, 18, 'capture failed; retrying', color=(255,255,255))
            continue
        canvas.draw_image(frame, x, y, x_scale=0.5, y_scale=0.5)
        canvas.draw_string_advanced(x+4, y+4, 24, 'CSI'+str(cid), color=(255,255,0))
        state['frames'] += 1
        state['consecutive_errors'] = 0
        state['status'] = 'LIVE'


def main():
    if sys.implementation.name != 'micropython':
        raise SystemExit('Use CanMV: Run Active Python Script on K230, not desktop Python.')
    import os
    import gc
    import json
    import image
    from media.sensor import Sensor
    from media.display import Display
    from media.media import MediaManager
    sensors = {}
    display_ready = media_ready = False
    states = new_states()
    rounds = 0
    started = time.ticks_ms()
    try:
        for cid in CAMERA_IDS:
            if cid in QUARANTINED_CAMERA_IDS:
                states[cid].update(status='QUARANTINED', last_error='Repeated detection failure; explicit diagnostics only')
                continue
            sensor = None
            try:
                sensor = Sensor(id=cid, width=1920, height=1080, fps=30)
                sensor.reset()
                sensor.set_framesize(width=640, height=480)
                sensor.set_pixformat(Sensor.RGB565)
                sensors[cid] = sensor
            except Exception as error:
                if 'IDE interrupt' in str(error): raise
                states[cid].update(status='OFFLINE', errors=1, last_error=str(error))
                print('K230_CAMERA_INIT_FAILED:'+json.dumps({'camera_id':cid,'error':str(error)}))
                if sensor is not None:
                    try: sensor.stop(is_del=True)
                    except Exception as cleanup_error: print('partial camera cleanup',cleanup_error)
        if not sensors:
            raise RuntimeError('No cameras initialized')
        Display.init(Display.ST7701, width=800, height=480, fps=30, to_ide=True)
        display_ready = True
        MediaManager.init()
        media_ready = True
        sensors[next(cid for cid in CAMERA_IDS if cid in sensors)].run()
        canvas = image.Image(800, 480, image.RGB565)
        print('K230_THREE_LIVE_STARTED:'+json.dumps({'active_camera_ids':list(sensors),'per_camera_retry':True,
              'priority':CAMERA_IDS,'quarantined_ids':QUARANTINED_CAMERA_IDS}))
        while True:
            os.exitpoint()
            canvas.draw_rectangle(0,0,800,480,color=(0,0,0),fill=True)
            render_round(sensors, canvas, states, os.exitpoint, QUARANTINED_CAMERA_IDS)
            canvas.draw_string_advanced(410,260,20,'CAMERAS '+str(len(sensors))+'/3',color=(255,255,255))
            for index, cid in enumerate(CAMERA_IDS):
                state = states[cid]
                text = 'CSI%d frames=%d errors=%d' % (cid,state['frames'],state['errors'])
                canvas.draw_string_advanced(410,300+index*30,16,text,color=(255,255,255))
            canvas.draw_string_advanced(410,410,18,'CanMV Stop / Ctrl+C',color=(255,255,255))
            Display.show_image(canvas)
            rounds += 1
            if rounds % 30 == 0:
                elapsed = time.ticks_diff(time.ticks_ms(), started)
                print('K230_THREE_LIVE:'+json.dumps({'rounds':rounds,'elapsed_ms':elapsed,
                      'preview_rounds_per_second':rounds*1000/elapsed,'states':states}))
                gc.collect()
            time.sleep_ms(10)
    except KeyboardInterrupt:
        print('K230_THREE_LIVE_STOPPED')
    except Exception as error:
        if 'IDE interrupt' in str(error):
            print('K230_THREE_LIVE_STOPPED')
        else:
            sys.print_exception(error)
    finally:
        for sensor in sensors.values():
            try:
                sensor.stop()
            except Exception as error:
                print('camera cleanup', error)
        if display_ready:
            try:
                Display.deinit()
            except Exception as error:
                print('display cleanup', error)
        if media_ready:
            try:
                MediaManager.deinit()
            except Exception as error:
                print('media cleanup', error)


if __name__ == '__main__':
    main()

