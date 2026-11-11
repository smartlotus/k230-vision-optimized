"""Optional measured exposure freeze for controlled lighting experiments."""


def configure_exposure(sensor,lock_requested=False):
    result={'lock_requested':lock_requested,'locked':False,'auto_before':None,
            'exposure_before_us':None,'auto_after':None,'exposure_after_us':None,'error':None}
    try:
        result['auto_before']=sensor.auto_exposure()
        exposure=sensor.exposure()
        result['exposure_before_us']=exposure
        if lock_requested:
            if type(exposure) not in (int,float) or exposure<=0:
                raise ValueError('Measured exposure unavailable; lock skipped')
            sensor.auto_exposure(False)
            sensor.exposure(exposure)
        result['auto_after']=sensor.auto_exposure()
        result['exposure_after_us']=sensor.exposure()
        result['locked']=lock_requested and result['auto_after'] is False
    except Exception as error:
        result['error']=str(error)
        if lock_requested:
            try:sensor.auto_exposure(True)
            except Exception:pass
    return result
