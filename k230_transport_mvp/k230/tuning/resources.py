"""Bounded telemetry. Pool denominators stay distinct from physical DRAM."""
import gc,os


def snapshot():
    result={'python_heap':{'used_bytes':gc.mem_alloc(),'free_bytes':gc.mem_free()},'cpu_percent':None,'cpu_source':None,
        'kpu_usage_percent':None,'kpu_note':'No neural model/KPU inference in this threshold pipeline',
        'scope':'Current CanMV RTOS runtime; not whole dual-core SoC/Linux accounting'}
    heap=result['python_heap'];heap['total_bytes']=heap['used_bytes']+heap['free_bytes']
    heap['used_percent']=100*heap['used_bytes']/max(1,heap['total_bytes'])
    for label,name in (('system_heap','sys_heap'),('system_pages','sys_page'),('media_mmz','sys_mmz')):
        try:
            metric=getattr(gc,name)();total,used,free=metric
            if total<=0 or min(used,free)<0:raise ValueError('Invalid memory pool')
            result[label]={'total_bytes':total,'used_bytes':used,'free_bytes':free,'used_percent':100*used/total}
        except (AttributeError,ValueError,TypeError,OSError) as error:result[label]={'available':False,'reason':str(error)}
    try:
        cpu=os.cpu_usage()
        if not 0<=cpu<=100:raise ValueError('Invalid CPU sample')
        result['cpu_percent']=cpu;result['cpu_source']='os.cpu_usage'
    except (AttributeError,ValueError,TypeError,OSError) as error:result['cpu_unavailable_reason']=str(error)
    return result
