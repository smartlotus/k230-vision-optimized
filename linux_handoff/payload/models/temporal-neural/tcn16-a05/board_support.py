"""Standalone bench counters adapted from the audited RoadNet/TopFace benches."""
import gc


def digest(path):
    import hashlib
    import binascii
    hasher = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            block = stream.read(4096)
            if not block:
                break
            hasher.update(block)
    return binascii.hexlify(hasher.digest()).decode()


def resources():
    result = dict(python_heap_used=gc.mem_alloc(), python_heap_free=gc.mem_free())
    for name in ('sys_heap', 'sys_page', 'sys_mmz'):
        try:
            total, used, free = getattr(gc, name)()
            result[name] = dict(total=total, used=used, free=free)
        except (AttributeError, ValueError, OSError):
            pass
    return result


class Latency:
    def __init__(self):
        self.counts = [0] * 1002
        self.count = 0
        self.total_us = 0

    def add(self, elapsed_us):
        index = min(1001, max(0, (elapsed_us + 999) // 1000))
        self.counts[index] += 1
        self.count += 1
        self.total_us += elapsed_us

    def report(self):
        def percentile(p):
            remaining = (self.count * p + 99) // 100
            for elapsed, count in enumerate(self.counts):
                remaining -= count
                if remaining <= 0:
                    return elapsed if elapsed <= 1000 else None
        return dict(samples=self.count,
                    mean_ms=self.total_us / self.count / 1000 if self.count else None,
                    p50_ms_upper=percentile(50) if self.count else None,
                    p95_ms_upper=percentile(95) if self.count else None,
                    p99_ms_upper=percentile(99) if self.count else None,
                    overflow_over_1000ms=self.counts[1001])


def sample_memory(result, reader, phase):
    try:
        memory = reader()
        if phase != 'active':
            result[phase + '_memory'] = memory
            return
        heap_free = memory['python_heap_free']
        result['minimum_python_heap_free'] = min(result.get('minimum_python_heap_free', heap_free), heap_free)
        for name, pool in memory.items():
            if isinstance(pool, dict) and pool['used'] > result['active_pool_peaks'].get(name, {}).get('used', -1):
                result['active_pool_peaks'][name] = pool
        result['active_memory_samples'] += 1
    except Exception as error:
        result['memory_sampling_failures'] += 1
        result['last_memory_sampling_failure'] = dict(phase=phase, message=str(error))
