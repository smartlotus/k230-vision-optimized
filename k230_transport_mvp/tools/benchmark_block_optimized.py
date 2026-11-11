"""Deterministic host tracking benchmark; excludes native image detection."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import time
import tracemalloc

from k230.block_profile import PARAMETERS
from vision.block_detector import BlockDetector as Legacy
from vision.block_detector_optimized import BlockDetector


def benchmark(iterations=1500):
    results = []
    for count in (3, 6, 12):
        candidates = [dict(target_id=i % 6 + 1, center=(30+i*20, 70), eligible=True)
                      for i in range(count)]
        timings = {Legacy: [], BlockDetector: []}
        peaks = {}
        for repeat in range(5):
            order = (Legacy, BlockDetector) if repeat % 2 == 0 else (BlockDetector, Legacy)
            for implementation in order:
                detector = implementation(copy.deepcopy(PARAMETERS))
                started = time.perf_counter()
                for i in range(iterations):
                    output = detector.update(candidates, i+1, (i+1)*40)
                timings[implementation].append(time.perf_counter()-started)
                assert all(row['stable'] for row in output)
        outputs = []
        for implementation in (Legacy, BlockDetector):
            detector = implementation(copy.deepcopy(PARAMETERS))
            tracemalloc.start()
            for i in range(100):
                output = detector.update(candidates, i+1, (i+1)*40)
            peaks[implementation] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            outputs.append(output)
        assert outputs[0] == outputs[1]
        row = {'candidates': count, 'iterations_per_repeat': iterations, 'equal_output': True}
        for name, implementation in (('legacy', Legacy), ('optimized', BlockDetector)):
            row[name] = dict(seconds=timings[implementation],
                             median_seconds=statistics.median(timings[implementation]),
                             peak_python_bytes=peaks[implementation])
        results.append(row)
    return dict(scope='CPython synthetic tracking only; excludes native find_blobs, cameras and display; not board FPS',
                measurements=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))
