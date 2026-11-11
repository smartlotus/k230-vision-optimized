"""Host-only UART TX allocation comparison; never opens hardware."""
import argparse
import json
from pathlib import Path
import tracemalloc

from k230.uart_adapter import UartAdapter as Legacy
from k230.uart_adapter_optimized import UartAdapter


class Sink:
    limit = 0
    accepted = 0

    def write(self, chunk):
        count = min(self.limit, len(chunk))
        self.accepted += count
        return count


def benchmark():
    rows = []
    for implementation in (Legacy, UartAdapter):
        sink = Sink()
        adapter = implementation(dict(enabled=True, pins_verified=True,
            uart_id=1, tx_pin=3, rx_pin=4), transport=sink)
        adapter._send(b'x' * 3072)
        sink.limit = 1
        # Queue storage is allocated before measurement: this measures drain churn.
        tracemalloc.start()
        for _ in range(3072):
            adapter._flush()
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert sink.accepted == 3072 and not adapter.tx_buffer
        rows.append(dict(implementation=implementation.__module__,
                         accepted_bytes=sink.accepted, peak_drain_python_bytes=peak))
    return dict(scope='Host synthetic drain allocation; excludes initial queue storage; not K230 FPS or RSS',
                ring_capacity_bytes=3072, measurements=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))
