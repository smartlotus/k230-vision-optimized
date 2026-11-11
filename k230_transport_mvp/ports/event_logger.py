"""Small JSONL event callback; transport receipt is distinct from completion."""
try:
    import json
except ImportError:
    import ujson as json


class EventLogger:
    def __init__(self, path=None, stream=None, max_record_bytes=4096):
        if path is not None and stream is not None:
            raise ValueError("choose path or stream")
        self.owned = path is not None
        self.stream = open(path, "a") if self.owned else stream
        self.max_record_bytes = max_record_bytes
        self.count = 0
        self.closed = False
        self.reason = None

    def __call__(self, event):
        if self.closed:
            self.reason = "LOGGER_CLOSED"
            return False
        if not isinstance(event, dict):
            raise ValueError("event must be a dictionary")
        line = json.dumps(event) + "\n"
        if len(line.encode("utf-8")) > self.max_record_bytes:
            self.reason = "EVENT_TOO_LONG"
            return False
        if self.stream is None:
            print(line.rstrip("\n"))
        else:
            self.stream.write(line)
            self.stream.flush()
        self.count += 1
        return True

    def close(self):
        if self.closed:
            return
        if self.stream is not None:
            self.stream.flush()
            if self.owned:
                self.stream.close()
        self.closed = True
