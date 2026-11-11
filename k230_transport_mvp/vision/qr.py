"""QR syntax validation, consecutive-frame confirmation and session deduplication."""
from .common import elapsed_ms


class QRTracker:
    def __init__(self, confirmations=3, timeout_ms=5000, max_gap_ms=500, validator=None):
        self.confirmations = confirmations
        self.timeout_ms = timeout_ms
        self.max_gap_ms = max_gap_ms
        self.validator = validator
        self.reset_session()

    def reset_session(self):
        self.submitted = set()
        self.reset_scan()

    def reset_scan(self):
        self.payload = None
        self.count = 0
        self.last_frame = None
        self.last_time = None
        self.started_ms = None
        self.source_key = None

    def update(self, payloads, frame_id, timestamp_ms, source_key=None):
        if self.started_ms is None or self.source_key != source_key:
            self.reset_scan()
            self.started_ms = timestamp_ms
            self.source_key = source_key
        if elapsed_ms(timestamp_ms, self.started_ms) >= self.timeout_ms:
            self.payload = None
            self.count = 0
            return {"status": "NOT_FOUND", "reason": "QR_TIMEOUT", "payload": None, "submitted": False}
        # An image replay is not another confirmation.
        if frame_id == self.last_frame:
            return {"status": "NOT_FOUND", "reason": "REPEATED_FRAME", "payload": None, "submitted": False}
        if self.last_frame is not None:
            gap = elapsed_ms(timestamp_ms, self.last_time)
            if frame_id < self.last_frame or gap <= 0 or gap > self.max_gap_ms:
                self.payload = None
                self.count = 0
        self.last_frame = frame_id
        self.last_time = timestamp_ms
        valid = set()
        for payload in payloads:
            try:
                if self.validator is None:
                    from core.task_parser import parse_task
                    parse_task(payload)
                else:
                    self.validator(payload)
                valid.add(payload)
            except (ValueError, TypeError):
                continue
        if len(valid) != 1:
            self.payload = None
            self.count = 0
            return {"status": "NOT_FOUND", "reason": "QR_AMBIGUOUS" if valid else "QR_INVALID_OR_ABSENT", "payload": None, "submitted": False}
        payload = next(iter(valid))
        self.count = self.count + 1 if payload == self.payload else 1
        self.payload = payload
        if payload in self.submitted:
            return {"status": "NOT_FOUND", "reason": "QR_DUPLICATE", "payload": payload, "submitted": False}
        if self.count < self.confirmations:
            return {"status": "NOT_FOUND", "reason": "QR_CONFIRMING", "payload": payload, "submitted": False}
        self.submitted.add(payload)
        return {"status": "OK", "reason": None, "payload": payload, "submitted": True}
