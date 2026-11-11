"""Host-only file hashing with bounded scratch memory and no digest cache."""
import hashlib
from pathlib import Path


def sha256_file(path):
    """Hash current file bytes, propagating missing-file and read errors."""
    digest = hashlib.sha256()
    buffer = bytearray(64 * 1024)
    view = memoryview(buffer)
    with Path(path).open("rb") as stream:
        while True:
            count = stream.readinto(buffer)
            if not count:
                break
            digest.update(view[:count])
    return digest.hexdigest()
