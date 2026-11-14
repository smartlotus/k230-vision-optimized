"""Locate one hash-bound input across Kaggle's supported mount layouts."""
import hashlib
from pathlib import Path


def locate_input(root, manifest_sha256):
    root = Path(root).resolve()
    matches = []
    candidates = list(root.rglob("INPUT_MANIFEST.json"))
    if len(candidates) > 20:
        raise RuntimeError("Too many candidate manifests; require explicit input review")
    for candidate in candidates:
        candidate = candidate.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError("Input manifest resolves outside Kaggle input mount")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() == manifest_sha256:
            matches.append(candidate.parent)
    if len(matches) != 1:
        names = [str(path.relative_to(root)) for path in candidates]
        raise RuntimeError(f"Expected exactly one hash-bound input, found {len(matches)}; candidates={names}")
    return matches[0]
