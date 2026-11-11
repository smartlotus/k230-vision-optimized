"""Prepare exact JSON envelopes and byte-exact source exports for the native bridge."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("source", "status", "stop", "preview", "view", "sample"))
    parser.add_argument("--file")
    parser.add_argument("--session")
    parser.add_argument("--seq", type=int)
    parser.add_argument("--color", type=int, default=4)
    args = parser.parse_args()
    if args.action == "source":
        data = Path(args.file).read_bytes()
        print(json.dumps({"base64": base64.b64encode(data).decode(), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}))
        return
    payload = {"action": args.action}
    if args.action == "preview":
        saved = json.loads(Path(args.file).read_text(encoding="utf-8-sig"))
        payload.update(profile=saved.get("profile", saved), context="TOP_BUFFER", color=args.color, slot=saved.get("slot", 1))
    if args.action == "view":
        payload["selected"] = True
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    print(json.dumps({"session": args.session, "seq": args.seq, "payload": raw, "sha256": hashlib.sha256(raw.encode()).hexdigest()}, separators=(",", ":")))


if __name__ == "__main__":
    main()
