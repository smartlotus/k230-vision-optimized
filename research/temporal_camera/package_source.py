"""Build the portable source and evidence companion to the model/data ZIPs."""
import argparse
from pathlib import Path

from package_offline import write_package


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    docs = root / 'docs/temporal_camera'
    files = {}
    for path in sorted(here.rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.json', '.pt') and '__pycache__' not in path.parts:
            files[path.relative_to(root).as_posix()] = (path, None)
    for path in sorted(docs.rglob('*')):
        if path.is_file() and path.suffix in ('.md', '.json', '.jsonl', '.ipynb', '.log', '.png', '.blend'):
            files[path.relative_to(root).as_posix()] = (path, None)
    for relative in ('AGENTS.md', '.ops/PROJECT.md', '.ops/PROTOCOL.md'):
        files[relative] = (root / relative, None)
    files['research/topface/cloud_store.py'] = (here.parent / 'topface/cloud_store.py', None)
    text = """# K230 CameraXY V1 offline delivery

Start with docs/temporal_camera/OFFLINE_HANDOVER.md for achieved results and pending physical acceptance.
Use docs/temporal_camera/OFFLINE_REPLAY.md for recorded RGB/video input, and
docs/temporal_camera/map-correction/BOARD_MULTIROUTE.md for the isolated board procedure.

The Python tools are under research/temporal_camera. Replay uses the separate
K230_Official_Gray_Map_Pilot_A06_Models.zip and
K230_Official_Gray_Map_A_TCN4_Board_Bench_R2.zip files without extracting weights.
Their compiled outputs and inference configuration remain experimental.

This archive contains source, configuration, map assets and evidence.
Model and dataset binaries are delivered as separate SHA256-verified archives.
ONNX and KModel compile smoke binaries referenced by older reports remain in
their corresponding model archives or the development repository.
The validated replay host used Python 3.11.5. requirements-replay.txt records
its three inference dependencies. Optional FFmpeg/ffprobe are required only
for preparing a timestamped video. Training runs on private Kaggle.

From this directory:

    python research/temporal_camera/verify_delivery.py ../K230_Official_Gray_Map_Pilot_A06_Models.zip
    python -m unittest discover -s research/temporal_camera -p "test_*.py"

Hardware measurements, real-camera replay, and full-scale training remain
NOT RUN. The original goal has pending accuracy and physical acceptance.
Cloud publication helpers refer to the original accepted account workflow.
Use the original project checkout and its recorded credentials for that path.
The portable .ops instructions describe development coordination in the
repository. Continue development in the recorded project checkout.
"""
    requirements = 'numpy==2.2.6\nPillow==12.3.0\nonnxruntime==1.24.4\n'
    write_package(args.out, files, {'START_HERE.md': text.encode(), 'requirements-replay.txt': requirements.encode()},
                  'OFFLINE_SOURCE_AND_EVIDENCE_ACCURACY_HOLD')


if __name__ == '__main__':
    main()
