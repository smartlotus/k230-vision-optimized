"""Decode a recorded video with its original frame presentation timestamps."""
import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess

from PIL import Image

from dataset import digest
from replay_camera import load_trajectories


def prepare(video, out, trajectory, source, start_zone=None, max_frames=None):
    if not trajectory.strip() or max_frames is not None and max_frames < 1:
        raise ValueError('Supply a trajectory identifier and a positive optional frame limit')
    if not video.is_file():
        raise FileNotFoundError(video)
    if out.exists():
        raise FileExistsError(out)
    for program in ('ffprobe', 'ffmpeg'):
        if shutil.which(program) is None:
            raise ValueError(f'Install {program} or provide an existing timestamped RGB manifest')
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames', '-show_streams',
         '-show_entries', 'frame=best_effort_timestamp_time:stream=width,height,codec_name,time_base',
         '-of', 'json', str(video.resolve())], capture_output=True, text=True, encoding='utf8', timeout=120, check=True)
    observed = json.loads(probe.stdout)
    stream = observed['streams'][0]
    if stream['width'] * 9 != stream['height'] * 16:
        raise ValueError('Use the original 16:9 camera recording')
    decoded = observed['frames'] if max_frames is None else observed['frames'][:max_frames]
    timestamps = [float(frame['best_effort_timestamp_time']) for frame in decoded]
    if (not timestamps or any(not math.isfinite(t) for t in timestamps)
            or any(b <= a for a, b in zip(timestamps, timestamps[1:]))):
        raise ValueError('Video must contain strictly increasing presentation timestamps')
    origin = timestamps[0]
    out.mkdir(parents=True)
    (out / 'frames').mkdir()
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-noautorotate',
               '-i', str(video.resolve()), '-map', '0:v:0', '-fps_mode', 'passthrough']
    if max_frames is not None:
        command.extend(['-frames:v', str(max_frames)])
    command.extend(['-start_number', '0', str(out.resolve() / 'frames/%08d.png')])
    subprocess.run(command, check=True)
    images = sorted((out / 'frames').glob('*.png'))
    if len(images) != len(timestamps):
        raise ValueError('Decoded frame count differs from the timestamp probe')
    rows = []
    for i, (image, stamp) in enumerate(zip(images, timestamps)):
        with Image.open(image) as rgb:
            if rgb.size != (stream['width'], stream['height']):
                raise ValueError('Decoded orientation differs from the original recording')
        row = {'trajectory_id': trajectory, 'frame_index': i, 'timestamp_s': stamp - origin,
               'image_path': image.relative_to(out).as_posix(), 'image_sha256': digest(image)}
        if i == 0 and start_zone is not None:
            row['start_zone'] = start_zone
        rows.append(row)
    manifest = out / 'manifest.jsonl'
    with manifest.open('x', encoding='utf8') as target:
        target.writelines(json.dumps(row) + '\n' for row in rows)
    load_trajectories(manifest)
    identity = {'status': 'TIMESTAMPED_RECORDING_PREPARED', 'declared_source': source,
                'video_sha256': digest(video), 'video_name': video.name, 'stream': stream,
                'frames': len(rows), 'recorded_pts_origin_s': origin,
                'timestamp_method': 'ffprobe best_effort_timestamp_time, subtract first decoded PTS',
                'manifest_sha256': digest(manifest), 'frame_limit': max_frames,
                'ground_truth': 'NOT PROVIDED', 'native_k230': 'NOT RUN'}
    with (out / 'CAPTURE_IDENTITY.json').open('x', encoding='utf8') as target:
        json.dump(identity, target, indent=2)
    return identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--trajectory-id', required=True)
    parser.add_argument('--source', choices=('ov2735-recording', 'simulated'), required=True)
    parser.add_argument('--start-zone', type=int, choices=(1, 2))
    parser.add_argument('--max-frames', type=int)
    args = parser.parse_args()
    print(json.dumps(prepare(args.video, args.out, args.trajectory_id, args.source,
                             args.start_zone, args.max_frames)))


if __name__ == '__main__':
    main()
