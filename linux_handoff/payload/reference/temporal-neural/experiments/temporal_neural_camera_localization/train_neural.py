"""Train a matched frame CNN and causal TCN pilot on Kaggle.

This run learns the spatial encoder and the XY head jointly. The previous
pilot froze the frame embeddings, so it could only refine the existing match.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import tarfile
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F


TARGETS = ('camera_x_mm', 'camera_y_mm')
SEED = 117
SETTINGS = {'cnn1': 1, 'tcn8': 8, 'tcn16': 16}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


class SpatialMixtureHead(nn.Module):
    """Rasterize eight learned continuous XY distributions on the shared grid."""
    def __init__(self):
        super().__init__()
        self.parameters_xy = nn.Linear(96, 8 * 5)
        nn.init.zeros_(self.parameters_xy.weight)
        x, y = torch.meshgrid(torch.linspace(-.75, .75, 4), torch.linspace(-.65, .65, 2), indexing='xy')
        init = torch.stack((x.flatten(), y.flatten(), torch.zeros(8), torch.zeros(8), torch.zeros(8)), dim=-1)
        init[:, :2] = torch.atanh(init[:, :2])
        with torch.no_grad():
            self.parameters_xy.bias.copy_(init.flatten())
        axis = torch.arange(-1175, 1200, 50, dtype=torch.float32) / 1200
        xx, yy = torch.meshgrid(axis, axis, indexing='xy')
        self.register_buffer('anchors', torch.stack((xx.flatten(), yy.flatten()), dim=-1))

    def forward(self, features):
        params = self.parameters_xy(features).reshape(-1, 8, 5)
        centers = torch.tanh(params[:, :, :2])
        sigma = .07 + .63 * torch.sigmoid(params[:, :, 2:4])
        displacement = (self.anchors[None, None] - centers[:, :, None]) / sigma[:, :, None]
        component = -.5 * (displacement * displacement).sum(-1) + params[:, :, 4, None]
        return torch.logsumexp(component, dim=1).reshape(*features.shape[:-1], 2304)


class CameraXY(nn.Module):
    def __init__(self, window, parent, head_type='linear'):
        super().__init__()
        self.window = window
        self.dilations = tuple(2**level for level in range(window.bit_length() - 1))
        layers = []
        for a, b in ((3, 16), (16, 24), (24, 32), (32, 48)):
            layers.extend([nn.Conv2d(a, b, 3, 2, 1, bias=False), nn.BatchNorm2d(b), nn.ReLU()])
        self.rgb = nn.Sequential(*layers, nn.AvgPool2d(2), nn.Flatten())
        self.rgb.load_state_dict({key[4:]: value for key, value in parent.items()
                                  if key.startswith('rgb.') and int(key.split('.')[1]) < 12})
        self.projection = nn.Sequential(nn.Linear(720, 96), nn.ReLU())
        if window > 1:
            self.temporal = nn.ModuleList(
                nn.Conv2d(96, 96, (1, 2), dilation=(1, dilation)) for dilation in self.dilations)
        self.head = SpatialMixtureHead() if head_type == 'mixture' else nn.Linear(96, 48 * 48)

    def forward(self, packed, return_history=False):
        batch = packed.shape[0]
        features = self.projection(self.rgb(packed.reshape(-1, 3, 96, 160)))
        if self.window > 1:
            frame_features = features.reshape(batch, self.window, 96)
            words = frame_features.transpose(1, 2).unsqueeze(2)
            for conv, dilation in zip(self.temporal, self.dilations):
                words = words + F.relu(conv(F.pad(words, (dilation, 0, 0, 0))))
            features = words[:, :, 0, -1]
            if return_history:
                return self.head(features), self.head(frame_features[:, :-1])
        return self.head(features)


def load(data, expected_manifest, train_and_val_only=False):
    manifest = data / 'manifest.jsonl'
    if sha(manifest) != expected_manifest:
        raise ValueError('Frozen renderer manifest differs')
    rows = [json.loads(line) for line in manifest.read_text(encoding='utf8').splitlines()]
    groups = defaultdict(list)
    split = defaultdict(list)
    rgb, xy = [], []
    for index, row in enumerate(rows):
        if set(row['camera_xy_target']) != set(TARGETS):
            raise ValueError('Every supervised target must contain camera optical-center XY only')
        groups[row['trajectory_id']].append(index)
        split[row['split']].append(index)
        path = (data / row['image_path']).resolve()
        if not path.is_relative_to(data.resolve()) or sha(path) != row['image_sha256']:
            raise ValueError('RGB differs from the frozen trajectory manifest')
        with Image.open(path) as image:
            resized = image.convert('RGB').resize((160, 90), Image.Resampling.BILINEAR)
            padded = ImageOps.expand(resized, (0, 3, 0, 3), fill=(114, 114, 114))
            rgb.append(np.asarray(padded).transpose(2, 0, 1))
        xy.append([row['camera_xy_target'][field] for field in TARGETS])
    sizes = {name: len(ids) for name, ids in split.items()}
    expected = {'train': 500, 'val': 250}
    if not train_and_val_only:
        expected['test'] = 250
    if sizes != expected or len(groups) != sum(expected.values()) // 25:
        raise ValueError('Expected the frozen complete-trajectory pilot split')
    for indices in groups.values():
        members = [rows[i] for i in indices]
        if (len({r['split'] for r in members}) != 1 or len(members) != 25
                or any(a['timestamp_s'] >= b['timestamp_s'] for a, b in zip(members, members[1:]))):
            raise ValueError('Expected complete increasing trajectories within each split')
    for key in ('appearance_profile_id', 'pose_group_id'):
        assignment = defaultdict(set)
        for row in rows:
            assignment[row[key]].add(row['split'])
        if any(len(split) > 1 for split in assignment.values()):
            raise ValueError('Appearance or pose group crosses a holdout split')
    return np.stack(rgb), np.asarray(xy), groups, split


def windows(groups, window, frames):
    value = np.empty((frames, window), dtype=np.int64)
    for indices in groups.values():
        for offset, target in enumerate(indices):
            history = indices[max(0, offset - window + 1):offset + 1]
            value[target] = [history[0]] * (window - len(history)) + history
    return value


def export(model, folder, example, selected_state, meta):
    model.load_state_dict(selected_state)
    model = model.cpu().eval()
    export_path = folder / 'camera_xy.onnx'
    torch.onnx.export(model, example, str(export_path), input_names=['rgb_window'],
                      output_names=['location_logits'], opset_version=13, dynamo=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(export_path), sess_options=options, providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        expected = model(example).numpy()
    actual, = session.run(['location_logits'], {'rgb_window': example.numpy()})
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3)
    torch.save(dict(state=selected_state, **meta), folder / 'best.pt')
    return float(np.max(np.abs(actual - expected)))


def photometric_rgb(packed, rng):
    """Apply one camera/illumination perturbation per clip while retaining frame order."""
    count, channels, height, width = packed.shape
    window = channels // 3
    value = packed.reshape(count, window, 3, height, width)
    # Draw the same factors for a clip's past and current observations. Keep
    # the preprocessor's constant three-pixel border unchanged.
    samples = torch.from_numpy(rng.random((count, 5), dtype=np.float32)).to(packed.device)
    brightness = torch.exp2(samples[:, :1] * 1.5 - .75)
    channel_gain = .77 + samples[:, 1:4] * .46
    gamma = .82 + samples[:, 4:5] * .36
    gain = (brightness * channel_gain).reshape(count, 1, 3, 1, 1)
    content = (value[..., 3:-3, :] * gain).clamp(0, 1).pow(gamma.reshape(count, 1, 1, 1, 1))
    return torch.cat((value[..., :3, :], content, value[..., -3:, :]), dim=-2).reshape_as(packed)


def history_xy_loss(history_logits, indices, targets):
    """Supervise valid past observations once each when startup repeats a frame."""
    past = indices[:, :-1]
    new_observation = torch.cat((torch.ones_like(past[:, :1], dtype=torch.bool), past[:, 1:] != past[:, :-1]), dim=1)
    valid = new_observation & (past != indices[:, -1:])
    error = F.cross_entropy(history_logits.flatten(0, 1), targets[past].flatten(0, 1), reduction='none')
    return (error * valid.flatten()).sum() / valid.sum().clamp_min(1)


def run_setting(setting, root, images, targets, centers, groups, split, parent, identity, epochs, device):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    color_rng = np.random.default_rng(SEED + 1024)
    folder = root / 'models' / setting
    folder.mkdir(parents=True)
    window = SETTINGS[setting]
    history_weight = identity['history_xy_weight'] if window > 1 else 0.
    index = torch.from_numpy(windows(groups, window, len(images))).to(device)
    model = CameraXY(window, parent, identity.get('head_type', 'linear')).to(device)
    optimizer = torch.optim.AdamW([{'params': model.rgb.parameters(), 'lr': 1e-4},
                                   {'params': list(model.projection.parameters()) + list(model.head.parameters())
                                    + list(getattr(model, 'temporal', nn.Identity()).parameters()), 'lr': 8e-4}],
                                  weight_decay=1e-3)

    def clips(ids):
        return images[index[ids]].reshape(len(ids), 3 * window, 96, 160).float() / 255

    train, val = split['train'], torch.tensor(split['val'], device=device)
    history = []
    best = float('inf')
    updates = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in np.array_split(rng.permutation(train), 16):
            ids = torch.as_tensor(batch, device=device)
            optimizer.zero_grad(set_to_none=True)
            rgb = clips(ids)
            if identity['recipe'] == 'clip_photometric':
                rgb = photometric_rgb(rgb, color_rng)
            if history_weight:
                current, past = model(rgb, return_history=True)
                loss = F.cross_entropy(current, targets[ids])
                loss = loss + history_weight * history_xy_loss(past, index[ids], targets)
            else:
                loss = F.cross_entropy(model(rgb), targets[ids])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            updates += 1
        model.eval()
        with torch.inference_mode():
            logits = torch.cat([model(clips(chunk)) for chunk in val.split(32)])
            nll = float(F.cross_entropy(logits, targets[val]))
        if not np.isfinite(nll):
            raise ValueError('Nonfinite validation loss')
        history.append(dict(epoch=epoch, optimizer_updates=updates, validation_xy_nll=nll))
        if nll < best:
            best = nll
            selected = epoch
            weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            temperature = min((.5, 1., 2., 4.),
                              key=lambda temp: float(F.cross_entropy(logits / temp, targets[val])))
        print(json.dumps(dict(setting=setting, **history[-1])), flush=True)
    torch.save(dict(state=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epochs,
                    optimizer_updates=updates, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=rng.bit_generator.state,
                    photometric_rng=color_rng.bit_generator.state,
                    **identity), folder / 'last.pt')
    metadata = dict(route='cnn' if window == 1 else 'tcn', window=window, targets=list(TARGETS),
                    centers_mm=centers.tolist(), selected_epoch=selected, **identity)
    error = export(model, folder, clips(val[:1]).cpu(), weights, metadata)
    metrics = dict(selected_epoch=selected, optimizer_updates=updates, temperature=temperature,
                   best_sha256=sha(folder / 'best.pt'), onnx_sha256=sha(folder / 'camera_xy.onnx'),
                   ort_max_abs=error, selection='minimum camera-XY soft NLL on validation trajectories')
    save_json(folder / 'METRICS.json', metrics)
    save_json(folder / 'history.json', history)


def main(args):
    started = time.monotonic()
    if Path.cwd() != Path('/kaggle/working') or not torch.cuda.is_available():
        raise RuntimeError('This training entry requires an accelerated Kaggle notebook')
    if sha(args.input / 'parent.pt') != args.parent_sha:
        raise ValueError('Pretrained encoder differs from frozen parent')
    images, xy, groups, split = load(args.input / 'data', args.manifest_sha, args.train_and_val_only)
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(2)
    images = torch.from_numpy(images).to(device)
    axis = np.arange(-1175, 1200, 50)
    xx, yy = np.meshgrid(axis, axis)
    centers = np.column_stack((xx.ravel(), yy.ravel()))
    delta = (xy[:, None, :] - centers[None, :, :]) / 85.
    targets = torch.softmax(torch.from_numpy(-.5 * (delta**2).sum(-1)).float(), dim=-1).to(device)
    parent = torch.load(args.input / 'parent.pt', weights_only=True, map_location='cpu')
    if tuple(parent['targets']) != TARGETS:
        raise ValueError('Pretrained model uses a different target contract')
    identity = dict(dataset_manifest_sha256=args.manifest_sha, parent_sha256=args.parent_sha, recipe=args.recipe,
                    history_xy_weight=args.history_xy_weight, head_type=args.head)
    out = Path(args.run_id)
    for setting in args.settings:
        run_setting(setting, out, images, targets, centers, groups, split, parent['state'], identity,
                    args.epochs, device)
    members = [dict(name=p.relative_to(out).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(out.rglob('*')) if p.is_file()]
    save_json(out / 'MEMBERS.json', members)
    archive = Path(args.run_id + '.tar.gz')
    with tarfile.open(archive, 'x:gz') as pack:
        pack.add(out, arcname=out.name)
    save_json(Path('RESULT.json'), dict(run_id=args.run_id, compute_status='PASS', train_exit_code=0,
                                       elapsed_seconds=time.monotonic() - started,
                                       archive=dict(name=archive.name, bytes=archive.stat().st_size, sha256=sha(archive)),
                                       automatic_replay=False, **identity))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--parent-sha', required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--recipe', choices=('plain', 'clip_photometric'), default='plain')
    parser.add_argument('--train-and-val-only', action='store_true')
    parser.add_argument('--history-xy-weight', type=float, default=0.)
    parser.add_argument('--settings', nargs='+', choices=SETTINGS, default=list(SETTINGS))
    parser.add_argument('--head', choices=('linear', 'mixture'), default='linear')
    args = parser.parse_args()
    if not 0 <= args.history_xy_weight <= 1 or len(args.settings) != len(set(args.settings)):
        parser.error('Use a finite history weight in [0, 1] and unique settings')
    main(args)
