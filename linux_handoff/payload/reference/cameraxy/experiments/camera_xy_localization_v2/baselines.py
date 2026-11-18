"""Reference retrieval and semantic template alignment in FIELD_MAP."""
import math

import numpy as np
from PIL import Image

from contract import classify, map_to_pixel
import torch


def rgb_descriptors(images):
    # Preserve normalized color ratios and local edges while suppressing
    # global illumination gain. Compare reference views across all yaw angles.
    feat = np.stack([np.asarray(Image.fromarray(image.transpose(1, 2, 0)[3:-3])
                      .resize((20, 12), Image.Resampling.BOX), np.float32)
                     for image in images])
    feat /= np.maximum(feat.mean(axis=(1, 2, 3), keepdims=True), 12)
    gx = np.diff(feat.mean(-1), axis=2, append=feat.mean(-1)[:, :, -1:])
    gy = np.diff(feat.mean(-1), axis=1, append=feat.mean(-1)[:, -1:, :])
    return np.column_stack((feat.reshape(len(feat), -1),
                            2 * gx.reshape(len(feat), -1), 2 * gy.reshape(len(feat), -1)))


def separated_top5(scores, reference_xy, distance_mm):
    order = np.argsort(scores, kind="stable")
    kept = []
    for i in order:
        if not kept or np.all(np.linalg.norm(reference_xy[kept] - reference_xy[i], axis=1) >= distance_mm):
            kept.append(i)
            if len(kept) == 5:
                break
    if len(kept) != 5:
        raise ValueError("Reference map has insufficient unique candidate locations")
    return np.array(kept)


def retrieval(features, xy, reference, queries):
    bank = features[reference]
    ref_xy = xy[reference]
    norms = (bank * bank).mean(1)
    cand, score, pairs = [], [], []
    for i in queries:
        query = features[i]
        d = np.maximum(norms + (query * query).mean() - 2 * (bank @ query) / query.size, 0)
        picked = separated_top5(d, ref_xy, 100)
        cand.append(ref_xy[picked])
        score.append(float((d[picked[1]] - d[picked[0]]) / (d[picked[0]] + .01)))
        distant = np.where(np.linalg.norm(ref_xy - xy[i], axis=1) >= 250)[0]
        j = distant[d[distant].argmin()]
        pairs.append((int(i), int(reference[j]), float(d[j]), float(np.linalg.norm(xy[i] - ref_xy[j]))))
    return np.asarray(cand), np.asarray(score), pairs


def template_map(field_mask, layout, margin, step=50, yaw_steps=24):
    """Use same static RoadNet map at candidate locations across all yaw hypotheses."""
    gx, gy = np.meshgrid(np.arange(-1200 + step / 2, 1200, step),
                         np.arange(-1200 + step / 2, 1200, step))
    centers = np.array([(x, y) for x, y in zip(gx.ravel(), gy.ravel())
                        if classify(x, y, layout, margin)])
    angles = np.arange(yaw_steps) * math.tau / yaw_steps
    fx = math.hypot(32, 18) / (2 * math.tan(math.radians(96) / 2))
    u, v = np.meshgrid((np.arange(32) - 15.5) * 400 / fx, (8.5 - np.arange(18)) * 400 / fx)
    a, b = np.cos(angles)[:, None], np.sin(angles)[:, None]
    x = centers[:, None, None, 0] + a * u.ravel() - b * v.ravel()
    y = centers[:, None, None, 1] + b * u.ravel() + a * v.ravel()
    i, j = map_to_pixel(x, y, *field_mask.shape[::-1], 3000)
    ui, vi = np.rint(i).astype(int), np.rint(j).astype(int)
    sampled = field_mask[vi.clip(0, field_mask.shape[0]-1), ui.clip(0, field_mask.shape[1]-1)]
    sampled[(ui < 0) | (ui >= field_mask.shape[1]) | (vi < 0) | (vi >= field_mask.shape[0])] = 255
    return centers, sampled


def mask_inputs(data, rows, queries, onnx):
    if onnx is None:
        for i in queries:
            with Image.open(data / rows[i]["mask"]) as m:
                yield np.asarray(m.resize((32, 18), Image.Resampling.NEAREST)).ravel()
        return
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx), sess_options=options, providers=["CPUExecutionProvider"])
    model_input, = session.get_inputs()
    if model_input.type != "tensor(float)" or model_input.shape != [1, 3, 192, 320]:
        raise ValueError("Unexpected RoadNet ONNX float input contract")
    for i in queries:
        with Image.open(data / rows[i]["image"]) as image:
            frame = np.asarray(image.resize((320, 180), Image.Resampling.BILINEAR))
        frame = np.pad(frame, ((6, 6), (0, 0), (0, 0)), constant_values=114)
        logits, = session.run(None, {model_input.name: frame.transpose(2, 0, 1)[None].astype(np.float32) / 255})
        cls = logits[0].argmax(0).astype(np.uint8)
        cls = np.asarray(Image.fromarray(cls).resize((320, 192), Image.Resampling.NEAREST))[6:-6]
        yield np.asarray(Image.fromarray(cls).resize((32, 18), Image.Resampling.NEAREST)).ravel()


def semantic(templates, masks, device):
    centers, bank = templates
    obs = np.stack(list(masks))
    valid = obs <= 3
    counts = np.stack([(obs == i).sum(1) for i in range(4)], axis=1).clip(min=1)
    w = valid / np.sqrt(np.take_along_axis(counts, obs.clip(max=3), axis=1))
    q = torch.as_tensor(w, device=device, dtype=torch.float32)
    references = torch.as_tensor(bank, device=device)
    n, angles, pixels = bank.shape
    similarity = torch.zeros((len(obs), n * angles), device=device)
    for i in range(4):
        known = torch.as_tensor(obs == i, device=device)
        pattern = (references == i).reshape(-1, pixels).float()
        similarity += (q * known) @ pattern.T
    cost = 1 - similarity.reshape(len(obs), n, angles).amax(2) / q.sum(1, keepdim=True).clamp(min=1e-5)
    cand, score = [], []
    for line in cost.cpu().numpy():
        picked = separated_top5(line, centers, 100)
        cand.append(centers[picked])
        score.append(float(line[picked[1]] - line[picked[0]]))
    return np.asarray(cand), np.asarray(score)
