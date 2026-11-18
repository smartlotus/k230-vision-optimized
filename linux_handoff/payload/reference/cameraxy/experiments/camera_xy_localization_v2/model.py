"""CameraXY-only model heads and train/inference pixel preprocessing."""
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

SPAN_MM = 2400
W, H = 160, 96


def pixels(image):
    resized = image.convert("RGB").resize((W, 90), Image.Resampling.BILINEAR)
    return np.asarray(ImageOps.expand(resized, (0, 3, 0, 3), fill=(114, 114, 114))).transpose(2, 0, 1)


def network(cell_mm):
    layers = []
    for begin, end in ((3, 16), (16, 24), (24, 32), (32, 48)):
        layers.extend((nn.Conv2d(begin, end, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(end), nn.ReLU()))
    outputs = 2 if cell_mm == 0 else (SPAN_MM // cell_mm)**2 * 3
    return nn.Sequential(*layers, nn.AvgPool2d(2), nn.Flatten(), nn.Linear(720, 64),
                         nn.ReLU(), nn.Linear(64, outputs))


def grid_targets(target_mm, cell_mm):
    n = SPAN_MM // cell_mm
    pos = (target_mm + SPAN_MM / 2) / cell_mm
    indices = pos.long().clamp(0, n - 1)
    return indices[:, 1] * n + indices[:, 0], (pos - indices - .5) * 2


def grid_heads(logits):
    cells = logits.shape[-1] // 3
    return logits[:, :cells], logits[:, cells:].reshape(-1, cells, 2)


def grid_decode(logits, cell_mm, k=5):
    n = SPAN_MM // cell_mm
    cells, offsets = grid_heads(logits)
    prob, idx = cells.softmax(-1).topk(k, dim=-1)
    origins = torch.stack((idx % n, idx // n), -1)
    residual = torch.gather(offsets, 1, idx.unsqueeze(-1).expand(-1, -1, 2)).clamp(-1, 1)
    return (origins + .5 + residual / 2) * cell_mm - SPAN_MM / 2, prob, idx


def predicted(model, x, cell_mm):
    if cell_mm:
        candidates, prob, grid = grid_decode(model(x), cell_mm)
        return candidates, prob[:, 0], grid
    a = model(x).clamp(-1, 1) * (SPAN_MM / 2)
    b = model(torch.flip(x, dims=(-2, -1))).clamp(-1, 1) * (SPAN_MM / 2)
    return ((a + b) / 2)[:, None, :], -torch.linalg.vector_norm(a - b, dim=1), None
