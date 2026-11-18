"""A shared RGB-to-map head scores camera XY centers, pooling internal map views."""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baselines import separated_top5
from model import network

EMBEDDING_DIM = 64
REFERENCE_CLASSES = (0, 1, 2, 3, 6, 255)
REFERENCE_DIM = len(REFERENCE_CLASSES) * 6 * 8


def reference_descriptors(bank):
    """Spatial histograms of the known map. These are inputs, never frame labels."""
    height, width = 18, 32
    regions = bank.reshape(*bank.shape[:2], 6, height // 6, 8, width // 8)
    histogram = np.stack([(regions == code).mean(axis=(3, 5), dtype=np.float32)
                          for code in REFERENCE_CLASSES], axis=2)
    return np.ascontiguousarray(histogram.reshape(*bank.shape[:2], REFERENCE_DIM))


class MapMatcher(nn.Module):
    """Predict one position score per map center with shared comparison weights."""

    def __init__(self):
        super().__init__()
        self.rgb = nn.Sequential(*list(network(0).children())[:-1], nn.Linear(64, EMBEDDING_DIM))
        self.reference = nn.Sequential(nn.Linear(REFERENCE_DIM, 96), nn.ReLU(),
                                       nn.Linear(96, EMBEDDING_DIM))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(16.)))

    def image_embedding(self, rgb):
        return F.normalize(self.rgb(rgb), dim=-1)

    def map_embedding(self, descriptor):
        return F.normalize(self.reference(descriptor), dim=-1)

    def score(self, rgb, map_embedding):
        shape = map_embedding.shape
        dots = self.image_embedding(rgb) @ map_embedding.reshape(-1, shape[-1]).T
        angles = dots.reshape(len(rgb), shape[0], shape[1])
        return angles.amax(-1) * self.log_temperature.exp().clamp(max=40)

    def forward(self, rgb, reference):
        return self.score(rgb, self.map_embedding(reference))


class FrozenMapHead(nn.Module):
    """Inference graph collapses internal reference views to XY logits."""

    def __init__(self, model, embedded_map):
        super().__init__()
        self.rgb = model.rgb
        self.centers, self.views, _ = embedded_map.shape
        scale = model.log_temperature.detach().exp().clamp(max=40)
        self.register_buffer("reference", embedded_map.flatten(0, 1).detach() * scale)

    def forward(self, rgb):
        features = F.normalize(self.rgb(rgb), dim=-1)
        scores = F.linear(features, self.reference)
        return scores.reshape(-1, self.centers, self.views).amax(-1)


def decode_scores(scores, centers, distance_mm=100):
    """Spatially distinct hypotheses and uncalibrated single/set score statistics."""
    if not np.isfinite(scores).all() or scores.shape[1] != len(centers):
        raise ValueError("Unexpected XY score tensor")
    choices = np.stack([separated_top5(-score, centers, distance_mm) for score in scores])
    # Summing the mass around a hypothesis avoids penalizing nearby grid cells
    # that describe the same location. Candidate sets use the union.
    p = np.exp(scores - scores.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    near = ((centers[:, None, :] - centers[None, :, :])**2).sum(-1) <= 75**2
    first_mass = np.sum(p * near[choices[:, 0]], axis=1)
    union_mass = np.sum(p * np.any(near[choices], axis=1), axis=1)
    return centers[choices], first_mass, union_mass


def analytic_masks_scores(templates, masks, device, batch=48):
    """Score coordinates from frozen RoadNet or oracle masks in bounded batches."""
    _, bank = templates
    observed = np.stack(list(masks))
    references = torch.as_tensor(bank, device=device)
    n, views, pixels = references.shape
    weights = np.stack([(observed == label).sum(1) for label in range(4)], axis=1).clip(min=1)
    weights = (observed <= 3) / np.sqrt(np.take_along_axis(weights, observed.clip(max=3), axis=1))
    result = np.zeros((len(observed), n * views), np.float32)
    for label in range(4):
        pattern = (references == label).reshape(-1, pixels).float()
        for first in range(0, len(observed), batch):
            end = first + batch
            q = torch.as_tensor(weights[first:end] * (observed[first:end] == label),
                                dtype=torch.float32, device=device)
            result[first:end] += (q @ pattern.T).cpu().numpy()
    score = result.reshape(-1, n, views).max(2)
    score /= np.maximum(weights.sum(1, keepdims=True), 1e-5)
    return score
