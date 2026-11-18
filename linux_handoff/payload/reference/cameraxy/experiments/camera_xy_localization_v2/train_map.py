"""Full-stage training of the shared RGB-to-map CameraXY matcher."""
import argparse
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from baselines import mask_inputs, template_map
from contract import TARGETS, block_and_split, digest, xy
from map_matcher import (FrozenMapHead, MapMatcher, analytic_masks_scores, decode_scores,
                         reference_descriptors)
from model import pixels
from train import calibrate, calibrated, metrics, read_dataset, save_json

BATCH = 64
NEARBY_CENTERS_PER_FRAME = 4
RANDOM_REFERENCE_NEGATIVES = 96
XY_SUPERVISION_SIGMA_MM = 38
SEMANTIC_LOGIT_SCALE = 18


def split_indices(rows, seed):
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val" and r["variant"] == 0])
    test = np.array([i for i, r in enumerate(rows) if r["split"] == "test"])
    val = np.random.default_rng(seed).permutation(val)
    tune, cal = np.array_split(val, 2)
    return train, tune, cal, test


def read_images(data, rows, device):
    sample = np.empty((len(rows), 3, 96, 160), dtype=np.uint8)
    for i, row in enumerate(rows):
        with Image.open(data / row["image"]) as im:
            sample[i] = pixels(im)
    return torch.from_numpy(sample).to(device)


def train_reference(centers, targets, train):
    allowed = np.flatnonzero([block_and_split(x, y)[1] == "train" for x, y in centers])
    centers_train = centers[allowed]
    d2 = ((targets[train, None] - centers_train[None])**2).sum(-1)
    nearby = np.argpartition(d2, NEARBY_CENTERS_PER_FRAME - 1, axis=1)[:, :NEARBY_CENTERS_PER_FRAME]
    if d2.min(1).max() > 100**2:
        raise ValueError("A training optical center lacks nearby training map references")
    return allowed, allowed[nearby]


def soft_xy_target(positions, candidate_coordinates):
    d2 = (positions[:, None, :] - candidate_coordinates[None, :, :]).square().sum(-1)
    return torch.softmax(-d2 / (2 * XY_SUPERVISION_SIGMA_MM**2), dim=1)


@torch.inference_mode()
def predict(net, imgs, idx, reference):
    net.eval()
    embedded = net.map_embedding(reference)
    return np.concatenate([net.score(imgs[p].float() / 255, embedded).cpu().numpy()
                           for p in np.array_split(idx, (len(idx) + BATCH-1)//BATCH)])


def latest_checkpoint(out, net, optimizer, scheduler, epoch, updates, rng, manifest):
    path = out / "map-last.pending"
    torch.save(dict(weight=net.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), targets=TARGETS, epoch=epoch,
                    updates=updates, dataset_manifest_sha256=manifest,
                    numpy_state=rng.bit_generator.state, torch_state=torch.get_rng_state(),
                    cuda_state=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    python_state=random.getstate()), path)
    path.replace(out / "map-last.pt")


def fit(imgs, position_mm, centers, refs, train, tune, epochs, out, seed, manifest):
    net = MapMatcher().to(imgs.device)
    opt = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=.002)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=.00015)
    rng = np.random.default_rng(seed + 1)
    center_tensor = torch.as_tensor(centers, dtype=torch.float32, device=imgs.device)
    target_tensor = torch.as_tensor(position_mm, dtype=torch.float32, device=imgs.device)
    allowed, nearby = train_reference(centers, position_mm, train)
    best_key = (-1., -np.inf)
    history = []
    updates = 0
    for epoch in range(1, epochs+1):
        net.train()
        total = 0.
        order = rng.permutation(len(train))
        for start in range(0, len(train), BATCH):
            ii = order[start:start+BATCH]
            indices = train[ii]
            distractors = rng.choice(allowed, min(len(allowed), RANDOM_REFERENCE_NEGATIVES), replace=False)
            candidates = np.unique(np.concatenate([nearby[ii].ravel(), distractors]))
            q = target_tensor[indices]
            y = soft_xy_target(q, center_tensor[candidates])
            logits = net(imgs[indices].float() / 255, refs[candidates])
            loss = -(y * F.log_softmax(logits, dim=1)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(indices)
            updates += 1
        scheduler.step()
        tune_scores = predict(net, imgs, tune, refs)
        candidates, _, _ = decode_scores(tune_scores, centers)
        error = np.linalg.norm(candidates - position_mm[tune, None], axis=2)
        top = float((error.min(1) <= 50).mean())
        median = float(np.median(error[:, 0]))
        record = dict(epoch=epoch, updates=updates, training_loss=total/len(train),
                      validation_top5_within50=top, validation_top1_median_mm=median,
                      lr=opt.param_groups[0]["lr"])
        history.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
        if (top, -median) > best_key:
            best_key = (top, -median)
            weights = {name: value.detach().cpu() for name, value in net.state_dict().items()}
            torch.save(dict(weights=weights, targets=TARGETS, dataset_manifest_sha256=manifest,
                            selected_epoch=epoch, validation=record), out / "map-best.pt")
        latest_checkpoint(out, net, opt, scheduler, epoch, updates, rng, manifest)
        save_json(out / "history.json", history)
    snapshot = torch.load(out / "map-best.pt", weights_only=True, map_location=imgs.device)
    net.load_state_dict(snapshot["weights"])
    net.eval()
    return net, snapshot["selected_epoch"], updates


def logprob(scores):
    offset = scores - scores.max(1, keepdims=True)
    return offset - np.log(np.exp(offset).sum(1, keepdims=True))


def fused_scores(learned, semantic):
    return np.logaddexp(logprob(learned), logprob(semantic)) - np.log(2.)


def candidate_set_metrics(truth, candidates, set_score, cal):
    min_error = np.linalg.norm(candidates - truth[:, None], axis=2).min(1)
    confidence, accepted = calibrated(set_score, cal)
    success = min_error <= 50
    idx = np.argsort(-set_score, kind="stable")
    curve = []
    for coverage in (.10, .25, .5, .75, 1.):
        i = idx[:max(1, int(len(idx) * coverage))]
        curve.append(dict(frames=len(i), miss50_rate=float((~success[i]).mean()),
                          min_median_mm=float(np.median(min_error[i]))))
    return dict(frames=len(truth), success_within50=float(success.mean()),
                brier50=float(np.mean((confidence - success)**2)), average_confidence=float(confidence.mean()),
                min_median_mm=float(np.median(min_error)), min_p95_mm=float(np.quantile(min_error, .95)),
                threshold_coverage=float(accepted.mean()), risk_coverage=curve)


def evaluate(method_scores, centers, rows, order, tune_len, cal_len, truth):
    selected_rows = [rows[i] for i in order]
    indices = np.arange(len(order))
    val_tune = indices < tune_len
    val_cal = (indices >= tune_len) & (indices < tune_len + cal_len)
    splits = np.array([r["split"] for r in selected_rows])
    variants = np.array([r["variant"] for r in selected_rows])
    regions = np.array([r["ambiguity_region"] for r in selected_rows])
    subsets = dict(validation_tune=val_tune, validation_calibration=val_cal,
                   spatial_holdout=(splits == "test") & (variants < 2),
                   appearance_holdout=variants == 2,
                   appearance_pair_nominal=(splits == "test") & (variants == 0),
                   repeated_straight_holdout=(splits == "test") & (regions == "repeated_straight"))
    output = {}
    evidence = dict(sample_ids=np.array([row["sample_id"] for row in selected_rows]), truth_mm=truth)
    for name, values in method_scores.items():
        cand, first, set_score = decode_scores(values, centers)
        d = np.linalg.norm(cand - truth[:, None], axis=2)
        cal_first = calibrate(first[val_cal], d[val_cal, 0] <= 50)
        cal_set = calibrate(set_score[val_cal], d[val_cal].min(1) <= 50)
        primary = {group: metrics(cand[s], truth[s], first[s], cal_first) for group, s in subsets.items()}
        shortlist = {group: candidate_set_metrics(truth[s], cand[s], set_score[s], cal_set)
                     for group, s in subsets.items() if s.any()}
        output[name] = dict(metrics=primary, shortlist=shortlist, primary_confidence=cal_first,
                            set_confidence=cal_set)
        evidence[name + "_candidates_mm"] = cand
        evidence[name + "_primary_score"] = first
        evidence[name + "_set_score"] = set_score
    return output, evidence


def run(data, onnx, layout_path, out, device, epochs, seed):
    layout = json.loads(layout_path.read_text(encoding="utf8"))
    start = time.monotonic()
    qa, rows = read_dataset(data)
    manifest = qa["manifest_sha256"]
    run_config = json.loads((data / "run.json").read_text(encoding="utf8"))
    position = np.asarray([xy(row) for row in rows], dtype=np.float32)
    train, tune, cal, test = split_indices(rows, seed)
    order = np.concatenate((tune, cal, test))
    torch.set_num_threads(4)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    save_json(out / "environment.json", dict(
        python=platform.python_version(), numpy=np.__version__, torch=str(torch.__version__),
        device=device, gpu=torch.cuda.get_device_name() if device == "cuda" else None,
        cuda=torch.version.cuda, targets=TARGETS, seed=seed, epochs=epochs, batch=BATCH,
        dataset_manifest_sha256=manifest, roadnet_onnx_sha256=digest(onnx), recipe="xy-map-v1",
        full_replay_automatic=False))
    with Image.open(data / "field_mask.png") as im:
        templates = template_map(np.asarray(im), layout, run_config["margin"])
    centers, bank = templates
    refs = torch.from_numpy(reference_descriptors(bank)).to(device)
    inputs = read_images(data, rows, device)
    print(f"TRAIN_FRAMES={len(train)} VALIDATION_TUNE={len(tune)} CALIBRATE={len(cal)} "
          f"TEST_FRAMES={len(test)} REFERENCE_SITES={len(centers)}", flush=True)
    net, best_epoch, updates = fit(inputs, position, centers, refs, train, tune, epochs, out, seed, manifest)
    learned = predict(net, inputs, order, refs)
    with torch.inference_mode():
        frozen = FrozenMapHead(net, net.map_embedding(refs)).eval()
        witness = frozen(inputs[order[:4]].float()/255).cpu().numpy()
    if not np.allclose(witness, learned[:4], atol=1e-4):
        raise ValueError("XY-only inference graph differs from training head")
    prefix = dict(centers_mm=centers.tolist(), orientations_collapsed=True, targets=TARGETS,
                  dataset_manifest_sha256=manifest, selected_epoch=best_epoch)
    torch.save(dict(state=frozen.state_dict(), reference_views=bank.shape[1], **prefix),
               out / "map-inference.pt")
    compare = {}
    for name, model in (("semantic_roadnet", onnx), ("semantic_oracle", None)):
        print(f"EVAL {name} frames={len(order)}", flush=True)
        score = analytic_masks_scores(templates, mask_inputs(data, rows, order, model), device)
        compare[name] = score * SEMANTIC_LOGIT_SCALE
    scores = dict(learned_map=learned, fused_map=fused_scores(learned, compare["semantic_roadnet"]), **compare)
    results, details = evaluate(scores, centers, rows, order, len(tune), len(cal), position[order])
    save_json(out / "metrics.json", results)
    np.savez_compressed(out / "predictions.npz", **details)
    selected = max(("learned_map", "fused_map", "semantic_roadnet"), key=lambda name: (
        results[name]["shortlist"]["validation_tune"]["success_within50"],
        -results[name]["metrics"]["validation_tune"]["median_mm"]))
    # The original study's 20/50 mm objective and confidence policy remain intact.
    accepted = [results[selected]["metrics"][s] for s in ("spatial_holdout", "appearance_holdout")]
    host_pass = all(m["coverage"] >= .25 and m["accepted_median_mm"] <= 20
                    and m["accepted_p95_mm"] <= 50 for m in accepted)
    detail = details[selected + "_candidates_mm"]
    test_index = np.where(np.array([rows[i]["split"] == "test" for i in order]))[0]
    spread = np.linalg.norm(detail[:, 0:1] - detail, axis=2).max(1)
    selected_rows = sorted(test_index, key=lambda i: (-spread[i], str(details["sample_ids"][i])))[:200]
    save_json(out / "ambiguity_pairs.json", [dict(
        sample_id=str(details["sample_ids"][i]), candidate_span_mm=float(spread[i]),
        top_xy_mm=detail[i].tolist(), true_xy_mm=details["truth_mm"][i].tolist())
        for i in selected_rows])
    report = dict(status="COMPUTE_COMPLETE", targets=TARGETS, dataset_manifest_sha256=manifest,
                  selected_method_by_validation_top5=selected, selected_epoch=best_epoch,
                  inference_sha256=digest(out / "map-inference.pt"),
                  calibration_metrics_sha256=digest(out / "metrics.json"),
                  semantic_inputs=dict(field_mask_sha256=digest(data / "field_mask.png"),
                                       layout_sha256=digest(layout_path), roadnet_sha256=digest(onnx),
                                       margin_mm=run_config["margin"]),
                  updates=updates, scientific_goal=dict(median_mm=20, p95_mm=50,
                                                        accepted_coverage_min=.25),
                  full_gate="PASS" if host_pass else "HOLD", elapsed_seconds=time.monotonic()-start)
    save_json(out / "result.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("data", "roadnet", "layout", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=37121)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("at least one epoch is required")
    args.out.mkdir()
    run(args.data, args.roadnet, args.layout, args.out, args.device, args.epochs, args.seed)
