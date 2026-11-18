"""Kaggle CameraXY experiments with grouped calibration and spatial evaluation."""
import argparse
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
from PIL import Image
import torch
from torch import nn

from baselines import mask_inputs, retrieval, rgb_descriptors, semantic, template_map
from contract import TARGETS, digest, load_rows, xy
from model import grid_heads, grid_targets, network, pixels, predicted


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def read_dataset(path):
    qa = json.loads((path / "qa.json").read_text(encoding="utf8"))
    rows = load_rows(path / "manifest.jsonl")
    if qa["status"] != "PASS" or qa["manifest_sha256"] != digest(path / "manifest.jsonl"):
        raise ValueError("Unverified dataset manifest")
    if qa["targets"] != list(TARGETS) or len(rows) != qa["frames"]:
        raise ValueError("Unexpected dataset label contract")
    return qa, rows


def loss_for(net, x, target, cell):
    raw = net(x)
    if not cell:
        return nn.functional.smooth_l1_loss(raw, target / 1200, beta=.07)
    logits, residual = grid_heads(raw)
    classes, offsets = grid_targets(target, cell)
    chosen = residual[torch.arange(len(x), device=x.device), classes]
    return nn.functional.cross_entropy(logits, classes) + nn.functional.smooth_l1_loss(chosen, offsets)


@torch.inference_mode()
def predict(net, images, ids, cell, batch=64):
    net.eval()
    candidates, scores, grids = [], [], []
    for idx in np.array_split(ids, math.ceil(len(ids) / batch)):
        cand, score, grid = predicted(net, images[idx].float() / 255, cell)
        candidates.append(cand.cpu().numpy())
        scores.append(score.cpu().numpy())
        if grid is not None:
            grids.append(grid.cpu().numpy())
    return np.concatenate(candidates), np.concatenate(scores), np.concatenate(grids) if grids else None


def fit(images, targets, train_ids, tune_ids, cell, epochs, seed, out, name, manifest_sha256):
    random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = network(cell).to(images.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=.0001)
    history, best_state, best_val = [], None, math.inf
    updates = 0
    for epoch in range(epochs):
        model.train()
        indices = rng.permutation(train_ids)
        total = 0.
        for start in range(0, len(indices), 32):
            idx = indices[start:start + 32]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, images[idx].float() / 255, targets[idx], cell)
            loss.backward()
            optimizer.step()
            updates += 1
            total += loss.item() * len(idx)
        scheduler.step()
        if out is None:
            model.eval()
            with torch.inference_mode():
                fitted = model(images[tune_ids].float() / 255).cpu().numpy()
            cand = fitted[:, None, :] * 1200
        else:
            cand, _, _ = predict(model, images, tune_ids, cell)
        err = np.linalg.norm(cand[:, 0] - targets[tune_ids].cpu().numpy(), axis=1)
        median = float(np.median(err))
        history.append(dict(epoch=epoch + 1, updates=updates, loss=total / len(indices),
                            lr=optimizer.param_groups[0]["lr"], val_median_mm=median))
        if median < best_val:
            best_val = median
            best_state = dict(
                weights={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                cell_mm=cell, selected_epoch=epoch + 1, dataset_manifest_sha256=manifest_sha256,
                targets=TARGETS, val_median_mm=median)
        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"{name} epoch {epoch+1}/{epochs} val median {median:.2f} mm", flush=True)
    if out is not None:
        torch.save(best_state, out / f"{name}-best.pt")
        torch.save(dict(weights=model.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), numpy_rng=rng.bit_generator.state,
                        torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                        cuda_rng=torch.cuda.get_rng_state_all() if images.is_cuda else [],
                        epoch=epochs, updates=updates, cell_mm=cell,
                        dataset_manifest_sha256=manifest_sha256, targets=TARGETS),
                   out / f"{name}-last.pt")
    model.load_state_dict(best_state["weights"])
    return model, dict(selected_epoch=best_state["selected_epoch"], updates=updates,
                       parameters=sum(p.numel() for p in model.parameters()), history=history)


def wilson_lower(success, total):
    if total == 0:
        return 0.
    p = success / total
    z = 1.64485362695
    return (p + z*z/(2*total) - z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))) / (1+z*z/total)


def calibrate(raw, good):
    edges = np.unique(np.quantile(raw, [.2, .4, .6, .8])) if len(raw) >= 40 else np.array([])
    index = np.searchsorted(edges, raw, side="left")
    total = np.bincount(index, minlength=len(edges)+1)
    successes = np.bincount(index, weights=good.astype(float), minlength=len(edges)+1)
    probability = (successes + 1) / (total + 2)
    threshold = None
    for point in np.unique(np.quantile(raw, [0, .5, .75, .9])):
        keep = raw >= point
        if keep.sum() >= 10 and wilson_lower(good[keep].sum(), keep.sum()) >= .8:
            threshold = float(point)
            break
    return dict(bin_edges=edges.tolist(), probabilities=probability.tolist(),
                threshold=threshold, calibration_groups=len(raw), within50_count=int(good.sum()))


def calibrated(raw, cal):
    confidence = np.asarray(cal["probabilities"])[np.searchsorted(cal["bin_edges"], raw, side="left")]
    threshold = cal["threshold"]
    accepted = np.zeros(len(raw), bool) if threshold is None else raw >= threshold
    return confidence, accepted


def metrics(cand, truth, raw, cal):
    if not len(truth):
        return {"frames": 0}
    residual = cand - truth[:, None]
    error = np.linalg.norm(residual, axis=2)
    first = error[:, 0]
    confidence, accept = calibrated(raw, cal)
    good = first <= 50
    order = np.argsort(-raw, kind="stable")
    curve = []
    for coverage in (.10, .25, .50, .75, 1.):
        idx = order[:math.ceil(coverage * len(first))]
        curve.append(dict(frames=len(idx), coverage=len(idx)/len(first),
                          mean_mm=float(first[idx].mean()), failure50_rate=float((~good[idx]).mean())))
    return dict(
        frames=len(first), mean_mm=float(first.mean()), median_mm=float(np.median(first)),
        p95_mm=float(np.quantile(first, .95)), xy_mae_mm=np.abs(residual[:, 0]).mean(0).tolist(),
        within_mm={str(mm): float((first <= mm).mean()) for mm in (10, 20, 30, 50)},
        best_of_k_within50={str(k): float((error[:, :k].min(1) <= 50).mean())
                            for k in (1, 3, 5) if error.shape[1] >= k},
        brier50=float(np.mean((confidence - good)**2)), risk_coverage=curve,
        accepted_count=int(accept.sum()), coverage=float(accept.mean()),
        accepted_median_mm=float(np.median(first[accept])) if accept.any() else None,
        accepted_p95_mm=float(np.quantile(first[accept], .95)) if accept.any() else None,
    )


def run(data, roadnet, layout, out, epochs, seed, device):
    started = time.monotonic()
    qa, rows = read_dataset(data)
    receipt = qa["manifest_sha256"]
    save_json(out / "environment.json", dict(
        python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
        gpu=torch.cuda.get_device_name(0) if device == "cuda" else None, seed=seed,
        cuda=torch.version.cuda, targets=TARGETS, manifest_sha256=receipt,
        roadnet_onnx_sha256=digest(roadnet) if roadnet else None, epochs=epochs,
        confidence_event="first candidate within 50 mm", full_replay_automatic=False))
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    all_xy = np.array([xy(r) for r in rows], dtype=np.float32)
    all_imgs = []
    for row in rows:
        with Image.open(data / row["image"]) as img:
            all_imgs.append(pixels(img))
    all_imgs = np.stack(all_imgs)
    imgs = torch.from_numpy(all_imgs).to(device)
    targets = torch.from_numpy(all_xy).to(device)
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    val = np.array([i for i, r in enumerate(rows) if r["split"] == "val" and r["variant"] == 0])
    test = np.array([i for i, r in enumerate(rows) if r["split"] == "test"])
    rng = np.random.default_rng(seed)
    val = rng.permutation(val)
    tune = val[:len(val)//2]
    cal_rows = val[len(val)//2:]
    queries = np.concatenate((tune, cal_rows, test))
    truth = all_xy[queries]
    subsets = {
        "validation_tune": np.isin(queries, tune),
        "validation_calibration": np.isin(queries, cal_rows),
        "spatial_holdout": np.array([rows[i]["split"] == "test" and rows[i]["variant"] < 2 for i in queries]),
        "appearance_holdout": np.array([rows[i]["variant"] == 2 for i in queries]),
        "appearance_pair_nominal": np.array([rows[i]["split"] == "test" and rows[i]["variant"] == 0 for i in queries]),
        "repeated_straight_holdout": np.array([rows[i]["split"] == "test" and
             rows[i]["ambiguity_region"] == "repeated_straight" for i in queries]),
    }
    calibration_mask = subsets["validation_calibration"]
    report = {}
    detailed = {"sample_ids": np.array([rows[i]["sample_id"] for i in queries]), "truth_mm": truth}
    small = rng.choice(train[::2], size=16, replace=False)
    _, overfit = fit(imgs, targets, small, small, 0, 160, seed, None, "overfit16", receipt)
    save_json(out / "overfit16.json", overfit)

    def record(name, cand, raw, lineage=None, grid=None, cell=None):
        good = np.linalg.norm(cand[:, 0] - truth, axis=1) <= 50
        cal = calibrate(raw[calibration_mask], good[calibration_mask])
        scores = {split: metrics(cand[mask], truth[mask], raw[mask], cal)
                  for split, mask in subsets.items()}
        if grid is not None:
            correct, _ = grid_targets(torch.from_numpy(truth), cell)
            for split, mask in subsets.items():
                scores[split]["grid_accuracy"] = {
                    str(k): float(np.mean(np.any(grid[mask, :k] == correct.numpy()[mask, None], axis=1)))
                    for k in (1, 3, 5)} if mask.any() else {}
        report[name] = dict(metrics=scores, confidence=cal, lineage=lineage)
        detailed[name + "_candidates_mm"] = cand
        detailed[name + "_raw_confidence"] = raw
        save_json(out / "metrics.json", report)
        print(f"{name}: evaluated {len(cand)} rows", flush=True)

    cells = (0, 100) if qa["pose_groups"] == 100 else (0, 50, 100, 150)
    for cell in cells:
        name = "direct" if cell == 0 else f"grid{cell}"
        model, lineage = fit(imgs, targets, train, tune, cell, epochs, seed, out, name, receipt)
        cand, raw, grid = predict(model, imgs, queries, cell)
        record(name, cand, raw, lineage, grid, cell)

    feats = rgb_descriptors(all_imgs)
    cand, raw, pairs = retrieval(feats, all_xy, train, queries)
    record("retrieval", cand, raw)
    ambiguity = []
    for i, j, d, dist in sorted(pairs, key=lambda item: item[2]):
        ambiguity.append(dict(sample_id=rows[i]["sample_id"], reference_id=rows[j]["sample_id"],
                              xy_mm=all_xy[i].tolist(), reference_xy_mm=all_xy[j].tolist(),
                              descriptor_mse=d, distance_mm=dist,
                              region=rows[i]["ambiguity_region"], split=rows[i]["split"]))
    save_json(out / "ambiguity_pairs.json", ambiguity)
    with Image.open(data / "field_mask.png") as img:
        margin = json.loads((data / "run.json").read_text(encoding="utf8"))["margin"]
        templates = template_map(np.asarray(img), layout, margin)
    for name, source in (("semantic_oracle", None), ("semantic_roadnet", roadnet)):
        if name == "semantic_roadnet" and source is None:
            continue
        cand, raw = semantic(templates, mask_inputs(data, rows, queries, source), device)
        record(name, cand, raw)
    np.savez_compressed(out / "predictions.npz", **detailed)
    deployable = ["direct", *(f"grid{cell}" for cell in cells if cell)]
    best = min(deployable, key=lambda name: report[name]["metrics"]["validation_tune"]["median_mm"])
    holdouts = [report[best]["metrics"][split] for split in ("spatial_holdout", "appearance_holdout")]
    # A full run needs evidence of useful reliable coverage at the nominal
    # localization objective. This first research threshold is configurable in future runs.
    full = all(m["coverage"] >= .25 and m["accepted_median_mm"] <= 20 and m["accepted_p95_mm"] <= 50
               for m in holdouts)
    result = dict(
        status="COMPUTE_COMPLETE", dataset_manifest_sha256=receipt, targets=TARGETS,
        model_selected_by_validation_median=best, full_gate="PASS" if full else "HOLD",
        scientific_goal=dict(median_mm=20, p95_mm=50, accepted_coverage_min=.25),
        elapsed_seconds=time.monotonic() - started,
    )
    save_json(out / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--roadnet", type=Path)
    p.add_argument("--layout", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--seed", type=int, default=37121)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = p.parse_args()
    if a.device == "cuda":
        assert torch.cuda.is_available(), "Kaggle CUDA runtime unavailable"
        assert torch.ones(2, device="cuda").sum().item() == 2
    a.out.mkdir()
    print(json.dumps(run(a.data, a.roadnet, json.loads(a.layout.read_text(encoding="utf8")),
                         a.out, a.epochs, a.seed, a.device)), flush=True)


if __name__ == "__main__":
    main()
