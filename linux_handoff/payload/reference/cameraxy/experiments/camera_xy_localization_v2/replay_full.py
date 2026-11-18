"""Replay selected held-out frames with the accepted CameraXY archive and draw one shortlist."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from contract import digest, load_rows, map_to_pixel, xy
from infer_map import MapLocalizer


def illustrate(path, image, field_preview, truth, candidates, label, errors):
    width = 960
    canvas = Image.new("RGB", (width, 590), "#202428")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 20), "CameraXY holdout | repeat-straight candidate set", fill="white")
    with Image.open(field_preview) as base:
        background = base.convert("RGB").resize((510, 510), Image.Resampling.BILINEAR)
    canvas.paste(background, (22, 55))
    with Image.open(image) as rgb:
        thumb = rgb.convert("RGB").resize((396, 223), Image.Resampling.BILINEAR)
    canvas.paste(thumb, (542, 55))

    def marker(pos, tag, color, radius):
        u, v = map_to_pixel(*pos, 510, 510, 3000)
        u, v = u + 22, v + 55
        draw.ellipse((u-radius, v-radius, u+radius, v+radius), fill=color, outline="white", width=1)
        draw.text((u+radius+3, v-8), tag, fill="white", stroke_width=1, stroke_fill="#222222")

    for i, center in reversed(list(enumerate(candidates, 1))):
        marker(center, str(i), "#bd5f42" if i == 1 else "#456ba4", 5)
    u,v = map_to_pixel(*truth,510,510,3000)
    u,v = u+22,v+55
    draw.line((u-8,v,u+8,v), fill="#318754", width=2)
    draw.line((u,v-8,u,v+8), fill="#318754", width=2)
    draw.text((542, 514), "Green cross: simulated optical-center truth", fill="#b9dfc7")
    draw.text((542, 290), label, fill="#d4d9dd")
    y = 318
    draw.text((542, y), "Rank   X (mm)   Y (mm)   XY error (mm)", fill="white")
    for i, (p, err) in enumerate(zip(candidates, errors), 1):
        y += 24
        draw.text((542, y), f"#{i:<3} {p[0]:>7.0f}  {p[1]:>7.0f}  {err:>9.1f}", fill="#d4d9dd")
    draw.text((542, y+33), "Primary location abstained by calibration.", fill="#f2b080")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    for p in ("data", "layout", "roadnet", "verified-output", "out"):
        parser.add_argument("--" + p, required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    root = args.verified_output
    result = json.loads((root / "result.json").read_text(encoding="utf8"))
    loc = MapLocalizer(root / "map-inference.pt", root / "metrics.json", result,
                       data=args.data, layout=args.layout, roadnet=args.roadnet)
    rows = {r["sample_id"]: r for r in load_rows(args.data / "manifest.jsonl")}
    manifest_path = args.data / "manifest.jsonl"
    if digest(manifest_path) != loc.manifest_sha:
        raise ValueError("Replay dataset manifest is not the trained dataset")
    pred_path = root / "predictions.npz"
    with np.load(pred_path, allow_pickle=False) as details:
        ids = details["sample_ids"]
        actual = np.asarray([xy(rows[name]) for name in ids])
        if not np.allclose(actual, details["truth_mm"], rtol=0, atol=.001):
            raise ValueError("Replay predictions differ from the XY label manifest")
        computed = details[loc.method + "_candidates_mm"]
        errors = np.linalg.norm(computed - actual[:, None, :], axis=-1)
    choices = [i for i, name in enumerate(ids)
               if rows[name]["split"] == "test" and rows[name]["ambiguity_region"] == "repeated_straight"
               and errors[i, 0] >= 500 and errors[i].min() <= 50]
    if not choices:
        raise ValueError("No archived shortlist witness satisfying the FAR/NEAR rule")
    chosen = choices[0]
    holdouts = [i for i, name in enumerate(ids) if rows[name]["split"] == "test"]
    witnesses = sorted({chosen, *[holdouts[i] for i in np.linspace(0, len(holdouts)-1, 5, dtype=int)]})
    records = []
    for i in witnesses:
        row = rows[ids[i]]
        image = args.data / row["image"]
        if digest(image) != row["image_sha256"]:
            raise ValueError("Replay frame bytes drift from its QA manifest")
        guess = loc.output(image)
        sites = [tuple(s[k] for k in ("camera_x_mm", "camera_y_mm")) for s in guess["top_k_camera_xy"]]
        exact = np.array_equal(sites, computed[i])
        overlap = len(set(sites) & set(map(tuple, computed[i])))
        records.append(dict(sample_id=str(ids[i]), image_sha256=row["image_sha256"], exact_rank_order=bool(exact),
                            exact_position_overlap=overlap, truth_mm=actual[i].tolist(),
                            holder_prediction=guess, cloud_candidates_mm=computed[i].tolist()))
    args.out.mkdir()
    path = args.out / "full-shortlist.png"
    illustrate(path, args.data / rows[ids[chosen]]["image"], args.data / "field_preview.png",
               actual[chosen], computed[chosen], str(ids[chosen]), errors[chosen])
    report = dict(status="HOST_REPLAY_COMPLETE", mode="SMALL_HELD_OUT_REPLAY",
                  original_manifest_sha256=loc.manifest_sha, prediction_bundle_sha256=digest(pred_path),
                  method=loc.method, frames=len(records), exact_order_count=sum(x["exact_rank_order"] for x in records),
                  plot_sha256=digest(path), plot_sample=str(ids[chosen]),
                  plot_first_error_mm=float(errors[chosen,0]), plot_nearest_candidate_mm=float(errors[chosen].min()),
                  records=records)
    (args.out / "replay.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
