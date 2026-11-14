"""Convert rendered face masks and projected circle boundaries into audited labels."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from geometry import ellipse_from_points
from cloud_store import atomic_json, sha


def audit(root):
    root = Path(root)
    rows, issues = [], []
    for path in sorted(root.glob("scene-*/labels_raw.json")):
        raw = json.loads(path.read_text())
        directory = path.parent
        rgb = cv2.imread(str(directory/"rgb.png"))
        if rgb is None or rgb.shape != (360,640,3):
            raise ValueError("RGB output is not the declared 640x360 frame")
        total_visible = np.zeros((360,640), np.uint8)
        for item in raw["instances"]:
            index = item["instance_id"]
            visible = cv2.imread(str(directory/item["top_face_visible_mask"]), 0)
            if visible is None or visible.shape != (360,640):
                raise ValueError("Missing or invalid face-level visible mask")
            visible = (visible >= 128).astype(np.uint8)
            if np.any(total_visible & visible):
                raise ValueError("An image pixel belongs to multiple visible top faces")
            total_visible |= visible
            boundary = np.asarray(item.pop("top_boundary_xy"), np.float32)
            full = np.zeros_like(visible)
            cv2.fillPoly(full, [np.round(boundary).astype(np.int32)], 1)
            allowance = cv2.dilate(full, np.ones((3,3),np.uint8))
            outside = int((visible & (1-allowance)).sum())
            if outside:
                issues.append({"scene": directory.name, "instance_id": index,
                               "reason": "VISIBLE_MASK_OUTSIDE_PROJECTED_TOP", "pixels": outside})
            ellipse = ellipse_from_points(boundary)
            area = float(cv2.contourArea(boundary))
            visible_pixels = int(visible.sum())
            oncanvas = int(full.sum())
            yy, xx = np.where(visible)
            truncated = bool(np.any(boundary[:,0] < 0) or np.any(boundary[:,0] >= 640)
                or np.any(boundary[:,1] < 0) or np.any(boundary[:,1] >= 360))
            fraction = min(1., visible_pixels/max(area,1.))
            ellipse_mask = np.zeros_like(visible)
            cv2.ellipse(ellipse_mask, tuple(np.round(ellipse["center"]).astype(int)),
                tuple(np.maximum(1,np.round([ellipse["semi_major"],ellipse["semi_minor"]])).astype(int)),
                ellipse["orientation_deg"],0,360,1,-1)
            fit_iou = float((ellipse_mask & full).sum()/max(1,(ellipse_mask | full).sum()))
            item.update(top_face_full_mask=f"full-{index}.png", center_xy=ellipse["center"],
                ellipse_center=ellipse["center"], semi_major=ellipse["semi_major"],
                semi_minor=ellipse["semi_minor"], orientation_deg=ellipse["orientation_deg"],
                orientation_valid=ellipse["angle_valid"], visible_fraction=fraction,
                occlusion_level=1-min(1.,visible_pixels/max(1,oncanvas)), truncated=truncated,
                geometry_valid=visible_pixels >= 16 and fraction >= .4 and not truncated,
                bbox_xyxy=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if len(xx) else None,
                full_projection_area_px=area, full_oncanvas_area_px=oncanvas,
                visible_area_px=visible_pixels, ellipse_raster_iou=fit_iou,
                camera_pose=raw["camera_pose_world_from_cv"],
                camera_intrinsics=raw["camera_intrinsics"], scene_group_id=raw["scene_group_id"],
                random_seed=raw["random_seed"])
            if fit_iou < .9:
                issues.append({"scene": directory.name,"instance_id":index,"reason":"ELLIPSE_FIT_IOU"})
            cv2.imwrite(str(directory/f"full-{index}.png"), full*255)
            cv2.ellipse(rgb, tuple(np.round(ellipse["center"]).astype(int)),
                tuple(np.maximum(1,np.round([ellipse["semi_major"],ellipse["semi_minor"]])).astype(int)),
                ellipse["orientation_deg"],0,360,(0,255,255),1)
        atomic_json(directory/"labels.json", raw)
        cv2.imwrite(str(directory/"overlay.jpg"), rgb)
        rows.append({"scene":directory.name,"instances":len(raw["instances"]),
                     "rgb_sha256":sha(directory/"rgb.png"),"label_sha256":sha(directory/"labels.json")})
    if not rows:
        raise ValueError("No rendered scenes found")
    report={"schema":"topface-label-audit-v1","status":"PASS" if not issues else "FAIL",
            "images":len(rows),"instances":sum(r["instances"] for r in rows),
            "issues":issues,"scenes":rows,"real_geometry_accuracy_verified":False}
    atomic_json(root/"AUDIT.json",report)
    if issues:
        raise RuntimeError("Rendered mask/geometry consistency audit failed")
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.root)))
