"""Fit normalized DLT on a PC; require independent validation before export."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def _points(values, minimum):
    import numpy as np
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError("Expected at least %d 2D points" % minimum)
    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Non-finite or duplicate points")
    if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
        raise ValueError("Collinear points")
    return points


def fit_homography(pixels, millimeters):
    import numpy as np
    src, dst = _points(pixels, 4), _points(millimeters, 4)
    if len(src) != len(dst):
        raise ValueError("Mismatched point counts")

    def normalize(points):
        center = points.mean(axis=0)
        scale = math.sqrt(2) / np.linalg.norm(points - center, axis=1).mean()
        matrix = np.array([[scale, 0, -scale * center[0]],
                           [0, scale, -scale * center[1]], [0, 0, 1]])
        normalized = (matrix @ np.column_stack((points, np.ones(len(points)))).T).T
        return normalized[:, :2], matrix

    a, ta = normalize(src)
    b, tb = normalize(dst)
    rows = []
    for (u, v), (x, y) in zip(a, b):
        rows.extend(([-u, -v, -1, 0, 0, 0, x*u, x*v, x],
                     [0, 0, 0, -u, -v, -1, y*u, y*v, y]))
    rows = np.asarray(rows)
    if np.linalg.matrix_rank(rows) < 8:
        raise ValueError("Degenerate correspondences")
    _, _, vh = np.linalg.svd(rows, full_matrices=True)
    h = np.linalg.inv(tb) @ vh[-1].reshape(3, 3) @ ta
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("Unusable projection scale")
    h /= h[2, 2]
    if not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
        raise ValueError("Singular homography")
    return h.tolist()


def project(h, pixels):
    result = []
    for u, v in pixels:
        denominator = h[2][0]*u + h[2][1]*v + h[2][2]
        if not math.isfinite(denominator) or abs(denominator) < 1e-12:
            raise ValueError("Invalid projective denominator")
        point = [(h[0][0]*u+h[0][1]*v+h[0][2])/denominator,
                 (h[1][0]*u+h[1][1]*v+h[1][2])/denominator]
        if not all(math.isfinite(x) for x in point):
            raise ValueError("Non-finite projection")
        result.append(point)
    return result


def calibrate(data, max_error_mm):
    if not math.isfinite(max_error_mm) or max_error_mm <= 0:
        raise ValueError("Specify a positive finite error tolerance")
    for key in ("calibration_id", "camera_id", "plane", "station", "pose_id", "image_profile", "is_simulated"):
        if key not in data or data[key] is None:
            raise ValueError("Missing binding: " + key)
    if type(data['is_simulated']) is not bool or type(data['camera_id']) is not int or data['camera_id'] not in (0, 1, 2):
        raise ValueError("Invalid source binding")
    profile=data['image_profile']
    if not isinstance(profile,dict) or any(k not in profile for k in ('width','height','crop','hmirror','vflip','rotation','pose_id')):
        raise ValueError('Complete image profile required')
    if profile['pose_id'] != data['pose_id'] or not data['pose_id']:
        raise ValueError('Known consistent docking pose required')
    if any(type(profile[k]) is not int or profile[k]<=0 for k in ('width','height')):
        raise ValueError('Positive frame dimensions required')
    if any(type(profile[k]) is not bool for k in ('crop','hmirror','vflip')) or profile['rotation'] not in (0,90,180,270):
        raise ValueError('Invalid image geometry')
    training, validation = data['training'], data['validation']
    if len(validation) < 2:
        raise ValueError("At least two independent validation points are required")
    train_px = [x['px'] for x in training]
    test_px = [x['px'] for x in validation]
    if any(p in train_px for p in test_px) or len({tuple(p) for p in test_px}) != len(test_px):
        raise ValueError("Validation points must be independent and distinct")
    if any(len(p) != 2 or not all(math.isfinite(float(v)) for v in p) for p in test_px):
        raise ValueError("Invalid validation pixels")
    h = fit_homography(train_px, [x['mm'] for x in training])
    predictions = project(h, test_px)
    errors = []
    for point, sample in zip(predictions, validation):
        mm = sample['mm']
        if len(mm) != 2 or not all(math.isfinite(float(v)) for v in mm):
            raise ValueError("Invalid validation millimeters")
        errors.append(math.hypot(point[0]-mm[0], point[1]-mm[1]))
    if max(errors) > max_error_mm:
        raise ValueError("Independent validation exceeds tolerance: %.3fmm" % max(errors))
    record = {k: data[k] for k in ("calibration_id", "camera_id", "plane", "station", "pose_id", "image_profile", "is_simulated")}
    record.update(id=data['calibration_id'], H=h, validated=True, target_pose_valid=False,
                  training_count=len(training), validation_count=len(validation),
                  validation_errors_mm=errors, max_error_mm=max(errors),
                  acceptance_tolerance_mm=max_error_mm,
                  note="Numerical validation only; motion authorization and measured hardware binding remain separate")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--max-error-mm', required=True, type=float)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = calibrate(json.loads(raw.decode('utf-8-sig')), args.max_error_mm)
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'max_error_mm': result['max_error_mm'],
                      'is_simulated': result['is_simulated']}))


if __name__ == '__main__':
    main()
