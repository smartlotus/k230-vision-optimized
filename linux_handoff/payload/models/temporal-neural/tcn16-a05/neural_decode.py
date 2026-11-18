"""Camera XY candidate decoding shared by the host replay and CanMV trial."""
import math


def decode_logits(values, centers, temperature, threshold=None):
    if (len(values) != len(centers) or len(values) != 2304
            or not all(math.isfinite(value) for value in values) or not math.isfinite(temperature)
            or temperature <= 0 or threshold is not None and not 0 <= threshold <= 1):
        raise ValueError('Invalid neural XY output or calibration')
    high = max(values)
    top = max(range(len(values)), key=lambda index: values[index])
    posterior_max = 1. / sum(math.exp((value - high) / temperature) for value in values)
    chosen = []
    for index in sorted(range(len(values)), key=lambda i: (-values[i], i)):
        if all(sum((a - b)**2 for a, b in zip(centers[index], site)) >= 100**2 for site in chosen):
            chosen.append(centers[index])
            if len(chosen) == 5:
                break
    accepted = threshold is not None and posterior_max >= threshold
    return dict(status='localized' if accepted else 'unlocalized',
                camera_x_mm=centers[top][0] if accepted else None,
                camera_y_mm=centers[top][1] if accepted else None, confidence=None,
                visual_peak_mass=posterior_max, confidence_qualification='VALIDATION_THRESHOLD_ONLY',
                top_k_camera_xy=[dict(camera_x_mm=point[0], camera_y_mm=point[1]) for point in chosen])
