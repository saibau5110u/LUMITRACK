"""
Feature Extraction Module for LumiTrack (Variant 10.03: Museum Exposure Limiter)

Extracts exactly 13 physically meaningful features from 7-point angular light scans
(15°, 40°, 65°, 90°, 115°, 140°, 165°).

Every machine learning model receives these exact 13 features in this fixed order.
"""

import numpy as np

# Approved canonical servo angles for LumiTrack STM32 NUCLEO-F446RE
APPROVED_ANGLES_DEG = [15, 40, 65, 90, 115, 140, 165]

# Canonical 13 feature names in fixed order
FEATURE_NAMES = [
    "peak_intensity",      # 1. Maximum light intensity across the scan (Lux)
    "peak_angle_deg",      # 2. Angle at which peak intensity was detected (Degrees)
    "min_intensity",       # 3. Minimum light intensity recorded across the scan (Lux)
    "mean_intensity",      # 4. Average light intensity across all 7 scan points (Lux)
    "std_intensity",       # 5. Standard deviation of intensity (spatial spread) (Lux)
    "curve_asymmetry",     # 6. Asymmetry between right wing (115-165°) and left wing (15-65°) [-1, 1]
    "gradient_mean",       # 7. Mean spatial gradient between adjacent angle checkpoints (Lux/deg)
    "gradient_max",        # 8. Maximum spatial gradient between adjacent angle checkpoints (Lux/deg)
    "contrast_ratio",      # 9. Dynamic contrast ratio: (peak - min) / (peak + min) [0, 1]
    "exposure_integral",   # 10. Numerical trapezoidal integral of light over the angle span (Lux·deg)
    "ratio_to_target",     # 11. Ratio of peak intensity to operator threshold (target_lux)
    "ratio_to_ambient",    # 12. Ratio of peak intensity to ambient dark baseline (ambient_lux)
    "partial_slope"        # 13. Early angular slope between POS_0 (15°) and POS_3 (90°) (Lux/deg)
]


def extract_features_from_scan(lux_array, target_lux=500.0, ambient_lux=40.0):
    """
    Extracts exactly 13 physical lighting features from a 7-point angular scan.

    Args:
        lux_array (list or np.ndarray): 7 calibrated lux values at approved angles
                                       [15°, 40°, 65°, 90°, 115°, 140°, 165°].
        target_lux (float): Operator target exposure limit from potentiometer (Lux).
        ambient_lux (float): Ambient/dark reference level (Lux).

    Returns:
        dict: Dictionary containing exactly 13 extracted physical features.
    """
    lux = np.array(lux_array, dtype=float)
    angles = np.array(APPROVED_ANGLES_DEG, dtype=float)

    # 1 & 2: Peak intensity and angle
    peak_idx = int(np.argmax(lux))
    peak_intensity = float(lux[peak_idx])
    peak_angle_deg = float(angles[peak_idx])

    # 3, 4, 5: Summary statistics
    min_intensity = float(np.min(lux))
    mean_intensity = float(np.mean(lux))
    std_intensity = float(np.std(lux))

    # 6: Curve asymmetry around safe neutral plane (POS_3, 90°)
    # Left wing: POS_0, POS_1, POS_2 (15°, 40°, 65°)
    # Right wing: POS_4, POS_5, POS_6 (115°, 140°, 165°)
    left_wing = float(np.sum(lux[:3]))
    right_wing = float(np.sum(lux[4:]))
    total_wing = left_wing + right_wing + 1e-5
    curve_asymmetry = float((right_wing - left_wing) / total_wing)

    # 7 & 8: Spatial gradients between adjacent angle steps (|Δlux| / Δdeg)
    diffs = np.abs(np.diff(lux))
    angle_steps = np.diff(angles)
    gradients = diffs / angle_steps
    gradient_mean = float(np.mean(gradients))
    gradient_max = float(np.max(gradients))

    # 9: Contrast ratio (measures localized glint/spotlight vs background)
    contrast_ratio = float((peak_intensity - min_intensity) / (peak_intensity + min_intensity + 1e-5))

    # 10: Angular exposure integral using trapezoidal rule (total cumulative flux)
    if hasattr(np, 'trapezoid'):
        trapz_sum = np.trapezoid(lux, angles)
    else:
        trapz_sum = np.trapz(lux, angles)
    exposure_integral = float(trapz_sum)

    # 11 & 12: Context ratios to operator threshold and ambient reference
    ratio_to_target = float(peak_intensity / (target_lux + 1e-5))
    ratio_to_ambient = float(peak_intensity / (ambient_lux + 1e-5))

    # 13: Early slope across the first half of the scan from 15° to 90°
    partial_slope = float((lux[3] - lux[0]) / (angles[3] - angles[0]))

    return {
        "peak_intensity": peak_intensity,
        "peak_angle_deg": peak_angle_deg,
        "min_intensity": min_intensity,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "curve_asymmetry": curve_asymmetry,
        "gradient_mean": gradient_mean,
        "gradient_max": gradient_max,
        "contrast_ratio": contrast_ratio,
        "exposure_integral": exposure_integral,
        "ratio_to_target": ratio_to_target,
        "ratio_to_ambient": ratio_to_ambient,
        "partial_slope": partial_slope
    }


def extract_feature_vector(lux_array, target_lux=500.0, ambient_lux=40.0):
    """
    Extracts the 13 features as a 1D numpy array in canonical order.
    """
    feats = extract_features_from_scan(lux_array, target_lux, ambient_lux)
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=float)


def extract_partial_features(partial_lux_array, target_lux=500.0, ambient_lux=40.0):
    """
    Extracts features from an early 4-point scan (15°, 40°, 65°, 90°)
    retained for early monitoring compatibility.
    """
    lux = np.array(partial_lux_array[:4], dtype=float)
    angles = np.array(APPROVED_ANGLES_DEG[:4], dtype=float)

    peak_intensity = float(np.max(lux))
    mean_intensity = float(np.mean(lux))
    diffs = np.abs(np.diff(lux))
    gradient_max = float(np.max(diffs / np.diff(angles)))
    ratio_to_target = float(peak_intensity / (target_lux + 1e-5))
    partial_slope = float((lux[-1] - lux[0]) / (angles[-1] - angles[0]))

    return {
        "partial_peak": peak_intensity,
        "partial_mean": mean_intensity,
        "partial_gradient_max": gradient_max,
        "partial_ratio_to_target": ratio_to_target,
        "partial_slope": partial_slope
    }
