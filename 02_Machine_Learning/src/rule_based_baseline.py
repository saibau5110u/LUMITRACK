"""
Algorithm 4: Rule-Based Baseline (Heuristic)
LumiTrack (Variant 10.03: Museum Exposure Limiter)

Characteristics:
- Uses a simple deterministic Python function without Machine Learning training.
- Rule:
    IF peak_light_intensity > 400.0 Lux:
        Prediction = Overexposure (1)
    ELSE:
        Prediction = Normal Lighting (0)
- Serves as an experimental control baseline.
- IMPORTANT: This is a binary classification method. It cannot predict class 2 (Specular Confuser)
  and triggers false alarms when harmless glints exceed 400 Lux.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from feature_extraction import extract_features_from_scan, FEATURE_NAMES, APPROVED_ANGLES_DEG

CLASS_NAMES = ["Normal Diffuse", "Overexposure Risk", "Specular Confuser"]


def load_dataset():
    """Loads dataset and extracts the 13 canonical features."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "seed_scan_dataset.csv")

    df = pd.read_csv(data_path)
    lux_cols = [f"lux_pos{i}_{deg}deg" for i, deg in enumerate(APPROVED_ANGLES_DEG)]

    rows = []
    for _, r in df.iterrows():
        lux_vals = [r[c] for c in lux_cols]
        feats = extract_features_from_scan(lux_vals, r["operator_target_lux"], r["ambient_ref_lux"])
        rows.append([feats[name] for name in FEATURE_NAMES])

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = df["regime_class"].values
    return X, y


def rule_based_predict_single(peak_light_intensity):
    """
    Core Rule-Based Function:
    IF peak_light_intensity > 400 Lux:
        Prediction = Overexposure (1)
    ELSE:
        Prediction = Normal Lighting (0)
    """
    if peak_light_intensity > 400.0:
        return 1, "OVEREXPOSURE_RISK"
    else:
        return 0, "NORMAL_DIFFUSE"


def rule_based_predict_batch(X_df):
    """Evaluates rule across multiple samples."""
    return np.where(X_df["peak_intensity"] > 400.0, 1, 0)


def evaluate_rule_based_baseline():
    print("=" * 70)
    print("  Algorithm 4: Rule-Based Baseline (Deterministic Threshold)")
    print("=" * 70)

    # 1. Load Data
    X, y = load_dataset()
    print(f"[Dataset] Total samples: {len(X)} with {len(FEATURE_NAMES)} features.")

    # 2. Stratified Train-Test Split (80% train, 20% test)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 3. Apply Heuristic Rule (NO ML TRAINING)
    y_pred = rule_based_predict_batch(X_test)

    # 4. Compute Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print("\n[Held-Out Test Results (Binary Baseline)]")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  Macro-F1:  {f1*100:.2f}%")

    print("\n[Confusion Matrix]")
    print(f"{'Actual \\ Predicted':<25} | {'Normal (0)':<12} | {'Overexp (1)':<12} | {'Specular (2)':<12}")
    print("-" * 68)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:<25} | {row[0]:<12} | {row[1]:<12} | {row[2]:<12}")

    # 5. Explain Failure Mode
    specular_total = cm[2].sum()
    specular_false_alarms = cm[2, 1]
    print("\n[Critical Baseline Limitation Analysis]")
    print(f"  * Total Specular Confuser (Glint) Samples in Test Set: {specular_total}")
    print(f"  * Specular Glints Falsely Flagged as Overexposure:     {specular_false_alarms} ({specular_false_alarms/specular_total*100:.1f}%)")
    print("  * Reason: The baseline relies solely on a single scalar (peak > 400 Lux).")
    print("            It cannot analyze spatial gradients or cumulative exposure flux,")
    print("            causing unnecessary blind closures in museums.")


if __name__ == "__main__":
    evaluate_rule_based_baseline()
