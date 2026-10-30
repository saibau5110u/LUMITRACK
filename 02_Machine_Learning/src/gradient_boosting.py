"""
Algorithm 3: Gradient Boosting Classifier
LumiTrack (Variant 10.03: Museum Exposure Limiter)

Characteristics:
- Uses GradientBoostingClassifier with 100 boosting estimators.
- Builds decision trees sequentially.
- Identifies errors/residuals from previous stages.
- Trains subsequent trees to correct those residual errors.
- Combines predictions through stage-wise additive modeling.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
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


def build_and_train_gradient_boosting():
    print("=" * 70)
    print("  Algorithm 3: Gradient Boosting Classifier (100 Boosting Stages)")
    print("=" * 70)

    # 1. Load Data
    X, y = load_dataset()
    print(f"[Dataset] Total samples: {len(X)} with {len(FEATURE_NAMES)} features.")

    # 2. Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 3. Instantiate Gradient Boosting Classifier
    # 100 boosting stages, learning_rate=0.1, max_depth=4 shallow trees
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42
    )

    # 4. 5-Fold Stratified Cross-Validation on Training Partition
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro")
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

    print(f"\n[Validation] 5-Fold CV Accuracy: {np.mean(cv_acc)*100:.2f}% (+/- {np.std(cv_acc)*100:.2f}%)")
    print(f"[Validation] 5-Fold CV Macro-F1: {np.mean(cv_f1):.4f} (+/- {np.std(cv_f1):.4f})")

    # 5. Fit Model on Full Training Set
    model.fit(X_train, y_train)

    # 6. Evaluate on Independent Held-Out Test Set (120 samples)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    print("\n[Held-Out Test Results]")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  Macro-F1:  {f1*100:.2f}%")

    print("\n[Confusion Matrix]")
    print(f"{'Actual \\ Predicted':<25} | {'Normal (0)':<12} | {'Overexp (1)':<12} | {'Specular (2)':<12}")
    print("-" * 68)
    for i, row in enumerate(cm):
        print(f"{CLASS_NAMES[i]:<25} | {row[0]:<12} | {row[1]:<12} | {row[2]:<12}")

    # 7. Save Serialized Model
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, "gradient_boosting.pkl"), "wb") as f:
        pickle.dump(model, f)

    print(f"\n[Saved] Model saved to: {os.path.join(models_dir, 'gradient_boosting.pkl')}")
    return model


if __name__ == "__main__":
    build_and_train_gradient_boosting()
