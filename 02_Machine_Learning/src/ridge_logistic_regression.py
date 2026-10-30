"""
Algorithm 1: Ridge Logistic Regression
LumiTrack (Variant 10.03: Museum Exposure Limiter)

Characteristics:
- Uses sklearn LogisticRegression with L2 (Ridge) regularization.
- Standardizes the 13 physical features using StandardScaler.
- Computes class probabilities using the Softmax function.
- Predicts the lighting category with the highest probability.
- Prevents overfitting through L2 shrinkage on weight coefficients.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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


def build_and_train_ridge_logistic_regression():
    print("=" * 70)
    print("  Algorithm 1: Ridge Logistic Regression (L2 Regularization)")
    print("=" * 70)

    # 1. Load Data
    X, y = load_dataset()
    print(f"[Dataset] Total samples: {len(X)} with {len(FEATURE_NAMES)} features.")

    # 2. Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 3. Feature Scaling (StandardScaler fitted strictly on training data)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Instantiate Ridge Logistic Regression
    # C=1.0 is inverse regularization strength; L2 penalty is default in scikit-learn
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    # 5. 5-Fold Stratified Cross-Validation on Training Partition
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1_macro")
    cv_acc = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="accuracy")

    print(f"\n[Validation] 5-Fold CV Accuracy: {np.mean(cv_acc)*100:.2f}% (+/- {np.std(cv_acc)*100:.2f}%)")
    print(f"[Validation] 5-Fold CV Macro-F1: {np.mean(cv_f1):.4f} (+/- {np.std(cv_f1):.4f})")

    # 6. Fit Model on Full Training Set
    model.fit(X_train_scaled, y_train)

    # 7. Evaluate on Independent Held-Out Test Set (120 samples)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

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

    # 8. Save Serialized Artifacts
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, "logistic_regression.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print(f"\n[Saved] Model saved to: {os.path.join(models_dir, 'logistic_regression.pkl')}")
    return model, scaler


if __name__ == "__main__":
    build_and_train_ridge_logistic_regression()
