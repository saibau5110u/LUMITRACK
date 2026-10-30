"""
Model Training and Comparison Pipeline for LumiTrack (Variant 10.03)

Trains and compares the four required algorithms:
1. Ridge Logistic Regression (L2 regularization, StandardScaler, class probabilities)
2. Random Forest Classifier (100 decision trees, majority voting)
3. Gradient Boosting Classifier (100 sequential boosting stages)
4. Rule-Based Baseline (Heuristic threshold on peak light intensity; binary baseline)

Model Selection:
Models are evaluated using 5-Fold Stratified Cross-Validation on the training set.
The model with the highest cross-validation Macro-F1 score is selected as the best model.
All models are then evaluated on the independent held-out test set for final verification.
"""

import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from feature_extraction import extract_features_from_scan, FEATURE_NAMES, APPROVED_ANGLES_DEG

# Regime class mappings
CLASS_NAMES = ["Normal Diffuse", "Overexposure Risk", "Specular Confuser"]


def load_dataset(csv_path):
    """
    Loads dataset from CSV and extracts the 13 canonical features for each scan.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    print(f"[Dataset] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    lux_columns = [f"lux_pos{i}_{deg}deg" for i, deg in enumerate(APPROVED_ANGLES_DEG)]

    feature_rows = []
    for _, row in df.iterrows():
        lux_values = [row[c] for c in lux_columns]
        target_lux = row["operator_target_lux"]
        ambient_lux = row["ambient_ref_lux"]

        feats = extract_features_from_scan(lux_values, target_lux, ambient_lux)
        # Ensure exact 13 features in fixed order
        feature_rows.append([feats[name] for name in FEATURE_NAMES])

    X = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    y = df["regime_class"].values
    sessions = df["session_id"].values

    print(f"[Dataset] Successfully loaded {len(X)} samples with {len(FEATURE_NAMES)} features.")
    return X, y, sessions


def evaluate_predictions(y_true, y_pred, model_name, is_binary=False):
    """
    Computes Accuracy, Precision, Recall, Macro-F1, and Confusion Matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Overexposure (Class 1) specific recall and false alarm rate
    rec_class1 = float(cm[1, 1] / (np.sum(cm[1, :]) + 1e-5))
    fp_class1 = float((cm[0, 1] + cm[2, 1]) / (np.sum(cm[0, :]) + np.sum(cm[2, :]) + 1e-5))

    return {
        "model_name": model_name,
        "is_binary_baseline": is_binary,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "macro_f1": round(f1, 4),
        "overexposure_recall": round(rec_class1, 4),
        "overexposure_false_alarm": round(fp_class1, 4),
        "confusion_matrix": cm.tolist()
    }


def rule_based_predict(X_subset):
    """
    Algorithm 4: Rule-Based Baseline
    Rule: IF peak_light_intensity > 400 Lux THEN Overexposure (1) ELSE Normal Lighting (0).
    Note: Binary classification baseline; does not predict class 2 (Specular Confuser).
    """
    y_pred = []
    for _, row in X_subset.iterrows():
        if row["peak_intensity"] > 400.0:
            y_pred.append(1)  # Overexposure
        else:
            y_pred.append(0)  # Normal Lighting
    return np.array(y_pred)


def train_and_evaluate_all():
    """
    Main training and evaluation pipeline.
    """
    # Relative path resolution
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "seed_scan_dataset.csv")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(os.path.dirname(base_dir), "04_Verification_and_Tests", "test_reports")

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Load dataset & extract 13 features
    X, y, sessions = load_dataset(data_path)

    # 2. Stratified train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"[Split] Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 3. Fit StandardScaler on Training Set ONLY (prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Instantiate the 3 ML Models
    # Model 1: Ridge Logistic Regression (L2 regularization is default in scikit-learn LogisticRegression)
    model_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    # Model 2: Random Forest Classifier (100 decision trees)
    model_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

    # Model 3: Gradient Boosting Classifier (100 boosting stages)
    model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)

    # 5. Model Selection via 5-Fold Stratified Cross-Validation on the Training Set
    # (Avoid selecting model using test-set results and then reporting same test results)
    print("\n--- Running 5-Fold Stratified Cross-Validation on Training Set ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores_lr = cross_val_score(model_lr, X_train_scaled, y_train, cv=cv, scoring="f1_macro")
    cv_scores_rf = cross_val_score(model_rf, X_train, y_train, cv=cv, scoring="f1_macro")
    cv_scores_gb = cross_val_score(model_gb, X_train, y_train, cv=cv, scoring="f1_macro")

    cv_summary = {
        "Ridge Logistic Regression": float(np.mean(cv_scores_lr)),
        "Random Forest Classifier": float(np.mean(cv_scores_rf)),
        "Gradient Boosting Classifier": float(np.mean(cv_scores_gb))
    }

    print(f"  Ridge Logistic Regression CV Macro-F1: {cv_summary['Ridge Logistic Regression']:.4f}")
    print(f"  Random Forest Classifier  CV Macro-F1: {cv_summary['Random Forest Classifier']:.4f}")
    print(f"  Gradient Boosting Clf     CV Macro-F1: {cv_summary['Gradient Boosting Classifier']:.4f}")

    # Determine best model from training validation results
    best_model_name = max(cv_summary, key=cv_summary.get)
    print(f"\n[Model Selection] Best performing model based on CV validation: {best_model_name}")

    # 6. Fit models on the full Training Set
    model_lr.fit(X_train_scaled, y_train)
    model_rf.fit(X_train, y_train)
    model_gb.fit(X_train, y_train)

    # Dictionary of trained ML models
    trained_models = {
        "Ridge Logistic Regression": model_lr,
        "Random Forest Classifier": model_rf,
        "Gradient Boosting Classifier": model_gb
    }

    # 7. Independent Final Evaluation on Held-Out Test Set
    print("\n--- Independent Final Evaluation on Held-Out Test Set ---")
    y_pred_lr = model_lr.predict(X_test_scaled)
    y_pred_rf = model_rf.predict(X_test)
    y_pred_gb = model_gb.predict(X_test)
    y_pred_base = rule_based_predict(X_test)

    metrics_lr = evaluate_predictions(y_test, y_pred_lr, "Ridge Logistic Regression")
    metrics_rf = evaluate_predictions(y_test, y_pred_rf, "Random Forest Classifier")
    metrics_gb = evaluate_predictions(y_test, y_pred_gb, "Gradient Boosting Classifier")
    metrics_base = evaluate_predictions(y_test, y_pred_base, "Rule-Based Baseline", is_binary=True)

    comparison_table = [metrics_lr, metrics_rf, metrics_gb, metrics_base]

    # Print summary table
    print(f"{'Model Name':<30} | {'Acc':<7} | {'Prec':<7} | {'Recall':<7} | {'Macro-F1':<8} | {'Overexp Rec':<11}")
    print("-" * 85)
    for m in comparison_table:
        print(f"{m['model_name']:<30} | {m['accuracy']:<7.4f} | {m['precision']:<7.4f} | {m['recall']:<7.4f} | {m['macro_f1']:<8.4f} | {m['overexposure_recall']:<11.4f}")

    # 8. Save Models and Preprocessing Objects (both .pkl and .joblib for compatibility)
    print("\n[Serialization] Saving models and preprocessing objects...")
    # Scaler
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    # Logistic Regression
    with open(os.path.join(models_dir, "logistic_regression.pkl"), "wb") as f:
        pickle.dump(model_lr, f)
    joblib.dump(model_lr, os.path.join(models_dir, "logistic_regression.joblib"))
    joblib.dump(model_lr, os.path.join(models_dir, "ridge_model.joblib"))

    # Random Forest
    with open(os.path.join(models_dir, "random_forest.pkl"), "wb") as f:
        pickle.dump(model_rf, f)
    joblib.dump(model_rf, os.path.join(models_dir, "random_forest.joblib"))

    # Gradient Boosting
    with open(os.path.join(models_dir, "gradient_boosting.pkl"), "wb") as f:
        pickle.dump(model_gb, f)
    joblib.dump(model_gb, os.path.join(models_dir, "gradient_boosting.joblib"))

    # Best Model
    best_model_obj = trained_models[best_model_name]
    with open(os.path.join(models_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model_obj, f)
    joblib.dump(best_model_obj, os.path.join(models_dir, "best_model.joblib"))

    # 9. Save Metrics Summary JSON
    summary_data = {
        "dataset_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "cross_validation_scores_macro_f1": cv_summary,
        "selected_best_model": best_model_name,
        "comparison_table": comparison_table
    }

    metrics_json_path = os.path.join(models_dir, "metrics_summary.json")
    with open(metrics_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[Serialization] Metrics summary saved to: {metrics_json_path}")

    # 10. Generate and Save Visualizations
    _generate_plots(reports_dir, comparison_table, y_test, y_pred_rf, y_pred_base, X_test)

    print("\n=== Training & Evaluation Pipeline Completed Successfully ===")
    return summary_data


def _generate_plots(reports_dir, comparison_table, y_test, y_pred_best, y_pred_base, X_test):
    """
    Generates confusion matrix, accuracy comparison, and decision timeline plots.
    """
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Confusion Matrix Plot (Baseline vs Best Model)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Baseline CM
    cm_base = np.array(comparison_table[3]["confusion_matrix"])
    axes[0].imshow(cm_base, interpolation="nearest", cmap="Blues")
    axes[0].set_title(f"Baseline Heuristic\nMacro-F1: {comparison_table[3]['macro_f1']:.3f} | Acc: {comparison_table[3]['accuracy']*100:.1f}%")
    axes[0].set_xticks(range(3))
    axes[0].set_yticks(range(3))
    axes[0].set_xticklabels(CLASS_NAMES, rotation=15)
    axes[0].set_yticklabels(CLASS_NAMES)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Ground Truth")
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, str(cm_base[i, j]), ha="center", va="center",
                         color="red" if i != j and cm_base[i, j] > 0 else "black")

    # Best ML Model CM
    best_name = comparison_table[1]["model_name"]  # Random Forest
    cm_best = np.array(comparison_table[1]["confusion_matrix"])
    axes[1].imshow(cm_best, interpolation="nearest", cmap="Greens")
    axes[1].set_title(f"{best_name}\nMacro-F1: {comparison_table[1]['macro_f1']:.3f} | Acc: {comparison_table[1]['accuracy']*100:.1f}%")
    axes[1].set_xticks(range(3))
    axes[1].set_yticks(range(3))
    axes[1].set_xticklabels(CLASS_NAMES, rotation=15)
    axes[1].set_yticklabels(CLASS_NAMES)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Ground Truth")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, str(cm_best[i, j]), ha="center", va="center",
                         color="red" if i != j and cm_best[i, j] > 0 else "black")

    plt.tight_layout()
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # 2. Model Accuracy & F1 Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = [m["model_name"] for m in comparison_table]
    accs = [m["accuracy"] * 100 for m in comparison_table]
    f1s = [m["macro_f1"] * 100 for m in comparison_table]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="#1f77b4")
    ax.bar(x + width/2, f1s, width, label="Macro-F1 (%)", color="#2ca02c")

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Algorithm Performance Comparison (Held-Out Test Set)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    for i in range(len(names)):
        ax.text(x[i] - width/2, accs[i] + 1.5, f"{accs[i]:.1f}%", ha="center", fontsize=9)
        ax.text(x[i] + width/2, f1s[i] + 1.5, f"{f1s[i]:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    comp_path = os.path.join(reports_dir, "model_comparison.png")
    plt.savefig(comp_path, dpi=150)
    plt.close()

    # 3. Synchronized Physical-Signal Decision Timeline
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    sample_runs = 25
    x_axis = np.arange(1, sample_runs + 1)

    sample_peak_lux = X_test["peak_intensity"].iloc[:sample_runs].values
    sample_actual = y_test[:sample_runs]
    sample_pred = y_pred_best[:sample_runs]

    # Top: Light intensity vs Potentiometer Threshold
    ax_top.plot(x_axis, sample_peak_lux, "b-o", label="Peak Measured Lux")
    ax_top.axhline(400.0, color="r", linestyle="--", label="Conservation Threshold (400 Lux)")
    ax_top.set_ylabel("Intensity (Lux)")
    ax_top.set_title("LumiTrack Synchronized Physical-Signal Decision Timeline")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper right")

    # Mid: Predicted Regime Class
    ax_mid.step(x_axis, sample_actual, where="mid", label="Ground Truth Regime", color="black", alpha=0.6)
    ax_mid.plot(x_axis, sample_pred, "g^", label="Model Classification", markersize=8)
    ax_mid.set_yticks([0, 1, 2])
    ax_mid.set_yticklabels(["Normal", "Overexposure", "Confuser"])
    ax_mid.set_ylabel("Class")
    ax_mid.grid(True, alpha=0.3)
    ax_mid.legend(loc="upper right")

    # Bottom: MCU Commanded Angle Outcome
    sample_angles = []
    for p in sample_pred:
        if p == 1:
            sample_angles.append(15)  # Clamped away / deflected
        else:
            sample_angles.append(90)  # Safe neutral
    ax_bot.step(x_axis, sample_angles, where="mid", color="purple", label="MCU Commanded Safe Angle (°)")
    ax_bot.set_yticks([15, 40, 65, 90, 115, 140, 165])
    ax_bot.set_ylabel("Servo Angle (°)")
    ax_bot.set_xlabel("Trial Run Number")
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(loc="upper right")

    plt.tight_layout()
    timeline_path = os.path.join(reports_dir, "decision_timeline.png")
    plt.savefig(timeline_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    train_and_evaluate_all()
