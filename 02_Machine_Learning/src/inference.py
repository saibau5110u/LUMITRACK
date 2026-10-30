"""
Real-Time Inference Engine for LumiTrack (Variant 10.03: Museum Exposure Limiter)

Loads serialized models and provides lighting classification and decision-making
for the IoT host controller and web dashboard.

Supported Algorithms:
1. Best Model (Selected during cross-validation, default Random Forest)
2. Ridge Logistic Regression (with StandardScaler)
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. Rule-Based Baseline (Heuristic comparison)
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from feature_extraction import (
    extract_features_from_scan,
    extract_partial_features,
    FEATURE_NAMES,
    APPROVED_ANGLES_DEG
)

# Canonical regime naming
REGIME_NAMES = {
    0: "NORMAL_DIFFUSE",
    1: "OVEREXPOSURE_RISK",
    2: "CONFUSER_SPECULAR"
}


def load_artifact(file_path):
    """Loads a serialized model or scaler using pickle (with joblib fallback)."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(file_path)


class LumiTrackPredictor:
    """
    Inference manager for LumiTrack.
    Loads trained models and provides real-time classification for incoming sensor scans.
    """

    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

        self.models_dir = models_dir

        # Load metrics summary to identify the best model
        self.metrics_summary = {}
        summary_path = os.path.join(models_dir, "metrics_summary.json")
        if os.path.exists(summary_path):
            import json
            with open(summary_path, "r") as f:
                self.metrics_summary = json.load(f)

        self.best_model_name = self.metrics_summary.get("selected_best_model", "Random Forest Classifier")

        # Load Preprocessing Object (StandardScaler for Logistic Regression)
        self.scaler = load_artifact(os.path.join(models_dir, "scaler.pkl"))
        if self.scaler is None:
            self.scaler = load_artifact(os.path.join(models_dir, "scaler.joblib"))

        # Load Best Model
        self.best_model = load_artifact(os.path.join(models_dir, "best_model.pkl"))
        if self.best_model is None:
            self.best_model = load_artifact(os.path.join(models_dir, "best_model.joblib"))

        # Load Individual ML Models
        self.lr_model = load_artifact(os.path.join(models_dir, "logistic_regression.pkl"))
        if self.lr_model is None:
            self.lr_model = load_artifact(os.path.join(models_dir, "ridge_model.joblib"))

        self.rf_model = load_artifact(os.path.join(models_dir, "random_forest.pkl"))
        self.gb_model = load_artifact(os.path.join(models_dir, "gradient_boosting.pkl"))

        # Fallback to best_model if individual file is missing
        if self.rf_model is None and self.best_model is not None:
            self.rf_model = self.best_model

        if self.best_model is None:
            raise FileNotFoundError(
                f"No trained models found in {models_dir}. Please run train_models.py first."
            )

    def rule_based_predict(self, peak_intensity):
        """
        Algorithm 4: Rule-Based Baseline
        IF peak_light_intensity > 400 Lux -> Overexposure (1) ELSE Normal (0).
        Binary classification baseline.
        """
        if peak_intensity > 400.0:
            return 1, "OVEREXPOSURE_RISK"
        return 0, "NORMAL_DIFFUSE"

    def evaluate_full_scan(self, lux_array, target_lux=500.0, ambient_lux=40.0, model_choice="Best Model"):
        """
        Evaluates a complete 7-point scan (13 features) and returns classification and safe action.

        Args:
            lux_array (list): 7 light readings in Lux.
            target_lux (float): Operator target threshold from potentiometer.
            ambient_lux (float): Reference ambient light level.
            model_choice (str): One of 'Best Model', 'Ridge Logistic Regression',
                                'Random Forest Classifier', 'Gradient Boosting Classifier',
                                or 'Rule-Based Baseline'.

        Returns:
            dict: Detailed prediction result including predicted regime, confidence,
                  feature values, and commanded safe action.
        """
        # 1. Extract the canonical 13 features
        feats = extract_features_from_scan(lux_array, target_lux, ambient_lux)
        X = pd.DataFrame([[feats[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
        peak_intensity = feats["peak_intensity"]

        # 2. Execute selected algorithm
        if model_choice == "Rule-Based Baseline":
            pred_class, regime_name = self.rule_based_predict(peak_intensity)
            confidence = 1.0  # Deterministic rule
            model_label = "Rule-Based Baseline (Heuristic)"
        elif model_choice == "Ridge Logistic Regression" and self.lr_model is not None:
            X_scaled = self.scaler.transform(X) if self.scaler is not None else X
            pred_class = int(self.lr_model.predict(X_scaled)[0])
            probas = self.lr_model.predict_proba(X_scaled)[0]
            confidence = float(probas[pred_class])
            regime_name = REGIME_NAMES.get(pred_class, "UNKNOWN")
            model_label = "Ridge Logistic Regression"
        elif model_choice == "Gradient Boosting Classifier" and self.gb_model is not None:
            pred_class = int(self.gb_model.predict(X)[0])
            probas = self.gb_model.predict_proba(X)[0]
            confidence = float(probas[pred_class])
            regime_name = REGIME_NAMES.get(pred_class, "UNKNOWN")
            model_label = "Gradient Boosting Classifier"
        elif model_choice == "Random Forest Classifier" and self.rf_model is not None:
            pred_class = int(self.rf_model.predict(X)[0])
            probas = self.rf_model.predict_proba(X)[0]
            confidence = float(probas[pred_class])
            regime_name = REGIME_NAMES.get(pred_class, "UNKNOWN")
            model_label = "Random Forest Classifier"
        else:
            # Default: Best performing model
            target_model = self.best_model
            # If best model happens to be Logistic Regression, scale features
            if isinstance(target_model, LogisticRegression) and self.scaler is not None:
                X_in = self.scaler.transform(X)
            else:
                X_in = X
            pred_class = int(target_model.predict(X_in)[0])
            probas = target_model.predict_proba(X_in)[0]
            confidence = float(probas[pred_class])
            regime_name = REGIME_NAMES.get(pred_class, "UNKNOWN")
            model_label = f"Best Model ({self.best_model_name})"

        # 3. Canonical Safe Angle Decision Policy:
        # Class 0 (NORMAL): Keep artifact viewable at safe neutral (POS_3, 90 deg)
        # Class 1 (OVEREXPOSURE): Direct hazard! Clamp shade / deflect away from peak angle
        # Class 2 (CONFUSER): Glint false alarm! Hold safe neutral without closing
        if pred_class == 1:
            peak_idx = int(np.argmax(lux_array))
            if peak_idx <= 2:
                rec_angle_id = 6  # Deflect toward 165°
            else:
                rec_angle_id = 0  # Deflect toward 15°
            cmd = "ANGLE_ID"
            action_desc = f"DEFLECT: Steer away from direct light (Deflected to {APPROVED_ANGLES_DEG[rec_angle_id]}°)"
        elif pred_class == 2:
            rec_angle_id = 3  # Neutral POS_3
            cmd = "HOLD"
            action_desc = "HOLD: Specular glint detected; maintain stable exhibit position (90°)"
        else:
            rec_angle_id = 3  # Neutral POS_3
            cmd = "ANGLE_ID"
            action_desc = "NORMAL: Ambient lighting within safe conservation tolerance (90°)"

        return {
            "regime_class": pred_class,
            "regime_name": regime_name,
            "confidence": round(confidence, 4),
            "command": cmd,
            "recommended_angle_id": rec_angle_id,
            "recommended_angle_deg": APPROVED_ANGLES_DEG[rec_angle_id],
            "action_description": action_desc,
            "peak_intensity": peak_intensity,
            "is_overexposed": (pred_class == 1 or peak_intensity > 400.0),
            "model_used": model_label,
            "features": feats
        }

    def compare_all_models(self, lux_array, target_lux=500.0, ambient_lux=40.0):
        """
        Runs all four algorithms on the same 13 features to allow side-by-side comparison.
        """
        feats = extract_features_from_scan(lux_array, target_lux, ambient_lux)
        X = pd.DataFrame([[feats[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
        peak_intensity = feats["peak_intensity"]

        results = {}

        # 1. Ridge Logistic Regression
        if self.lr_model is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            pred_lr = int(self.lr_model.predict(X_scaled)[0])
            conf_lr = float(self.lr_model.predict_proba(X_scaled)[0][pred_lr])
            results["Ridge Logistic Regression"] = {
                "class": pred_lr,
                "regime": REGIME_NAMES.get(pred_lr, "UNKNOWN"),
                "confidence": round(conf_lr, 4)
            }

        # 2. Random Forest
        if self.rf_model is not None:
            pred_rf = int(self.rf_model.predict(X)[0])
            conf_rf = float(self.rf_model.predict_proba(X)[0][pred_rf])
            results["Random Forest Classifier"] = {
                "class": pred_rf,
                "regime": REGIME_NAMES.get(pred_rf, "UNKNOWN"),
                "confidence": round(conf_rf, 4)
            }

        # 3. Gradient Boosting
        if self.gb_model is not None:
            pred_gb = int(self.gb_model.predict(X)[0])
            conf_gb = float(self.gb_model.predict_proba(X)[0][pred_gb])
            results["Gradient Boosting Classifier"] = {
                "class": pred_gb,
                "regime": REGIME_NAMES.get(pred_gb, "UNKNOWN"),
                "confidence": round(conf_gb, 4)
            }

        # 4. Rule-Based Baseline
        pred_base, regime_base = self.rule_based_predict(peak_intensity)
        results["Rule-Based Baseline"] = {
            "class": pred_base,
            "regime": regime_base,
            "confidence": 1.0
        }

        return results

    def evaluate_partial_scan(self, partial_lux_array, target_lux=500.0, ambient_lux=40.0, conf_threshold=0.85):
        """
        Early safety evaluation across early scan points (POS_0 to POS_3).
        Enables rapid overexposure early-detection for museum protection.
        """
        if len(partial_lux_array) < 4:
            return None

        peak_partial = float(np.max(partial_lux_array[:4]))
        is_overexposed_early = peak_partial > target_lux

        regime = "OVEREXPOSURE_RISK" if is_overexposed_early else "NORMAL_DIFFUSE"
        confidence = 0.95 if is_overexposed_early else 0.85

        return {
            "early_stop": is_overexposed_early,
            "regime_class": 1 if is_overexposed_early else 0,
            "regime_name": regime,
            "confidence": confidence,
            "recommended_angle_id": 0 if is_overexposed_early else 3,
            "features": extract_partial_features(partial_lux_array, target_lux, ambient_lux)
        }
