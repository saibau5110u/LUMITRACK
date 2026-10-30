# LumiTrack: Machine Learning Subsystem (Variant 10.03)

Intelligent Light Mapping and Museum Exposure Management Machine Learning Engine.

## 1. Directory Structure

```
02_Machine_Learning/
│
├── data/
│   ├── data_dictionary.md          # Dataset schema and column definitions
│   ├── generate_dataset.py         # Synthetic experimental scan generator
│   └── seed_scan_dataset.csv       # 600 physical scan records across 3 lighting regimes
│
├── models/
│   ├── best_model.pkl              # Best-performing model selected via 5-fold CV (Random Forest)
│   ├── random_forest.pkl           # Random Forest Classifier (100 estimators)
│   ├── logistic_regression.pkl     # Ridge Logistic Regression (L2 regularization)
│   ├── gradient_boosting.pkl       # Gradient Boosting Classifier (100 stages)
│   ├── scaler.pkl                  # StandardScaler fitted on training partition
│   └── metrics_summary.json        # Serialized evaluation metrics and comparison table
│
├── src/
│   ├── feature_extraction.py       # Computes exactly 13 physical lighting features
│   ├── ridge_logistic_regression.py# Algorithm 1: Ridge Logistic Regression standalone script
│   ├── random_forest.py            # Algorithm 2: Random Forest Classifier standalone script
│   ├── gradient_boosting.py        # Algorithm 3: Gradient Boosting Classifier standalone script
│   ├── rule_based_baseline.py      # Algorithm 4: Rule-Based Baseline standalone script
│   ├── train_models.py             # 4-Algorithms unified training & 5-fold CV pipeline
│   ├── evaluate_models.py          # Standalone evaluation & verification engine
│   └── inference.py                # Real-time inference engine for host app & dashboard
│
└── README.md                       # This document
```

---

## 2. The Four Algorithms

The LumiTrack ML subsystem implements strictly four algorithms:

1. **Ridge Logistic Regression**
   - Implemented via `sklearn.linear_model.LogisticRegression(C=1.0)`.
   - Uses L2 (Ridge) penalty to prevent overfitting.
   - Requires feature normalization using `StandardScaler`.
   - Computes multiclass probabilities using the softmax function.

2. **Random Forest Classifier**
   - Implemented via `sklearn.ensemble.RandomForestClassifier(n_estimators=100, max_depth=6)`.
   - Ensembles 100 decision trees trained on bootstrap feature subsets.
   - Outputs final classification via majority voting across trees.
   - Selected as the **Best-Performing Model** based on 5-fold cross-validation.

3. **Gradient Boosting Classifier**
   - Implemented via `sklearn.ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4)`.
   - Builds 100 decision trees sequentially, with each tree minimizing the residual errors of preceding trees.

4. **Rule-Based Baseline (Heuristic)**
   - Simple deterministic threshold function:
     ```python
     if peak_intensity > 400.0:
         return "OVEREXPOSURE_RISK"
     else:
         return "NORMAL_DIFFUSE"
     ```
   - **Crucial Distinction**: This is a simple binary heuristic, not a trained ML classifier. It fails to identify Specular Confusers (transient glints/reflections), producing false alarms.

---

## 3. The 13 Extracted Physical Features

All models receive the exact same 13 physical features in this fixed sequence:

| # | Feature Name | Unit | Physical Description |
|---|:---|:---:|:---|
| 1 | `peak_intensity` | Lux | Peak light intensity detected across 7 angular positions |
| 2 | `peak_angle_deg` | Degrees | Servo angle where peak intensity occurred (15° to 165°) |
| 3 | `min_intensity` | Lux | Lowest light level observed across the scan |
| 4 | `mean_intensity` | Lux | Average light flux across all 7 points |
| 5 | `std_intensity` | Lux | Spatial standard deviation (measures flux heterogeneity) |
| 6 | `curve_asymmetry` | Ratio [-1, 1] | Balance between right wing (115-165°) and left wing (15-65°) |
| 7 | `gradient_mean` | Lux/deg | Mean spatial rate of change between adjacent angles |
| 8 | `gradient_max` | Lux/deg | Maximum spatial intensity jump between adjacent angles |
| 9 | `contrast_ratio` | Ratio [0, 1] | Dynamic contrast ratio: `(peak - min) / (peak + min)` |
| 10 | `exposure_integral` | Lux·deg | Trapezoidal numerical integral (cumulative spatial flux) |
| 11 | `ratio_to_target` | Ratio | Ratio of peak intensity to operator potentiometer limit |
| 12 | `ratio_to_ambient` | Ratio | Ratio of peak intensity to dark reference baseline |
| 13 | `partial_slope` | Lux/deg | Early slope between POS_0 (15°) and neutral home (90°) |

---

## 4. Execution Commands

### Run Each Algorithm Individually:
```bash
# Algorithm 1: Ridge Logistic Regression
python src/ridge_logistic_regression.py

# Algorithm 2: Random Forest Classifier (Best Model)
python src/random_forest.py

# Algorithm 3: Gradient Boosting Classifier
python src/gradient_boosting.py

# Algorithm 4: Rule-Based Baseline Heuristic
python src/rule_based_baseline.py
```

### Train All 4 Algorithms & 5-Fold Cross-Validation:
```bash
python 02_Machine_Learning/src/train_models.py
```
- Performs 80/20 stratified split.
- Executes 5-fold cross-validation on training data.
- Selects best model (`Random Forest Classifier`).
- Evaluates on held-out test data.
- Serializes models to `02_Machine_Learning/models/`.

### Run Standalone Model Evaluation:
```bash
python 02_Machine_Learning/src/evaluate_models.py
```
- Loads saved models and scaler.
- Computes Accuracy, Precision, Recall, Macro-F1, and Confusion Matrices.
- Displays comparative analysis against the Rule-Based Baseline.

### Real-Time Inference:
Used directly by the host application and dashboard:
```python
from inference import LumiTrackPredictor

predictor = LumiTrackPredictor()
result = predictor.evaluate_full_scan([80.0, 110.0, 150.0, 175.0, 140.0, 105.0, 75.0], target_lux=400.0)
print(result["regime_name"], result["confidence"])
```
