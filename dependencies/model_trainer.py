import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score, \
    precision_recall_curve, f1_score, roc_curve, auc
import joblib
import os

def train_and_evaluate_model(processed_data, config):
    """Train, calibrate and evaluate the model.

    Note: All artifacts are saved under the diagnostics subfolder named after the
    requested model in config["MODEL_TYPE"], even if training falls back to a
    different algorithm. This keeps per-request runs grouped consistently.
    """

    # Train model (returns estimator and resolved model_name)
    model, model_name = train_model(processed_data["X_train_res"], processed_data["y_train_res"], config)

    # Determine output directory by requested model (not the fallback one)
    requested_model = str(config.get("MODEL_TYPE", model_name))
    base_dir = config["DIAGNOSTICS_DIR"]
    output_dir = os.path.join(base_dir, requested_model)
    os.makedirs(output_dir, exist_ok=True)

    # Calibrate probabilities
    calibrated = calibrate_model(model, processed_data["X_train_res"], processed_data["y_train_res"], config)

    # Optimize threshold
    optimal_threshold = optimize_threshold(calibrated, processed_data["X_train"], processed_data["y_train"])

    # Evaluate model
    evaluation_results = evaluate_model(calibrated, optimal_threshold, processed_data["X_test"],
                                        processed_data["y_test"])

    # Save model
    save_model(calibrated, optimal_threshold, processed_data, output_dir, model_name)

    # If fallback occurred, record it for transparency
    if requested_model != model_name:
        try:
            with open(os.path.join(output_dir, 'fallback_info.txt'), 'w') as f:
                f.write(f"Requested model: {requested_model}\n")
                f.write(f"Trained model: {model_name}\n")
                f.write("Reason: Requested model unavailable or failed during training; "
                        "automatic fallback applied.\n")
        except Exception:
            pass

    # Collect CatBoost training history if available
    catboost_evals_result = None
    if model_name == 'CatBoost':
        try:
            if hasattr(model, 'get_evals_result'):
                catboost_evals_result = model.get_evals_result()
        except Exception:
            catboost_evals_result = None

    return {
        "model": model,
        "calibrated": calibrated,
        "optimal_threshold": optimal_threshold,
        "evaluation_results": evaluation_results,
        "model_name": model_name,
        "requested_model_name": requested_model,
        "output_dir": output_dir,
        # Add y_test to model_results for visualization
        "y_test": processed_data["y_test"],
        "catboost_evals_result": catboost_evals_result
    }


def train_model(X_train, y_train, config):
    """Train model based on config.MODEL_TYPE. Falls back to RandomForest on failure."""
    model_type = (config.get("MODEL_TYPE") or "RandomForest").strip()

    if model_type.lower() == "mlp":
        try:
            # Import locally to avoid hard dependency at module import time
            from sklearn.neural_network import MLPClassifier
            model = MLPClassifier(
                hidden_layer_sizes=config.get("MLP_HIDDEN_LAYER_SIZES", (256, 128, 64)),
                activation=config.get("MLP_ACTIVATION", "relu"),
                solver='adam',
                alpha=config.get("MLP_ALPHA", 0.001),
                batch_size=config.get("MLP_BATCH_SIZE", 256),
                learning_rate=config.get("MLP_LEARNING_RATE", 'adaptive'),
                learning_rate_init=config.get("MLP_LEARNING_RATE_INIT", 0.001),
                max_iter=config.get("MLP_MAX_ITER", 1000),
                early_stopping=config.get("MLP_EARLY_STOPPING", True),
                validation_fraction=config.get("MLP_VALIDATION_FRACTION", 0.15),
                n_iter_no_change=config.get("MLP_N_ITER_NO_CHANGE", 30),
                tol=config.get("MLP_TOL", 1e-5),
                random_state=config.get("RANDOM_STATE", 42),
                verbose=config.get("MLP_VERBOSE", False)
            )
            model.fit(X_train, y_train)
            return model, 'MLP'
        except Exception as e:
            print(f"MLP training failed ({e}). Falling back to RandomForest...")

    if model_type.lower() == "xgboost":
        try:
            from importlib import import_module
            xgb_mod = import_module('xgboost')
            XGBClassifier = xgb_mod.XGBClassifier
            model = XGBClassifier(
                max_depth=config.get("XGB_MAX_DEPTH", 4),
                learning_rate=config.get("XGB_LEARNING_RATE", 0.1),
                n_estimators=config.get("XGB_N_ESTIMATORS", 300),
                subsample=config.get("XGB_SUBSAMPLE", 0.9),
                colsample_bytree=config.get("XGB_COLSAMPLE_BYTREE", 0.7),
                gamma=config.get("XGB_GAMMA", 0.0),
                reg_alpha=config.get("XGB_REG_ALPHA", 0.0),
                reg_lambda=config.get("XGB_REG_LAMBDA", 1.0),
                scale_pos_weight=config.get("XGB_SCALE_POS_WEIGHT", 1.0),
                eval_metric=config.get("XGB_EVAL_METRIC", "logloss"),
                random_state=config.get("RANDOM_STATE", 42),
                n_jobs=-1,
                tree_method="hist"
            )
            model.fit(X_train, y_train, verbose=False)
            return model, 'XGBoost'
        except Exception as e:
            print(f"XGBoost training failed ({e}). Falling back to RandomForest...")

    if model_type.lower() == "catboost":
        try:
            from importlib import import_module
            from sklearn.model_selection import train_test_split
            cb_mod = import_module('catboost')
            CatBoostClassifier = cb_mod.CatBoostClassifier
            # Create a small validation split for early stopping/learning curves
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=config.get("RANDOM_STATE", 42)
            )
            model = CatBoostClassifier(
                iterations=config.get("CB_ITERATIONS", 500),
                learning_rate=config.get("CB_LEARNING_RATE", 0.05),
                depth=config.get("CB_DEPTH", 6),
                l2_leaf_reg=config.get("CB_L2_LEAF_REG", 3.0),
                random_seed=config.get("RANDOM_STATE", 42),
                verbose=100,
                auto_class_weights=config.get("CB_AUTO_CLASS_WEIGHTS", 'Balanced'),
                loss_function='Logloss',
                early_stopping_rounds=config.get("CB_EARLY_STOPPING_ROUNDS", 20),
                allow_writing_files=False,
                use_best_model=True
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=100)
            return model, 'CatBoost'
        except Exception as e:
            print(f"CatBoost training failed ({e}). Falling back to RandomForest...")

    if model_type.lower() == "svm":
        try:
            # Import locally to avoid hard dependency at module import time
            from sklearn.svm import SVC
            svc_kwargs = {
                'kernel': config.get('SVM_KERNEL', 'linear'),
                'C': config.get('SVM_C', 1.0),
                'probability': config.get('SVM_PROBABILITY', True),
                'random_state': config.get('RANDOM_STATE', 42)
            }
            # Only add gamma if relevant kernel
            if str(svc_kwargs['kernel']).lower() in ('rbf', 'poly', 'sigmoid'):
                svc_kwargs['gamma'] = config.get('SVM_GAMMA', 'scale')
            # Optional class_weight
            if config.get('SVM_CLASS_WEIGHT', None) is not None:
                svc_kwargs['class_weight'] = config.get('SVM_CLASS_WEIGHT')

            model = SVC(**svc_kwargs)
            model.fit(X_train, y_train)
            return model, 'SVM'
        except Exception as e:
            print(f"SVM training failed ({e}). Falling back to RandomForest...")

    # Default or fallback: RandomForest
    model = RandomForestClassifier(
        n_estimators=config["N_ESTIMATORS"],
        max_depth=config["MAX_DEPTH"],
        min_samples_split=config["MIN_SAMPLES_SPLIT"],
        min_samples_leaf=config["MIN_SAMPLES_LEAF"],
        class_weight='balanced',
        random_state=config["RANDOM_STATE"],
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model, 'RandomForest'


def calibrate_model(model, X_train, y_train, config):
    """Calibrate model probabilities.

    Special handling for CatBoost: CalibratedClassifierCV clones the estimator and fits
    it internally without an eval_set. CatBoost with use_best_model=True requires a
    non-empty eval_set, which would raise an error. To avoid this, we provide a fresh
    CatBoostClassifier configured with use_best_model=False and no early stopping for
    calibration, while still training the main model with early stopping and eval_set.
    """

    base_estimator = model

    # Detect CatBoost model and prepare a calibration-friendly estimator
    try:
        model_class_name = type(model).__name__
    except Exception:
        model_class_name = ""

    if model_class_name.lower().startswith('catboost'):
        try:
            from importlib import import_module
            cb_mod = import_module('catboost')
            CatBoostClassifier = cb_mod.CatBoostClassifier

            # Build a fresh CatBoost estimator with similar core params but no eval_set requirements
            base_estimator = CatBoostClassifier(
                iterations=config.get("CB_ITERATIONS", 500),
                learning_rate=config.get("CB_LEARNING_RATE", 0.05),
                depth=config.get("CB_DEPTH", 6),
                l2_leaf_reg=config.get("CB_L2_LEAF_REG", 3.0),
                random_seed=config.get("RANDOM_STATE", 42),
                verbose=False,
                auto_class_weights=config.get("CB_AUTO_CLASS_WEIGHTS", 'Balanced'),
                loss_function='Logloss',
                allow_writing_files=False,
                use_best_model=False,
                early_stopping_rounds=None
            )
        except Exception:
            # If for any reason CatBoost import or construction fails, fall back to the given model
            base_estimator = model

    calibrated = CalibratedClassifierCV(base_estimator, cv=3, method='isotonic')
    calibrated.fit(X_train, y_train)
    return calibrated


def optimize_threshold(calibrated, X_train, y_train):
    """Find optimal classification threshold"""
    y_probs_train = calibrated.predict_proba(X_train)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, y_probs_train)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.nanargmax(f1_scores)
    return thresholds[optimal_idx]


def evaluate_model(calibrated, threshold, X_test, y_test):
    """Evaluate model performance"""
    y_probs = calibrated.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_probs)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "y_probs": y_probs,
        "y_pred": y_pred,
        "roc_auc": roc_auc,
        "balanced_accuracy": bal_acc,
        "f1_score": f1,
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

def save_model(calibrated, optimal_threshold, processed_data, output_dir, model_name):
    """Save trained model and metadata"""
    model_data = {
        'model': calibrated,
        'threshold': optimal_threshold,
        'feature_names': processed_data["layer_names"],
        'model_type': model_name.lower(),
        'scaler': processed_data["scaler"],
        'selector': processed_data["selector"]
    }
    joblib.dump(model_data, os.path.join(output_dir, 'best_model.pkl'))