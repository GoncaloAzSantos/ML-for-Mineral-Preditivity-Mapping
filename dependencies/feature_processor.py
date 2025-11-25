import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def preprocess_features(data, config):
    """Preprocess features including scaling, splitting, and feature selection"""

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data["X"])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, data["y"],
        test_size=config["TEST_SIZE"],
        stratify=data["y"],
        random_state=config["RANDOM_STATE"]
    )

    # Apply SMOTE for class imbalance
    smote = SMOTE(sampling_strategy=0.5, random_state=config["RANDOM_STATE"])
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Feature selection
    selector, selected_indices = perform_feature_selection(
        X_train_res, y_train_res, data["layer_names"], config
    )

    # Apply feature selection if performed
    if selector is not None or selected_indices is not None:
        X_train_res, X_train, X_test, layer_names = apply_feature_selection(
            X_train_res, X_train, X_test, data["layer_names"], selector, selected_indices
        )
    else:
        layer_names = data["layer_names"]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_res": X_train_res,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_res": y_train_res,
        "scaler": scaler,
        "selector": selector,
        "selected_indices": selected_indices,
        "layer_names": layer_names
    }


def perform_feature_selection(X_train_res, y_train_res, layer_names, config):
    """Perform feature selection if enabled"""
    if not config["FEATURE_SELECTION"]:
        return None, None

    print("   Performing feature selection...")
    selector_model = RandomForestClassifier(
        n_estimators=100,
        random_state=config["RANDOM_STATE"],
        n_jobs=-1
    )
    selector_model.fit(X_train_res, y_train_res)
    importances = selector_model.feature_importances_

    if config["N_FEATURES_TO_SELECT"] is not None:
        # Select top N features
        sorted_idx = np.argsort(importances)[::-1]
        selected_indices = sorted_idx[:config["N_FEATURES_TO_SELECT"]]
        selector = None
    else:
        # Use threshold-based selection
        selector = SelectFromModel(selector_model, threshold=config["FEATURE_SELECTION_THRESHOLD"])
        selector.fit(X_train_res, y_train_res)
        selected_indices = selector.get_support(indices=True)

    return selector, selected_indices


def apply_feature_selection(X_train_res, X_train, X_test, layer_names, selector, selected_indices):
    """Apply feature selection to datasets"""
    if selector is not None:
        # Threshold-based selection
        X_train_res = selector.transform(X_train_res)
        X_train = selector.transform(X_train)
        X_test = selector.transform(X_test)
        selected_layer_names = [layer_names[i] for i in selected_indices]
    else:
        # Top N features selection
        X_train_res = X_train_res[:, selected_indices]
        X_train = X_train[:, selected_indices]
        X_test = X_test[:, selected_indices]
        selected_layer_names = [layer_names[i] for i in selected_indices]

    print(f"   Selected {len(selected_layer_names)} features")
    return X_train_res, X_train, X_test, selected_layer_names