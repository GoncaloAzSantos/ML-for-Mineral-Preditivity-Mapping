import os
import numpy as np
from dependencies import data_loader, feature_processor, model_trainer, predictor, visualizer

# --- CONFIGURATION ---
CONFIG = {
    "MASK_PATH": r"C:\\Users\\gonca\\OneDrive\\Ambiente de Trabalho\\Aramo512\\mask_512.tif",
    "TRAINING_LABELS_PATH": r"C:\\Users\\gonca\\OneDrive\\Ambiente de Trabalho\\Aramo512\\Training_data_512.tif",
    "FEATURES_DIR": r"C:\\Users\\gonca\\OneDrive\\Ambiente de Trabalho\\Aramo512\\dados_treino",
    "OUTPUT_PREDICTION_PATH": r"C:\\Users\\gonca\\OneDrive\\Ambiente de Trabalho\\Aramo512\\prediction_likelihood.tif",
    "DIAGNOSTICS_DIR": r"C:\\Users\\gonca\\OneDrive\\Ambiente de Trabalho\\Aramo512\\diagnostics",

    # Feature selection and model parameters
    "FEATURE_SELECTION": False,
    "N_FEATURES_TO_SELECT": 25,
    "FEATURE_SELECTION_THRESHOLD": -np.inf,

    # Model selection
    "MODEL_TYPE": ["RandomForest", "XGBoost", "CatBoost", "MLP", "SVM"],

    # RandomForest hyperparameters
    "N_ESTIMATORS": 200,
    "MAX_DEPTH": 20,
    "MIN_SAMPLES_SPLIT": 5,
    "MIN_SAMPLES_LEAF": 2,

    # XGBoost hyperparameters
    "XGB_MAX_DEPTH": 4,
    "XGB_LEARNING_RATE": 0.1,
    "XGB_N_ESTIMATORS": 300,
    "XGB_SUBSAMPLE": 0.9,
    "XGB_COLSAMPLE_BYTREE": 0.7,
    "XGB_GAMMA": 0.2,
    "XGB_REG_ALPHA": 0.1,
    "XGB_REG_LAMBDA": 0.5,
    "XGB_SCALE_POS_WEIGHT": 1.0,
    "XGB_EVAL_METRIC": "logloss",

    # CatBoost hyperparameters
    "CB_ITERATIONS": 500,
    "CB_LEARNING_RATE": 0.05,
    "CB_DEPTH": 6,
    "CB_L2_LEAF_REG": 3.0,
    "CB_EARLY_STOPPING_ROUNDS": 20,
    "CB_AUTO_CLASS_WEIGHTS": "Balanced",

    # MLP hyperparameters
    "MLP_HIDDEN_LAYER_SIZES": (256, 128, 64),
    "MLP_ACTIVATION": "relu",
    "MLP_ALPHA": 0.001,
    "MLP_BATCH_SIZE": 256,
    "MLP_LEARNING_RATE": "adaptive",
    "MLP_LEARNING_RATE_INIT": 0.001,
    "MLP_MAX_ITER": 1000,
    "MLP_EARLY_STOPPING": True,
    "MLP_VALIDATION_FRACTION": 0.15,
    "MLP_N_ITER_NO_CHANGE": 30,
    "MLP_TOL": 1e-5,
    "MLP_VERBOSE": False,
    
    # SVM hyperparameters
    "SVM_KERNEL": "linear",   # options: 'linear', 'rbf', 'poly', 'sigmoid'
    "SVM_C": 1.0,
    "SVM_GAMMA": "scale",     # for rbf/poly/sigmoid; 'scale' or 'auto' or float
    "SVM_CLASS_WEIGHT": None,  # e.g., 'balanced' or None
    "SVM_PROBABILITY": True,
    "TEST_SIZE": 0.3,
    "RANDOM_STATE": 42
}

def main():
    # Create diagnostics directory
    os.makedirs(CONFIG["DIAGNOSTICS_DIR"], exist_ok=True)
    # Also create subfolders for each model type
    os.makedirs(os.path.join(CONFIG["DIAGNOSTICS_DIR"], 'XGBoost'), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["DIAGNOSTICS_DIR"], 'RandomForest'), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["DIAGNOSTICS_DIR"], 'CatBoost'), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["DIAGNOSTICS_DIR"], 'MLP'), exist_ok=True)
    os.makedirs(os.path.join(CONFIG["DIAGNOSTICS_DIR"], 'SVM'), exist_ok=True)

    print("=== MINERAL PREDICTION PIPELINE ===")

    # Determine which model types to run
    mt = CONFIG.get("MODEL_TYPE", "RandomForest")
    if isinstance(mt, list):
        model_types = mt
    elif isinstance(mt, str) and mt.lower() == "both":
        model_types = ["XGBoost", "RandomForest"]
    else:
        model_types = [mt]

    output_dirs = []

    for requested_model in model_types:
        print(f"\n>>> Running pipeline for model: {requested_model}")

        # Local config per run
        run_config = dict(CONFIG)
        run_config["MODEL_TYPE"] = requested_model

        # 1. Load and prepare data
        print("\n1. Loading data...")
        data = data_loader.load_data(run_config)
        print(f"   Loaded {len(data['layer_names'])} features")
        print(f"   Mineralized pixels: {np.sum(data['y'] == 1)}")
        print(f"   Non-mineralized pixels: {np.sum(data['y'] == 0)}")

        # 2. Preprocess
        print("\n2. Preprocessing features...")
        processed_data = feature_processor.preprocess_features(data, run_config)

        # 3. Train
        print("\n3. Training model...")
        model_results = model_trainer.train_and_evaluate_model(processed_data, run_config)

        # 4. Predict
        print("\n4. Generating predictions...")
        prediction_results = predictor.generate_predictions(data, processed_data, model_results, run_config)

        # 5. Visualize
        print("\n5. Creating visualizations...")
        visualizer.create_all_visualizations(data, processed_data, model_results, prediction_results, run_config)

        # Track outputs using the requested model name for folder grouping
        output_dir = model_results.get('output_dir', run_config['DIAGNOSTICS_DIR'])
        requested_name = model_results.get('requested_model_name', requested_model)
        trained_name = model_results.get('model_name', requested_model)
        output_dirs.append((requested_name, output_dir))
        suffix = "" if requested_name == trained_name else f" (trained: {trained_name})"
        print(f"\n--- Completed for {requested_name}{suffix}. Outputs: {output_dir}")

    print("\n=== PROCESS COMPLETE ===")
    if len(output_dirs) > 1:
        print("All results saved to the following directories:")
        for name, od in output_dirs:
            print(f" - {name}: {od}")
    else:
        print(f"All results saved to: {output_dirs[0][1] if output_dirs else CONFIG['DIAGNOSTICS_DIR']}")

if __name__ == "__main__":
    main()