import numpy as np
import rasterio
import os

def generate_predictions(data, processed_data, model_results, config):
    """Generate predictions for the entire study area"""

    print("   Preparing full map prediction...")

    # Prepare features for prediction
    flat_features = data["features_full"].reshape(-1, data["features_full"].shape[-1])

    # Impute NaNs in full-map features before scaling (needed for models like MLP)
    if np.isnan(flat_features).any():
        try:
            col_means = getattr(processed_data["scaler"], "mean_", None)
        except Exception:
            col_means = None
        if col_means is None:
            # Fallback to column means from the full-map data itself
            col_means = np.nanmean(flat_features, axis=0)
            # If any column is entirely NaN, replace remaining NaNs with zeros
            col_means = np.nan_to_num(col_means, nan=0.0)

        nan_rows, nan_cols = np.where(np.isnan(flat_features))
        if nan_rows.size > 0:
            flat_features[nan_rows, nan_cols] = col_means[nan_cols]

    # Scale with the training scaler
    flat_features_scaled = processed_data["scaler"].transform(flat_features)

    # Apply feature selection if used
    if processed_data["selector"] is not None or processed_data["selected_indices"] is not None:
        flat_features_scaled = apply_feature_selection_to_map(
            flat_features_scaled, processed_data["selector"], processed_data["selected_indices"]
        )

    # Predict probabilities
    print("   Predicting probabilities...")
    likelihood = model_results["calibrated"].predict_proba(flat_features_scaled)[:, 1]
    likelihood_map = likelihood.reshape(data["features_full"].shape[:2])

    # Apply mask
    if config["MASK_PATH"] is not None:
        likelihood_map[~data["mask_valid"]] = data["profile"].get('nodata', -1)

    # Create binary map
    binary_map = (likelihood_map >= model_results["optimal_threshold"]).astype(np.uint8)
    if config["MASK_PATH"] is not None:
        binary_map[~data["mask_valid"]] = 0

    # Save outputs into model-specific diagnostics folder
    save_prediction_maps(likelihood_map, binary_map, data["profile"], model_results["output_dir"])

    return {
        "likelihood_map": likelihood_map,
        "binary_map": binary_map
    }


def apply_feature_selection_to_map(flat_features, selector, selected_indices):
    """Apply feature selection to full map features"""
    if selector is not None:
        return selector.transform(flat_features)
    else:
        return flat_features[:, selected_indices]


def save_prediction_maps(likelihood_map, binary_map, profile, output_dir):
    """Save prediction maps to files in the specified output directory"""
    # Save likelihood map
    likelihood_path = os.path.join(output_dir, 'prediction_likelihood.tif')
    profile.update(dtype=rasterio.float32, count=1, compress='LZW', nodata=-1)
    with rasterio.open(likelihood_path, 'w', **profile) as dst:
        dst.write(likelihood_map.astype(np.float32), 1)

    # Save binary map
    binary_path = os.path.join(output_dir, 'prediction_binary.tif')
    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
    with rasterio.open(binary_path, 'w', **profile) as dst:
        dst.write(binary_map, 1)

    print(f"   Continuous prediction saved to: {likelihood_path}")
    print(f"   Binary prediction saved to: {binary_path}")