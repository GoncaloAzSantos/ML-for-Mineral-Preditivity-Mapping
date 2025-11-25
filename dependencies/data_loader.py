import os
import numpy as np
import rasterio
from glob import glob
import pathlib

def load_data(config):
    """Load labels, mask, and feature data"""

    # Load training labels
    with rasterio.open(config["TRAINING_LABELS_PATH"]) as src:
        labels = src.read(1)
        profile = src.profile

    # Load mask if provided
    if config["MASK_PATH"] is not None:
        with rasterio.open(config["MASK_PATH"]) as msk:
            mask_data = msk.read(1)
            mask_valid = (mask_data == 1)
            if mask_valid.shape != labels.shape:
                raise ValueError(f"Mask shape {mask_valid.shape} does not match labels shape {labels.shape}")
    else:
        mask_valid = np.ones_like(labels, dtype=bool)

    # Create mask for labeled data and valid areas
    mask = (labels != -1) & mask_valid
    y = (labels[mask] == 1).astype(int)  # 1: mineralized, 0: not mineralized

    # Load features
    features, layer_names = load_features(config["FEATURES_DIR"])

    # Apply mask to features
    X = features[mask]

    # Handle NaNs
    X = handle_nans(X)

    return {
        "X": X,
        "y": y,
        "labels": labels,
        "mask": mask,
        "mask_valid": mask_valid,
        "features_full": features,
        "layer_names": layer_names,
        "profile": profile
    }


def load_features(features_dir):
    """Load feature rasters from directory or multi-band TIFF"""
    feature_path = pathlib.Path(features_dir)
    layer_names = []

    if feature_path.is_file():
        # Single multi-band TIFF
        with rasterio.open(str(feature_path)) as src:
            features = src.read().transpose(1, 2, 0)
            num_bands = src.count
            base_name = os.path.splitext(os.path.basename(feature_path))[0]
            layer_names = [f"{base_name}_band{i + 1}" for i in range(num_bands)]
            print(f"   Loaded multi-band TIFF with {num_bands} bands: {base_name}")
    elif feature_path.is_dir():
        # Multiple single-band TIFFs
        feature_files = sorted(glob(os.path.join(features_dir, "*.tif")))
        if len(feature_files) == 0:
            raise FileNotFoundError(f"No TIFF files found in {features_dir}")
        features = []
        for f in feature_files:
            with rasterio.open(f) as src:
                arr = src.read(1)
                features.append(arr)
                base_name = os.path.splitext(os.path.basename(f))[0]
                layer_names.append(base_name)
        features = np.stack(features, axis=-1)
        print(f"   Loaded {len(feature_files)} single-band TIFFs")
    else:
        raise FileNotFoundError(f"FEATURES_DIR path {features_dir} does not exist.")

    return features, layer_names


def handle_nans(X):
    """Handle NaN values in feature matrix"""
    if np.isnan(X).sum() > 0:
        print("   Warning: NaNs detected - replacing with feature means")
        col_means = np.nanmean(X, axis=0)
        nan_indices = np.where(np.isnan(X))
        X[nan_indices] = np.take(col_means, nan_indices[1])
    return X