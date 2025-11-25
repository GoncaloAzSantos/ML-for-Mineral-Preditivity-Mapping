import os
import numpy as np
import seaborn as sns
import matplotlib as mpl
# Force a non-interactive backend to avoid Tkinter-related errors on headless/CLI runs
try:
    mpl.use('Agg')
except Exception:
    # If setting backend fails for any reason, proceed with default; plots will still attempt to save
    pass
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def create_all_visualizations(data, processed_data, model_results, prediction_results, config):
    """Create all diagnostic visualizations"""

    # Feature importance plot
    create_feature_importance_plot(data, processed_data, model_results)

    # Feature correlation analysis
    if len(processed_data["layer_names"]) > 0:
        create_feature_correlation_plot(data, processed_data, model_results)

    # ROC curve
    create_roc_curve(model_results)

    # Diagnostic report
    create_diagnostic_report(data, model_results, prediction_results)

    # Save evaluation metrics
    save_evaluation_metrics(model_results)

    # Additional thesis/general visualizations
    create_precision_recall_plot(model_results)
    create_calibration_plot(model_results)
    create_class_distribution_plot(processed_data, model_results)
    create_pca_plot(processed_data, model_results)

    # CatBoost-specific visualizations
    if model_results.get("model_name") == 'CatBoost':
        create_catboost_specific_visualizations(processed_data, model_results)

    # MLP-specific visualizations
    if model_results.get("model_name") == 'MLP':
        create_mlp_specific_visualizations(model_results)


def create_feature_importance_plot(data, processed_data, model_results):
    """Create feature importance visualization"""
    output_dir = model_results["output_dir"]

    if processed_data["selected_indices"] is not None or processed_data["selector"] is not None:
        # Use the feature selection model for importances
        importance_model = RandomForestClassifier(n_estimators=100, random_state=42)
        importance_model.fit(processed_data["X_train_res"], processed_data["y_train_res"])
        importances = importance_model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in importance_model.estimators_], axis=0)
        layer_names = processed_data["layer_names"]

        plt.figure(figsize=(12, 8))
        sorted_idx = importances.argsort()[::-1]
        sorted_importances = importances[sorted_idx]
        sorted_std = std[sorted_idx]
        sorted_names = np.array([layer_names[i] for i in sorted_idx])

        plt.bar(range(len(importances)), sorted_importances, color='steelblue',
                yerr=sorted_std, align="center")
        plt.xticks(range(len(importances)), sorted_names, rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.title("Feature Importances (Selected Features)")

    else:
        # Plot all features
        importance_model = RandomForestClassifier(n_estimators=100, random_state=42)
        importance_model.fit(processed_data["X_train_res"], processed_data["y_train_res"])
        importances = importance_model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in importance_model.estimators_], axis=0)

        plt.figure(figsize=(12, 8))
        sorted_idx = importances.argsort()[::-1]
        sorted_importances = importances[sorted_idx]
        sorted_std = std[sorted_idx]
        sorted_names = np.array([data["layer_names"][i] for i in sorted_idx])

        plt.bar(range(len(importances)), sorted_importances, color='steelblue',
                yerr=sorted_std, align="center")
        plt.xticks(range(len(importances)), sorted_names, rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Importance Score")
        plt.title("Feature Importances (All Features)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()


def create_feature_correlation_plot(data, processed_data, model_results):
    """Create feature-target correlation plot"""
    output_dir = model_results["output_dir"]

    # Get features for correlation analysis
    if processed_data["selected_indices"] is not None:
        X_corr = data["X"][:, processed_data["selected_indices"]]
        display_names = processed_data["layer_names"]
    else:
        X_corr = data["X"]
        display_names = data["layer_names"]

    # Compute correlations
    corrs = []
    p_values = []
    for i in range(X_corr.shape[1]):
        r, p = pearsonr(X_corr[:, i], data["y"])
        corrs.append(r)
        p_values.append(p)

    # Sort by absolute correlation
    abs_corrs = np.abs(corrs)
    sorted_idx = np.argsort(abs_corrs)[::-1]
    sorted_corrs = np.array(corrs)[sorted_idx]
    sorted_p_values = np.array(p_values)[sorted_idx]
    sorted_names = np.array(display_names)[sorted_idx]

    # Create significance indicators
    significance = ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    for p in sorted_p_values]

    # Create plot
    plt.figure(figsize=(12, 8))
    colors = ['steelblue' if c >= 0 else 'indianred' for c in sorted_corrs]
    bars = plt.barh(range(len(sorted_corrs)), sorted_corrs, color=colors)

    # Add significance annotations
    for i, (corr, sig) in enumerate(zip(sorted_corrs, significance)):
        x_pos = 0.05 if corr >= 0 else -0.05
        plt.text(x_pos, i, sig, ha='center' if corr < 0 else 'left',
                 va='center', fontsize=12, fontweight='bold')

    plt.yticks(range(len(sorted_corrs)), [f"{name}" for name in sorted_names])
    plt.xlabel("Pearson Correlation Coefficient")
    plt.title("Feature-Target Correlation with Statistical Significance")
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Create legend
    legend_elements = [
        mpl.patches.Patch(color='steelblue', label='Positive Correlation'),
        mpl.patches.Patch(color='indianred', label='Negative Correlation'),
        mpl.lines.Line2D([0], [0], marker='', label='Significance levels:'),
        mpl.lines.Line2D([0], [0], marker='', label='*** p < 0.001'),
        mpl.lines.Line2D([0], [0], marker='', label='**  p < 0.01'),
        mpl.lines.Line2D([0], [0], marker='', label='*   p < 0.05')
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlation.png'), dpi=300)
    plt.close()


def create_roc_curve(model_results):
    """Create ROC curve plot"""
    eval_results = model_results["evaluation_results"]
    y_test = model_results["y_test"]  # Get y_test from model_results
    output_dir = model_results["output_dir"]

    fpr, tpr, _ = roc_curve(y_test, eval_results["y_probs"])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.grid(True)

    # Add point for optimal threshold
    optimal_idx = np.argmin(np.sqrt((1 - tpr) ** 2 + fpr ** 2))
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], marker='o', color='red', s=100,
                label=f'Optimal Threshold: {model_results["optimal_threshold"]:.3f}')

    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
    plt.close()


def create_diagnostic_report(data, model_results, prediction_results):
    """Create comprehensive diagnostic report"""
    output_dir = model_results["output_dir"]
    plt.figure(figsize=(15, 10))

    # 1. Prediction map
    plt.subplot(2, 2, 1)
    plt.imshow(prediction_results["likelihood_map"], cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Probability')
    plt.title("Mineralization Likelihood")

    # 2. Binary classification
    plt.subplot(2, 2, 2)
    plt.imshow(prediction_results["binary_map"], cmap='jet')
    plt.title(f"Binary Classification (Threshold={model_results['optimal_threshold']:.2f})")

    # 3. Training labels
    plt.subplot(2, 2, 3)
    plt.imshow(data["labels"], cmap='jet', vmin=-1, vmax=1)
    plt.title("Training Labels")

    # 4. Confusion matrix
    plt.subplot(2, 2, 4)
    cm = model_results["evaluation_results"]["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Mineralized', 'Mineralized'],
                yticklabels=['Not Mineralized', 'Mineralized'])
    plt.title(f"Confusion Matrix (F1={model_results['evaluation_results']['f1_score']:.2f})")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'diagnostic_report.png'), dpi=300)
    plt.close()


def save_evaluation_metrics(model_results):
    """Save evaluation metrics to file"""
    eval_results = model_results["evaluation_results"]
    output_dir = model_results["output_dir"]

    with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(f"Optimal Threshold: {model_results['optimal_threshold']:.4f}\n")
        f.write("Classification Report:\n")
        f.write(eval_results["classification_report"])
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(eval_results["confusion_matrix"]))
        f.write(f"\nROC AUC: {eval_results['roc_auc']:.4f}\n")
        f.write(f"Balanced Accuracy: {eval_results['balanced_accuracy']:.4f}\n")


def create_precision_recall_plot(model_results):
    """Create Precision-Recall curve and save AUPRC"""
    output_dir = model_results["output_dir"]
    y_test = model_results["y_test"]
    y_probs = model_results["evaluation_results"]["y_probs"]

    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AUC = {pr_auc:.4f})')
    plt.fill_between(recall, precision, alpha=0.2, color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300)
    plt.close()


def create_calibration_plot(model_results):
    """Create calibration (reliability) curve"""
    output_dir = model_results["output_dir"]
    y_test = model_results["y_test"]
    y_probs = model_results["evaluation_results"]["y_probs"]

    prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10, strategy='quantile')

    plt.figure(figsize=(10, 8))
    plt.plot(prob_pred, prob_true, 's-', label='Model Calibration')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_curve.png'), dpi=300)
    plt.close()


def create_class_distribution_plot(processed_data, model_results):
    """Plot class distribution after SMOTE"""
    output_dir = model_results["output_dir"]
    y_train_res = processed_data["y_train_res"]

    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_train_res)
    plt.title('Class Distribution After SMOTE')
    plt.xlabel('Class (0: Non-mineralized, 1: Mineralized)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
    plt.close()


def create_pca_plot(processed_data, model_results):
    """2D PCA scatter of feature space"""
    output_dir = model_results["output_dir"]
    X_train_res = processed_data["X_train_res"]
    y_train_res = processed_data["y_train_res"]

    try:
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_train_res)

        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[y_train_res == 0, 0], X_pca[y_train_res == 0, 1],
                    alpha=0.5, label='Non-mineralized', color='blue')
        plt.scatter(X_pca[y_train_res == 1, 0], X_pca[y_train_res == 1, 1],
                    alpha=0.5, label='Mineralized', color='red')
        plt.title('Feature Space Visualization (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_space_pca.png'), dpi=300)
        plt.close()
    except Exception:
        # PCA failed; skip silently
        pass


def create_catboost_specific_visualizations(processed_data, model_results):
    """Create CatBoost-specific importance and learning history plots"""
    output_dir = model_results["output_dir"]
    model = model_results["model"]

    # Feature importance
    try:
        if hasattr(model, 'get_feature_importance'):
            feature_importance = model.get_feature_importance()
            sorted_idx = np.argsort(feature_importance)[::-1]
            sorted_importance = np.array(feature_importance)[sorted_idx]
            layer_names = processed_data["layer_names"]
            sorted_names = np.array(layer_names)[sorted_idx]

            plt.figure(figsize=(12, 8))
            plt.bar(range(len(sorted_importance)), sorted_importance, color='steelblue')
            plt.xticks(range(len(sorted_importance)), sorted_names, rotation=90)
            plt.xlabel("Features")
            plt.ylabel("Importance Score")
            plt.title("CatBoost Feature Importances")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'catboost_feature_importance.png'), dpi=300)
            plt.close()
    except Exception:
        pass

    # Training history and convergence analysis
    results = model_results.get('catboost_evals_result')
    if results:
        # Attempt to find Logloss curve
        loss_curve = None
        accuracy_curve = None
        scope_key = None
        if 'validation' in results and 'Logloss' in results['validation']:
            loss_curve = results['validation']['Logloss']
            scope_key = 'validation'
            if 'Accuracy' in results['validation']:
                accuracy_curve = results['validation']['Accuracy']
        elif 'learn' in results and 'Logloss' in results['learn']:
            loss_curve = results['learn']['Logloss']
            scope_key = 'learn'
            if 'Accuracy' in results['learn']:
                accuracy_curve = results['learn']['Accuracy']
        else:
            for key in results:
                if 'Logloss' in results[key]:
                    loss_curve = results[key]['Logloss']
                    scope_key = key
                    if 'Accuracy' in results[key]:
                        accuracy_curve = results[key]['Accuracy']
                    break

        if loss_curve is not None and len(loss_curve) > 0:
            plt.figure(figsize=(12, 10))
            # Loss
            plt.subplot(2, 1, 1)
            plt.plot(loss_curve, 'b-', linewidth=2, label=f'Logloss ({scope_key})')
            plt.title('CatBoost Training History')
            plt.ylabel('Loss')
            plt.xlabel('Iteration')
            plt.legend()
            plt.grid(True)

            # Accuracy if available
            if accuracy_curve is not None:
                plt.subplot(2, 1, 2)
                plt.plot(accuracy_curve, 'g-', linewidth=2, label=f'Accuracy ({scope_key})')
                plt.ylabel('Accuracy')
                plt.xlabel('Iteration')
                plt.legend()
                plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'catboost_training_history.png'), dpi=300)
            plt.close()

            # Convergence analysis
            try:
                final_loss = loss_curve[-1]
                min_loss = min(loss_curve)
                loss_reduction = 100.0 * (loss_curve[0] - final_loss) / max(loss_curve[0], 1e-12)
                best_iteration = getattr(model, 'best_iteration_', None)
                max_iterations = getattr(model, 'tree_count_', None) or len(loss_curve)
                early_stop = best_iteration is not None and max_iterations is not None and best_iteration < max_iterations - 1

                with open(os.path.join(output_dir, 'catboost_convergence_analysis.txt'), 'w') as f:
                    f.write("=== CATBOOST CONVERGENCE ANALYSIS ===\n")
                    f.write(f"Final training loss: {final_loss:.6f}\n")
                    f.write(f"Minimum training loss: {min_loss:.6f}\n")
                    f.write(f"Total loss reduction: {loss_reduction:.2f}%\n")
                    f.write(f"Training iterations: {len(loss_curve)}\n")
                    f.write(f"Early stopping triggered: {'Yes' if early_stop else 'No'}\n")
                    if best_iteration is not None:
                        f.write(f"Best iteration: {best_iteration}\n")

                    if accuracy_curve is not None and len(accuracy_curve) > 0:
                        initial_acc = accuracy_curve[0]
                        final_acc = accuracy_curve[-1]
                        max_acc = max(accuracy_curve)
                        acc_impr = 100.0 * (final_acc - initial_acc) / max(abs(initial_acc), 1e-12)
                        f.write("\nAccuracy Metrics:\n")
                        f.write(f"Initial accuracy: {initial_acc:.4f}\n")
                        f.write(f"Final accuracy: {final_acc:.4f}\n")
                        f.write(f"Maximum accuracy: {max_acc:.4f}\n")
                        f.write(f"Accuracy improvement: {acc_impr:.2f}%\n")

                    f.write("\n=== TRAINING INSIGHTS ===\n")
                    f.write(f"Initial loss: {loss_curve[0]:.4f}\n")
                    f.write(f"Final loss: {final_loss:.4f}\n")
                    f.write(f"Training iterations: {len(loss_curve)}\n")

                    if loss_reduction > 95:
                        f.write("\nConvergence Status: Excellent - Loss reduced by >95%\n")
                    elif loss_reduction > 90:
                        f.write("\nConvergence Status: Good - Loss reduced by >90%\n")
                    elif loss_reduction > 80:
                        f.write("\nConvergence Status: Fair - Loss reduced by >80%\n")
                    else:
                        f.write("\nConvergence Status: Poor - Loss reduction <80%\n")
            except Exception:
                pass


def create_mlp_specific_visualizations(model_results):
    """Create MLP training history and convergence analysis if available."""
    output_dir = model_results["output_dir"]
    model = model_results["model"]

    has_loss = hasattr(model, 'loss_curve_') and isinstance(model.loss_curve_, (list, tuple)) and len(model.loss_curve_) > 0
    has_val = hasattr(model, 'validation_scores_') and isinstance(model.validation_scores_, (list, tuple)) and len(model.validation_scores_) > 0

    if not has_loss and not has_val:
        return

    try:
        plt.figure(figsize=(12, 10))
        # Loss curve
        if has_loss:
            plt.subplot(2, 1, 1)
            plt.plot(model.loss_curve_, 'b-', linewidth=2, label='Training Loss')
            plt.title('MLP Training History')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)

        # Validation accuracy curve
        if has_val:
            plt.subplot(2, 1, 2 if has_loss else 1)
            plt.plot(model.validation_scores_, 'g-', linewidth=2, label='Validation Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history_extended.png'), dpi=300)
        plt.close()
    except Exception:
        # Ensure we don't break the pipeline if plotting fails
        pass

    # Convergence analysis
    try:
        with open(os.path.join(output_dir, 'convergence_analysis.txt'), 'w') as f:
            f.write("=== MLP CONVERGENCE ANALYSIS ===\n")
            if has_loss:
                final_loss = model.loss_curve_[-1]
                min_loss = min(model.loss_curve_)
                loss_reduction = 100.0 * (model.loss_curve_[0] - final_loss) / max(abs(model.loss_curve_[0]), 1e-12)
                f.write(f"Final training loss: {final_loss:.6f}\n")
                f.write(f"Minimum training loss: {min_loss:.6f}\n")
                f.write(f"Total loss reduction: {loss_reduction:.2f}%\n")
                f.write(f"Training epochs (n_iter_): {getattr(model, 'n_iter_', len(model.loss_curve_))}\n")
                f.write(f"Early stopping triggered: {getattr(model, 'n_iter_', 0) < getattr(model, 'max_iter', 0)}\n")

            if has_val:
                final_val_acc = model.validation_scores_[-1]
                max_val_acc = max(model.validation_scores_)
                acc_drop = 100.0 * (max_val_acc - final_val_acc)
                f.write("\nValidation Accuracy Analysis:\n")
                f.write(f"Maximum validation accuracy: {max_val_acc:.4f}\n")
                f.write(f"Final validation accuracy: {final_val_acc:.4f}\n")
                f.write(f"Accuracy change from peak: {acc_drop:.2f}%\n")

            # Determine convergence status
            if has_loss:
                if loss_reduction > 95:
                    status = "Excellent - Loss reduced by >95%"
                elif loss_reduction > 90:
                    status = "Good - Loss reduced by >90%"
                elif loss_reduction > 80:
                    status = "Fair - Loss reduced by >80%"
                else:
                    status = "Poor - Loss reduction <80%"
                f.write(f"\nConvergence Status: {status}\n")
    except Exception:
        pass