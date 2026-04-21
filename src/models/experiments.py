import os
import numpy as np
import pandas as pd

from src.data.load_data import load_data
from src.models.gmm_model import (
    fit_gmm,
    get_gmm_scores,
    fit_class_gmms,
    get_class_score_difference
)
from src.evaluation.thresholding import (
    predict_with_threshold_high_score, 
    predict_with_threshold_low_score, 
    find_best_threshold_high_score, 
    find_best_threshold_low_score
)
from src.evaluation.metrics import get_all_metrics
from src.evaluation.plots import (
    plot_roc_curve,
    plot_pr_curve,
    plot_confusion_matrix,
    plot_threshold_vs_f1
)
from src.utils.seed import set_seed

set_seed(42)

OUTPUT_TABLE_DIR = "outputs/tables"


def save_results_table(results_df, filename):
    """
    Save experiment results as a CSV file.
    """
    os.makedirs(OUTPUT_TABLE_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_TABLE_DIR, filename)
    results_df.to_csv(save_path, index=False)
    print(f"Results saved to: {save_path}")


def run_single_feature_gmm_experiments(X_train, X_val, y_train, y_val):
    """
    Run single-feature GMM experiments for all features.
    Fit one Gaussian per feature and evaluate on validation set.
    """

    results = []

    for feature in X_train.columns:
        # Fit one Gaussian on training data for this feature
        gmm = fit_gmm(X_train[[feature]], n_components=1)

        # Raw GMM log-likelihood scores
        val_scores_raw = get_gmm_scores(gmm, X_val[[feature]])

        # Convert to anomaly scores so higher = more likely fraud
        val_scores = -val_scores_raw

        # Threshold search is done on raw scores because lower raw score = fraud
        best_threshold, best_f1, _, _ = find_best_threshold_low_score(val_scores_raw, y_val)
        y_val_pred = predict_with_threshold_low_score(val_scores_raw, best_threshold)

        metrics = get_all_metrics(y_val, val_scores, y_val_pred)

        results.append({
            "feature": feature,
            "best_threshold": best_threshold,
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1", ascending=False).reset_index(drop=True)

    save_results_table(results_df, "single_feature_gmm_results.csv")

    return results_df


def run_multifeature_gmm_experiment(X_train, X_val, y_train, y_val, features, n_components=1):
    """
    Run a multifeature GMM experiment using one model.
    """

    gmm = fit_gmm(X_train[features], n_components=n_components)

    val_scores_raw = get_gmm_scores(gmm, X_val[features])
    val_scores = -val_scores_raw

    best_threshold, best_f1, thresholds, f1_scores = find_best_threshold_low_score(val_scores_raw, y_val)
    y_val_pred = predict_with_threshold_low_score(val_scores_raw, best_threshold)

    metrics = get_all_metrics(y_val, val_scores, y_val_pred)

    # Save plots
    plot_roc_curve(y_val, val_scores, filename="multifeature_roc_curve.png")
    plot_pr_curve(y_val, val_scores, filename="multifeature_pr_curve.png")
    plot_confusion_matrix(y_val, y_val_pred, filename="multifeature_confusion_matrix.png")
    plot_threshold_vs_f1(
    thresholds,
    f1_scores,
    filename="multifeature_threshold_vs_f1.png"
)

    result_df = pd.DataFrame([{
        "features": features,
        "n_components": n_components,
        "best_threshold": best_threshold,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"]
        }])

    save_results_table(result_df, "multifeature_gmm_result.csv")

    return result_df


def run_two_model_gmm_experiment(
    X_train,
    X_val,
    y_train,
    y_val,
    features,
    n_components_normal=1,
    n_components_fraud=1
):
    """
    Run a two-model GMM experiment:
    one model for normal transactions and one model for fraud transactions.
    """

    gmm_normal, gmm_fraud = fit_class_gmms(
        X_train[features],
        y_train,
        n_components_normal=n_components_normal,
        n_components_fraud=n_components_fraud
    )

    # Higher difference means more fraud-like
    val_scores = get_class_score_difference(gmm_normal, gmm_fraud, X_val[features])

    best_threshold, best_f1, thresholds, f1_scores = find_best_threshold_high_score(val_scores, y_val)
    best_y_pred = predict_with_threshold_high_score(val_scores, best_threshold)

    metrics = get_all_metrics(y_val, val_scores, best_y_pred)

    # Save plots
    plot_roc_curve(y_val, val_scores, filename="two_model_roc_curve.png")
    plot_pr_curve(y_val, val_scores, filename="two_model_pr_curve.png")
    plot_confusion_matrix(y_val, best_y_pred, filename="two_model_confusion_matrix.png")
    plot_threshold_vs_f1(thresholds, f1_scores, filename="two_model_threshold_vs_f1.png")

    result_df = pd.DataFrame([{
        "features": features,
        "n_components_normal": n_components_normal,
        "n_components_fraud": n_components_fraud,
        "best_threshold": best_threshold,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"]
        }])

    save_results_table(result_df, "two_model_gmm_result.csv")

    return result_df


if __name__ == "__main__":
    file_path = "data/creditcard.csv"

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(file_path)

    print("Running single-feature experiments...")
    single_feature_results = run_single_feature_gmm_experiments(
        X_train, X_val, y_train, y_val
    )
    print(single_feature_results.head())

    print("\nRunning multifeature one-model experiment...")
    multi_result = run_multifeature_gmm_experiment(
        X_train,
        X_val,
        y_train,
        y_val,
        features=["V14", "V17"],
        n_components=1
    )
    print(multi_result)

    print("\nRunning two-model GMM experiment...")
    two_model_result = run_two_model_gmm_experiment(
        X_train,
        X_val,
        y_train,
        y_val,
        features=["V11", "V12", "V14", "V16", "V17"],
        n_components_normal=2,
        n_components_fraud=3
    )
    print(two_model_result)