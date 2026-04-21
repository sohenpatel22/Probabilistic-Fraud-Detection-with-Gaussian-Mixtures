import os
import pandas as pd

from src.data.load_data import load_data
from src.models.gmm_model import fit_class_gmms, get_class_score_difference
from src.evaluation.thresholding import (
    find_best_threshold_high_score,
    predict_with_threshold_high_score
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


def save_final_results(results_df, filename):
    """
    Save final evaluation results as a CSV file.
    """
    os.makedirs(OUTPUT_TABLE_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_TABLE_DIR, filename)
    results_df.to_csv(save_path, index=False)
    print(f"Final results saved to: {save_path}")


def run_final_evaluation(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Fit the best two-model GMM, tune threshold on validation set,
    and evaluate final performance on the test set.
    """

    best_features = ["V11", "V12", "V14", "V16", "V17"]
    n_components_normal = 2
    n_components_fraud = 3

    # Fit best model on training set
    gmm_normal, gmm_fraud = fit_class_gmms(
        X_train[best_features],
        y_train,
        n_components_normal=n_components_normal,
        n_components_fraud=n_components_fraud
    )

    # Validation scores for threshold tuning
    val_scores = get_class_score_difference(gmm_normal, gmm_fraud, X_val[best_features])

    best_threshold, best_val_f1, thresholds, f1_scores = find_best_threshold_high_score(
        val_scores,
        y_val
    )

    # Test set evaluation
    test_scores = get_class_score_difference(gmm_normal, gmm_fraud, X_test[best_features])
    y_test_pred = predict_with_threshold_high_score(test_scores, best_threshold)

    test_metrics = get_all_metrics(y_test, test_scores, y_test_pred)

    # Save final plots
    plot_roc_curve(y_test, test_scores, filename="final_test_roc_curve.png")
    plot_pr_curve(y_test, test_scores, filename="final_test_pr_curve.png")
    plot_confusion_matrix(y_test, y_test_pred, filename="final_test_confusion_matrix.png")
    plot_threshold_vs_f1(thresholds, f1_scores, filename="final_validation_threshold_vs_f1.png")

    results_df = pd.DataFrame([{
        "features": best_features,
        "n_components_normal": n_components_normal,
        "n_components_fraud": n_components_fraud,
        "best_threshold": best_threshold,
        "validation_f1": best_val_f1,
        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"]
    }])

    save_final_results(results_df, "final_evaluation_results.csv")

    return results_df


if __name__ == "__main__":
    file_path = "data/creditcard.csv"

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(file_path)

    final_results = run_final_evaluation(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    print("\nFinal Evaluation Results:")
    print(final_results)