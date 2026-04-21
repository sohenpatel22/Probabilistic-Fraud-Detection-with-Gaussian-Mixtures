from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)


def compute_roc_auc(y_true, scores):
    """
    Compute ROC-AUC using anomaly scores.

    Higher score should mean more likely fraud.
    """
    return roc_auc_score(y_true, scores)


def compute_pr_auc(y_true, scores):
    """
    Compute PR-AUC using anomaly scores.

    Higher score should mean more likely fraud.
    """
    return average_precision_score(y_true, scores)


def compute_f1(y_true, y_pred):
    """
    Compute F1 score.
    """
    return f1_score(y_true, y_pred)


def compute_precision(y_true, y_pred):
    """
    Compute precision score.
    """
    return precision_score(y_true, y_pred)


def compute_recall(y_true, y_pred):
    """
    Compute recall score.
    """
    return recall_score(y_true, y_pred)


def get_classification_metrics(y_true, y_pred):
    """
    Return precision, recall, and F1 score in a dictionary.
    """
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }


def get_all_metrics(y_true, scores, y_pred):
    """
    Return ROC-AUC, PR-AUC, precision, recall, and F1 in one dictionary.
    """
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }


if __name__ == "__main__":
    y_true = [0, 0, 0, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.8, 0.9]
    y_pred = [0, 0, 0, 1, 1]

    print("ROC-AUC:", compute_roc_auc(y_true, scores))
    print("PR-AUC:", compute_pr_auc(y_true, scores))
    print("F1:", compute_f1(y_true, y_pred))
    print("Precision:", compute_precision(y_true, y_pred))
    print("Recall:", compute_recall(y_true, y_pred))

    print("\nAll metrics:")
    print(get_all_metrics(y_true, scores, y_pred))
    