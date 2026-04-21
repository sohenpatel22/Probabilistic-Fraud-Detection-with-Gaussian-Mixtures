import numpy as np


def find_best_threshold_low_score(scores, y_true):
    """
    Find the threshold that maximizes F1 score.

    Lower score = more likely fraud
    """

    scores = np.array(scores)
    y_true = np.array(y_true)

    # Sort scores ascending
    sorted_idx = np.argsort(scores)
    scores_sorted = scores[sorted_idx]
    y_sorted = y_true[sorted_idx]

    total_positives = np.sum(y_sorted)

    TP = 0
    FP = 0
    FN = total_positives

    best_f1 = 0
    best_threshold = None
    thresholds = []
    f1_scores = []

    for i in range(len(scores_sorted)):

        if y_sorted[i] == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1

        denom = 2 * TP + FP + FN
        f1 = (2 * TP) / denom if denom > 0 else 0

        thresholds.append(scores_sorted[i])
        f1_scores.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = scores_sorted[i]

    return best_threshold, best_f1, thresholds, f1_scores


def predict_with_threshold_low_score(scores, threshold):
    """
    Convert scores into predictions.

    score < threshold -> fraud (1)
    score >= threshold -> normal (0)
    """

    scores = np.array(scores)
    y_pred = (scores < threshold).astype(int)

    return y_pred


def find_best_threshold_high_score(scores, y_true):
    """
    Find the threshold that maximizes F1 score.

    Higher score = more likely fraud
    """

    scores = np.array(scores)
    y_true = np.array(y_true)

    # Sort scores descending
    sorted_idx = np.argsort(-scores)
    scores_sorted = scores[sorted_idx]
    y_sorted = y_true[sorted_idx]

    total_positives = np.sum(y_sorted)

    TP = 0
    FP = 0
    FN = total_positives

    best_f1 = 0
    best_threshold = None
    thresholds = []
    f1_scores = []

    for i in range(len(scores_sorted)):

        if y_sorted[i] == 1:
            TP += 1
            FN -= 1
        else:
            FP += 1

        denom = 2 * TP + FP + FN
        f1 = (2 * TP) / denom if denom > 0 else 0

        thresholds.append(scores_sorted[i])
        f1_scores.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = scores_sorted[i]

    return best_threshold, best_f1, thresholds, f1_scores


def predict_with_threshold_high_score(scores, threshold):
    """
    Convert scores into predictions.

    score > threshold -> fraud (1)
    score <= threshold -> normal (0)
    """

    scores = np.array(scores)
    y_pred = (scores > threshold).astype(int)

    return y_pred


if __name__ == "__main__":
    # Example 1: lower score means more likely fraud
    scores_low = np.array([-10, -8, -6, -3, -1, 0])
    y_true = np.array([1, 1, 0, 0, 0, 0])

    threshold_low, best_f1_low, _, _ = find_best_threshold_low_score(scores_low, y_true)
    y_pred_low = predict_with_threshold_low_score(scores_low, threshold_low)

    print("Low-score case")
    print("Best threshold:", threshold_low)
    print("Best F1:", best_f1_low)
    print("Predictions:", y_pred_low)

    # Example 2: higher score means more likely fraud
    scores_high = np.array([10, 8, 6, 3, 1, 0])
    threshold_high, best_f1_high, _, _ = find_best_threshold_high_score(scores_high, y_true)
    y_pred_high = predict_with_threshold_high_score(scores_high, threshold_high)

    print("\nHigh-score case")
    print("Best threshold:", threshold_high)
    print("Best F1:", best_f1_high)
    print("Predictions:", y_pred_high)