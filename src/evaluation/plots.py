import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


OUTPUT_DIR = "outputs/figures"


def _save_plot(filename):
    """
    Save plot to outputs/figures directory.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    print(f"Plot saved to: {path}")


def plot_feature_distributions(X, y, filename="feature_distributions.png"):
    """
    Plot feature distributions for normal vs fraud.
    """

    features = X.columns
    plt.figure(figsize=(15, 4 * len(features)))

    for i, feature in enumerate(features, 1):
        plt.subplot(len(features), 1, i)

        sns.histplot(X.loc[y == 0, feature], bins=50, stat="density", kde=True, color="blue")
        sns.histplot(X.loc[y == 1, feature], bins=50, stat="density", kde=True, color="red")

        plt.title(f"Distribution of {feature}")
        plt.legend(["Normal", "Fraud"])

    plt.tight_layout()
    _save_plot(filename)
    plt.show()
    plt.close()


def plot_roc_curve(y_true, scores, filename="roc_curve.png"):
    """
    Plot ROC curve.
    """

    fpr, tpr, _ = roc_curve(y_true, scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    _save_plot(filename)
    plt.show()
    plt.close()


def plot_pr_curve(y_true, scores, filename="pr_curve.png"):
    """
    Plot Precision-Recall curve.
    """

    precision, recall, _ = precision_recall_curve(y_true, scores)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="PR Curve")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()

    _save_plot(filename)
    plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    """
    Plot confusion matrix.
    """

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Fraud"],
        yticklabels=["Normal", "Fraud"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    _save_plot(filename)
    plt.show()
    plt.close()


def plot_threshold_vs_f1(thresholds, f1_scores, filename="threshold_vs_f1.png"):
    """
    Plot threshold vs F1 score.
    """

    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, f1_scores)

    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("Threshold vs F1 Score")

    _save_plot(filename)
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("Plot module ready.")