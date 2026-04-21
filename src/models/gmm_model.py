import numpy as np
from sklearn.mixture import GaussianMixture


def fit_gmm(X, n_components=1, random_state=42):
    """
    Fit a Gaussian Mixture Model on the given data.
    """

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(X)

    return gmm


def get_gmm_scores(gmm, X):
    """
    Get log-likelihood scores from a fitted GMM.

    Lower score means the sample is less likely under the model.
    """
    return gmm.score_samples(X)


def fit_class_gmms(X_train, y_train, n_components_normal=1, n_components_fraud=1, random_state=42):
    """
    Fit two separate GMMs:
    - one on normal transactions
    - one on fraud transactions
    """

    X_train_array = np.array(X_train)
    y_train_array = np.array(y_train)

    X_normal = X_train_array[y_train_array == 0]
    X_fraud = X_train_array[y_train_array == 1]

    gmm_normal = GaussianMixture(
        n_components=n_components_normal,
        random_state=random_state
    )
    gmm_normal.fit(X_normal)

    gmm_fraud = GaussianMixture(
        n_components=n_components_fraud,
        random_state=random_state
    )
    gmm_fraud.fit(X_fraud)

    return gmm_normal, gmm_fraud


def get_class_score_difference(gmm_normal, gmm_fraud, X):
    """
    Compute score difference between fraud and normal models.

    Higher values mean the sample looks more like fraud compared to normal.
    """

    X_array = np.array(X)

    normal_scores = gmm_normal.score_samples(X_array)
    fraud_scores = gmm_fraud.score_samples(X_array)

    score_diff = fraud_scores - normal_scores

    return score_diff


if __name__ == "__main__":
    from src.data.load_data import load_data

    file_path = "data/creditcard.csv"

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(file_path)

    # Example 1: single GMM on one feature
    feature = ["V14"]

    gmm = fit_gmm(X_train[feature], n_components=1)
    train_scores = get_gmm_scores(gmm, X_train[feature])

    print("Single GMM example")
    print("First 5 scores:", train_scores[:5])

    # Example 2: separate normal/fraud models
    features = ["V11", "V12", "V14", "V16", "V17"]

    gmm_normal, gmm_fraud = fit_class_gmms(
        X_train[features],
        y_train,
        n_components_normal=2,
        n_components_fraud=3
    )

    val_score_diff = get_class_score_difference(gmm_normal, gmm_fraud, X_val[features])

    print("\nTwo-model GMM example")
    print("First 5 score differences:", val_score_diff[:5])