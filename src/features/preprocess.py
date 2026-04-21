import pandas as pd
from sklearn.preprocessing import StandardScaler


def check_missing_values(df):
    """
    Check if dataset contains missing values.
    """

    missing = df.isnull().sum().sum()

    if missing > 0:
        print(f"Dataset contains {missing} missing values.")
    else:
        print("No missing values found.")

    return missing


def scale_features(X_train, X_val, X_test):
    """
    Standardize features using training data statistics.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled