import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path, selected_features=None, target_col="Class", random_state=0):
    """
    Load the dataset, keep selected columns, and create train/val/test splits.
    """

    # Read csv file
    df = pd.read_csv(file_path)

    # Default features used in this project
    if selected_features is None:
        selected_features = [f"V{i}" for i in range(11, 21)]

    # Keep only selected features + target
    required_cols = selected_features + [target_col]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataset.")

    df = df[required_cols].copy()

    # Split into X and y
    X = df.drop(columns=target_col)
    y = df[target_col]

    # First split: train 70%, temp 30%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y
    )

    # Second split: validation 15%, test 15%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    file_path = "data/creditcard.csv"

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(file_path)

    print("Train shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape:", X_test.shape)
    print("\nTraining class distribution:")
    print(y_train.value_counts(normalize=True))