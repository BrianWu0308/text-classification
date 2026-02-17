import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df["text"] = train_df["Title"].astype(str) + " " + train_df["Description"].astype(str)
    test_df["text"] = test_df["Title"].astype(str) + " " + test_df["Description"].astype(str)

    return train_df, test_df

def split_data(df, val_size=0.1, random_state=42):
    X = df["text"]
    y = df["Class Index"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_val, y_train, y_val
