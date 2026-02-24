import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Combine "Title" and "Description" into a single "text" column for both train and test datasets
    train_df["text"] = train_df["Title"].astype(str) + " " + train_df["Description"].astype(str)
    test_df["text"] = test_df["Title"].astype(str) + " " + test_df["Description"].astype(str)

    return train_df, test_df

def split_data(df, val_size=0.1, random_state=42):
    X = df["text"]
    
    # labels are 1-indexed in the dataset, so we convert them to 0-indexed by subtracting 1
    y = df["Class Index"].astype(int) - 1

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=random_state,
        stratify=y
    )

    return (
        X_train.to_list(),
        X_val.to_list(),
        y_train.to_list(),
        y_val.to_list(),
    )
