from data import load_data, split_data
from baseline import train_baseline


def main():
    train_df, test_df = load_data("data/train.csv", "data/test.csv")
    X_train, X_val, y_train, y_val = split_data(train_df, val_size=0.1, random_state=42)

    model, vectorizer, acc, report, cm = train_baseline(X_train, y_train, X_val, y_val)

    print("===== Baseline Result =====")
    print("Accuracy:", acc)
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)


if __name__ == "__main__":
    main()
