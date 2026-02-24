import json
import joblib
import pandas as pd
from pathlib import Path
from dataclasses import asdict

from sklearn.metrics import classification_report

from src.data import load_data, split_data
from src.utils import make_run_dir
from src.metrics import compute_metrics, plot_normalized_confusion_matrix
from src.tfidf.model import build_pipeline, TfidfConfig



def main():
    # create config
    cfg = TfidfConfig()

    # create run dir
    run_dir = make_run_dir(Path(cfg.output_root))
    print(f"Run dir: {run_dir}")

    # load data
    df_train, df_test = load_data("data/train.csv", "data/test.csv")

    # split
    X_train, X_val, y_train, y_val = split_data(df_train, val_size=cfg.val_size, random_state=cfg.seed)

    # build model
    pipe = build_pipeline(cfg)

    # train
    print("Training...")
    pipe.fit(X_train, y_train)

    # eval
    print("Evaluating...")
    y_pred = pipe.predict(X_val)
    metrics = compute_metrics(y_val, y_pred)
    report = classification_report(
        y_val, y_pred, 
        target_names=cfg.labels, 
        output_dict=True
        )
    plot_normalized_confusion_matrix(
        y_val, y_pred, 
        labels=cfg.labels, 
        save_path=run_dir / "normalized_confusion_matrix.png"
        )

    print("Results:\n", metrics)
    print("\nClassification report:\n")
    print(classification_report(y_val, y_pred, target_names=cfg.labels))


    # save pipeline（vectorizer + classifier）
    joblib.dump(pipe, run_dir / "pipeline.joblib")

    # save report
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # save metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # save predictions
    pd.DataFrame(
        {
            "text": X_val,
            "y_true": y_val,
            "y_pred": y_pred,
        }
    ).to_csv(run_dir / "preds.csv", index=False)

    print("Saved to: ", run_dir)


if __name__ == "__main__":
    main()