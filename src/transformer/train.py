import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict

import torch
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score, classification_report, f1_score

from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
)

from src.data import load_data, split_data
from src.utils import make_run_dir
from src.metrics import plot_normalized_confusion_matrix
from src.transformer.model import TransformerConfig, build_model


class HFTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = int(self.labels[idx])
        return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "weighted_f1": float(f1_score(labels, preds, average="weighted")),
    }


def main():
    cfg = TransformerConfig()

    set_seed(cfg.seed)

    run_dir = make_run_dir(Path(cfg.output_root))
    print("Run_dir: ", run_dir)

    # load & split
    df_train, df_test = load_data("data/train.csv", "data/test.csv")
    X_train, X_val, y_train, y_val = split_data(df_train, val_size=cfg.val_size, random_state=cfg.seed)

    # model / tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = build_model(cfg)

    # datasets
    train_ds = HFTextDataset(X_train, y_train, tokenizer, max_length=cfg.max_length)
    val_ds = HFTextDataset(X_val, y_val, tokenizer, max_length=cfg.max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # training args
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,

        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,

        report_to="none",
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # train
    trainer.train()

    # save best model
    best_dir = run_dir / "best_model"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    pred_out = trainer.predict(val_ds)
    metrics = pred_out.metrics
    logits = pred_out.predictions

    y_pred = np.argmax(logits, axis=1)
    y_true = np.array(y_val, dtype=int)

    report = classification_report(
        y_true, y_pred,
        target_names=cfg.labels,
        output_dict=True,
    )

    plot_normalized_confusion_matrix(
        y_true, y_pred,
        labels=cfg.labels,
        save_path=run_dir / "normalized_confusion_matrix.png",
    )

    print("Results:\n", metrics)
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=cfg.labels))


    # save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # save metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # save report
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # save preds.csv
    pd.DataFrame({
        "text": X_val,
        "y_true": y_true,
        "y_pred": y_pred,
    }).to_csv(run_dir / "preds.csv", index=False)


    print("Saved to: ", run_dir)


if __name__ == "__main__":
    main()