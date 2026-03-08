import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import asdict

from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import load_data, split_data
from src.utils import make_run_dir
from src.metrics import plot_normalized_confusion_matrix, compute_metrics

from src.bilstm.data import build_vocab, collate_fn, TextDataset
from src.bilstm.model import BiLSTMConfig, BiLSTMClassifier



def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_pred, all_true = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        pred = logits.argmax(dim=1)

        # move pred and true labels to CPU and store
        all_pred.append(pred.cpu())
        all_true.append(y.cpu())

    y_pred = torch.cat(all_pred).numpy()
    y_true = torch.cat(all_true).numpy()
    return y_true, y_pred


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch and return the average loss.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        # zero gradients, forward pass, compute loss, backward pass, and update weights
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_samples += x.size(0)
        total_loss += loss.item() * x.size(0)

    return total_loss / total_samples


def main():
    cfg = BiLSTMConfig()

    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    print("Device: ", device)

    run_dir = make_run_dir(Path(cfg.output_root))
    print("Run_dir: ", run_dir)

    df_train, _ = load_data("data/train.csv", "data/test.csv")

    # split
    X_train, X_val, y_train, y_val = split_data(
        df_train, val_size=cfg.val_size, random_state=cfg.seed
    )

    # create vocab from training set only
    stoi, itos = build_vocab(X_train, min_freq=cfg.min_freq, max_size=cfg.max_vocab)

    train_ds = TextDataset(X_train, y_train, stoi, max_len=cfg.max_len)
    val_ds = TextDataset(X_val, y_val, stoi, max_len=cfg.max_len)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = BiLSTMClassifier(
        vocab_size=len(stoi),
        num_classes=cfg.num_classes,
        emb_dim=cfg.emb_dim,
        hidden_dim=cfg.hidden_dim,
        padding_idx=cfg.pad_idx,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = run_dir / "bilstm_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        y_true, y_pred = predict(model, val_loader, device)
        metrics = compute_metrics(y_true, y_pred)

        print(f"[Epoch{epoch}] loss={loss:.4f} acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]

            # save best model weights
            torch.save(model.state_dict(), best_path)

    # eval
    model.load_state_dict(torch.load(best_path, map_location=device))
    y_true, y_pred = predict(model, val_loader, device)

    final_metrics = compute_metrics(y_true, y_pred)
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

    print("Results:\n", final_metrics)
    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=cfg.labels))

    # save vocab
    joblib.dump(stoi, run_dir / "vocab.joblib")

    # save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # save metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # save report
    with open(run_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # save predictions
    pd.DataFrame(
        {
            "text": X_val,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).to_csv(run_dir / "preds.csv", index=False)
    

    print("Saved to: ", run_dir)


if __name__ == "__main__":
    main()