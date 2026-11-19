import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

from models.speech_pipeline.model import OriginalCNNTransformerSpeech


EMOTIONS = [
    "angry", "disgust", "fear", "happy",
    "neutral", "pleasant_surprise", "sad"
]

LABEL_TO_ID = {label: i for i, label in enumerate(EMOTIONS)}


class TESSAudioDataset(Dataset):
    def __init__(self, split_manifest, full_manifest, feature_dir):
        self.df = pd.read_csv(split_manifest)
        full_df = pd.read_csv(full_manifest)

        self.feature_dir = Path(feature_dir)
        self.feature_map = {
            filename: idx
            for idx, filename in enumerate(full_df["filename"])
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        feature_index = self.feature_map[row["filename"]]
        feature_file = self.feature_dir / f"{feature_index:04d}.npy"

        features = np.load(feature_file).astype(np.float32)
        features = torch.from_numpy(features.T.copy())

        label = LABEL_TO_ID[row["emotion"]]
        return features, torch.tensor(label, dtype=torch.long)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TESSAudioDataset(args.train_manifest, args.full_manifest, args.feature_dir)
    val_ds = TESSAudioDataset(args.val_manifest, args.full_manifest, args.feature_dir)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )

    model = OriginalCNNTransformerSpeech().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0005,
        weight_decay=0.0001
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    for epoch in range(args.epochs):
        model.train()
        train_losses, train_preds, train_labels = [], [], []

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(outputs.argmax(1).detach().cpu().tolist())
            train_labels.extend(labels.cpu().tolist())

        train_loss = float(np.mean(train_losses))
        train_acc = accuracy_score(train_labels, train_preds)

        model.eval()
        val_losses, val_preds, val_labels = [], [], []

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                val_losses.append(criterion(outputs, labels).item())
                val_preds.extend(outputs.argmax(1).cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_loss = float(np.mean(val_losses))
        val_acc = accuracy_score(val_labels, val_preds)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_weights)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        output_dir / "speech_cnn_transformer_best.pth"
    )

    with open(output_dir / "speech_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("Training completed.")
    print("Best validation loss:", best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--full-manifest", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", default="Results/checkpoints")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)

    train(parser.parse_args())
