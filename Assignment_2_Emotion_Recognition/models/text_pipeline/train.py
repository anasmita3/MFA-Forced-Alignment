import os
import re
import json
import copy
import argparse
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

from models.text_pipeline.model import (
    TextBiLSTM,
    TransformerTextModel,
    UpdatedTextTransformer
)


EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant_surprise",
    "sad"
]

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
MAX_LENGTH = 6


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(text):
    return re.findall(r"[a-z]+", str(text).lower())


def build_original_vocab(train_df):
    word_counts = Counter()

    for text in train_df["transcript"]:
        word_counts.update(tokenize(text))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    for word in sorted(word_counts):
        vocab[word] = len(vocab)

    return vocab


def build_secondary_vocab(train_df, val_df, test_df):
    all_transcripts = pd.concat([
        train_df["transcript"],
        val_df["transcript"],
        test_df["transcript"]
    ]).astype(str)

    word_counts = Counter()

    for transcript in all_transcripts:
        word_counts.update(transcript.lower().split())

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    for word in sorted(word_counts):
        if word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def encode_text(text, vocab):
    tokens = tokenize(text)

    ids = [
        vocab.get(token, vocab[UNK_TOKEN])
        for token in tokens
    ]

    ids = ids[:MAX_LENGTH]
    ids += [vocab[PAD_TOKEN]] * (MAX_LENGTH - len(ids))

    return ids


def encode_secondary(text, vocab):
    tokens = str(text).lower().split()

    ids = [
        vocab.get(token, vocab[UNK_TOKEN])
        for token in tokens
    ]

    ids = ids[:MAX_LENGTH]
    ids += [vocab[PAD_TOKEN]] * (MAX_LENGTH - len(ids))

    return ids
class TESSTextDataset(Dataset):
    def __init__(self, dataframe, vocab, encoding="original"):
        self.tokens = []

        for transcript in dataframe["transcript"]:
            if encoding == "original":
                ids = encode_text(transcript, vocab)
            else:
                ids = encode_secondary(transcript, vocab)

            self.tokens.append(ids)

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)

        self.labels = torch.tensor(
            dataframe["emotion"].map(
                {emotion: i for i, emotion in enumerate(EMOTIONS)}
            ).tolist(),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.tokens[index], self.labels[index]


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for tokens, labels in loader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            outputs = model(tokens)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predictions.extend(outputs.argmax(1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return (
        total_loss / len(true_labels),
        accuracy_score(true_labels, predictions),
        predictions,
        true_labels
    )


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    output_path,
    epochs,
    patience,
    scheduler=None,
    min_delta=0.0
):
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        train_predictions = []
        train_labels = []

        for tokens, labels in train_loader:
            tokens = tokens.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(tokens)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            train_predictions.extend(outputs.argmax(1).detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        train_loss = total_loss / len(train_labels)
        train_acc = accuracy_score(train_labels, train_predictions)

        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    model.load_state_dict(best_weights)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)

    return model, history, best_val_loss
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-manifest", default="Results/train_manifest.csv")
    parser.add_argument("--val-manifest", default="Results/validation_manifest.csv")
    parser.add_argument("--test-manifest", default="Results/test_manifest.csv")
    parser.add_argument("--output-dir", default="Results/text_models")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_df = pd.read_csv(args.train_manifest)
    val_df = pd.read_csv(args.val_manifest)
    test_df = pd.read_csv(args.test_manifest)

    os.makedirs(args.output_dir, exist_ok=True)

    label_to_id = {
        emotion: i for i, emotion in enumerate(EMOTIONS)
    }

    # Build vocabularies
    original_vocab = build_original_vocab(train_df)

    secondary_vocab = build_secondary_vocab(
        train_df, val_df, test_df
    )

    print("Original vocabulary:", len(original_vocab))
    print("Updated vocabulary:", len(secondary_vocab))

    # Save vocabulary and label mappings
    with open(os.path.join(args.output_dir, "vocabularies.json"), "w") as f:
        json.dump({
            "original_vocab": original_vocab,
            "secondary_vocab": secondary_vocab,
            "label_to_id": label_to_id,
            "emotions": EMOTIONS,
            "max_length": MAX_LENGTH
        }, f, indent=4)

    # Prepare datasets
    original_train = TESSTextDataset(train_df, original_vocab, "original")
    original_val = TESSTextDataset(val_df, original_vocab, "original")
    original_test = TESSTextDataset(test_df, original_vocab, "original")

    updated_train = TESSTextDataset(train_df, secondary_vocab, "updated")
    updated_val = TESSTextDataset(val_df, secondary_vocab, "updated")
    updated_test = TESSTextDataset(test_df, secondary_vocab, "updated")

    def make_loaders(train_ds, val_ds, test_ds):
        return (
            DataLoader(train_ds, batch_size=args.batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=args.batch_size),
            DataLoader(test_ds, batch_size=args.batch_size)
        )

    original_train_loader, original_val_loader, original_test_loader = (
        make_loaders(original_train, original_val, original_test)
    )

    updated_train_loader, updated_val_loader, updated_test_loader = (
        make_loaders(updated_train, updated_val, updated_test)
    )

    results = {}

    # ------------------------------------------------
    # 1. Original Text BiLSTM
    # ------------------------------------------------
    print("\nTraining Original Text BiLSTM")

    bilstm = TextBiLSTM(
        vocab_size=len(original_vocab)
    ).to(device)

    bilstm_optimizer = torch.optim.Adam(
        bilstm.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )

    bilstm, bilstm_history, bilstm_best_loss = train_model(
        bilstm,
        original_train_loader,
        original_val_loader,
        bilstm_optimizer,
        device,
        os.path.join(args.output_dir, "text_bilstm_best.pth"),
        epochs=30,
        patience=5
    )

    # ------------------------------------------------
    # 2. Original Text Transformer
    # ------------------------------------------------
    print("\nTraining Original Text Transformer")

    transformer = TransformerTextModel(
        vocab_size=len(original_vocab),
        num_classes=len(EMOTIONS),
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
        pad_idx=0
    ).to(device)

    transformer_optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=0.0005,
        weight_decay=0.0001
    )

    transformer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        transformer_optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    transformer, transformer_history, transformer_best_loss = train_model(
        transformer,
        original_train_loader,
        original_val_loader,
        transformer_optimizer,
        device,
        os.path.join(args.output_dir, "transformer_text_best.pth"),
        epochs=35,
        patience=7,
        scheduler=transformer_scheduler,
        min_delta=1e-5
    )

    # ------------------------------------------------
    # 3. Updated Text Transformer
    # ------------------------------------------------
    print("\nTraining Updated Text Transformer")

    updated_transformer = UpdatedTextTransformer(
        vocab_size=len(secondary_vocab),
        num_classes=len(EMOTIONS)
    ).to(device)

    updated_optimizer = torch.optim.AdamW(
        updated_transformer.parameters(),
        lr=0.0005,
        weight_decay=0.0001
    )

    updated_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        updated_optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    updated_transformer, updated_history, updated_best_loss = train_model(
        updated_transformer,
        updated_train_loader,
        updated_val_loader,
        updated_optimizer,
        device,
        os.path.join(args.output_dir, "updated_text_transformer_best.pth"),
        epochs=35,
        patience=7,
        scheduler=updated_scheduler
    )

    # ------------------------------------------------
    # Evaluate all three models
    # ------------------------------------------------
    criterion = nn.CrossEntropyLoss()

    for name, model, loader in [
        ("Text BiLSTM", bilstm, original_test_loader),
        ("Original Transformer", transformer, original_test_loader),
        ("Updated Transformer", updated_transformer, updated_test_loader)
    ]:
        loss, accuracy, predictions, labels = evaluate(
            model, loader, criterion, device
        )

        results[name] = {
            "test_loss": loss,
            "test_accuracy": accuracy,
            "predictions": predictions,
            "labels": labels
        }

        print(f"\n{name} Test Accuracy: {accuracy * 100:.2f}%")

    # Save histories and evaluation results
    histories = {
        "text_bilstm": bilstm_history,
        "original_transformer": transformer_history,
        "updated_transformer": updated_history
    }

    with open(os.path.join(args.output_dir, "training_histories.json"), "w") as f:
        json.dump(histories, f, indent=4)

    summary = {}

    for name, result in results.items():
        summary[name] = {
            "test_loss": result["test_loss"],
            "test_accuracy": result["test_accuracy"]
        }

    with open(os.path.join(args.output_dir, "text_results.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print("\nText pipeline completed.")
    print("Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
