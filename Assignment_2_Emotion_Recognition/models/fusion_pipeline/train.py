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
from torch.utils.data import Dataset, DataLoader

from models.speech_pipeline.model import OriginalCNNTransformerSpeech
from models.text_pipeline.model import TransformerTextModel
from models.fusion_pipeline.model import CrossModalFusion


EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant_surprise",
    "sad"
]

BATCH_SIZE = 32
FUSION_EPOCHS = 35
FUSION_PATIENCE = 7
MAX_LENGTH = 6


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpeechFeatureExtractor(nn.Module):
    """Extract intermediate temporal features from the trained speech model."""

    def __init__(self, trained_model):
        super().__init__()

        self.cnn = trained_model.cnn
        self.projection = trained_model.projection
        self.position = trained_model.position
        self.transformer = trained_model.transformer

    def forward(self, x):
        # Input: (B, 300, 64)
        x = x.transpose(1, 2).unsqueeze(1)

        x = self.cnn(x)

        batch, channels, frequency, time = x.shape

        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch, time, channels * frequency)

        x = self.projection(x)
        x = self.position(x)
        x = self.transformer(x)

        return x


class TextFeatureExtractor(nn.Module):
    """Extract contextual token features from the trained text model."""

    def __init__(self, trained_model):
        super().__init__()

        self.pad_idx = trained_model.pad_idx
        self.embedding = trained_model.embedding
        self.position = trained_model.position
        self.transformer = trained_model.transformer

    def forward(self, tokens):
        padding_mask = tokens == self.pad_idx

        x = self.embedding(tokens)
        x = self.position(x)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        return x, padding_mask


class TESSMultimodalDataset(Dataset):
    """Load paired audio features, transcripts and emotion labels."""

    def __init__(
        self,
        split_manifest,
        full_manifest,
        feature_dir,
        vocab
    ):
        self.df = pd.read_csv(split_manifest)
        full_df = pd.read_csv(full_manifest)

        self.feature_dir = Path(feature_dir)
        self.vocab = vocab

        self.feature_map = {
            filename: idx
            for idx, filename in enumerate(full_df["filename"])
        }

        missing = set(self.df["filename"]) - set(self.feature_map)

        if missing:
            raise ValueError(
                f"{len(missing)} split files are missing from full manifest."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["filename"]

        feature_idx = self.feature_map[filename]
        feature_path = self.feature_dir / f"{feature_idx:04d}.npy"

        # Original feature shape: (64, 300)
        # Speech encoder input shape: (300, 64)
        audio_features = np.load(feature_path)

        audio_features = torch.tensor(
            audio_features.T.copy(),
            dtype=torch.float32
        )

        words = str(row["transcript"]).lower().split()

        token_ids = [
            self.vocab.get(word, self.vocab["<UNK>"])
            for word in words
        ]

        token_ids = token_ids[:MAX_LENGTH]

        token_ids += [self.vocab["<PAD>"]] * (
            MAX_LENGTH - len(token_ids)
        )

        text_tokens = torch.tensor(
            token_ids,
            dtype=torch.long
        )

        label = EMOTIONS.index(row["emotion"])

        return (
            audio_features,
            text_tokens,
            torch.tensor(label, dtype=torch.long)
        )


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state)
    return model


def train(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # Load the original vocabulary
    # --------------------------------------------------

    with open(args.vocab_file, "r", encoding="utf-8") as f:
        vocabularies = json.load(f)

    vocab = vocabularies["original_vocab"]

    print("Original vocabulary size:", len(vocab))

    # --------------------------------------------------
    # Load pretrained speech and text models
    # --------------------------------------------------

    speech_model = OriginalCNNTransformerSpeech().to(device)

    speech_model = load_checkpoint(
        speech_model,
        args.speech_checkpoint,
        device
    )

    text_model = TransformerTextModel(
        vocab_size=len(vocab),
        num_classes=len(EMOTIONS),
        d_model=128,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
        pad_idx=0
    ).to(device)

    text_model = load_checkpoint(
        text_model,
        args.text_checkpoint,
        device
    )

    # --------------------------------------------------
    # Create intermediate feature extractors
    # --------------------------------------------------

    speech_encoder = SpeechFeatureExtractor(
        speech_model
    ).to(device)

    text_encoder = TextFeatureExtractor(
        text_model
    ).to(device)

    # Freeze both pretrained encoders
    for parameter in speech_encoder.parameters():
        parameter.requires_grad = False

    for parameter in text_encoder.parameters():
        parameter.requires_grad = False

    speech_encoder.eval()
    text_encoder.eval()

    print("Speech encoder restored and frozen.")
    print("Text encoder restored and frozen.")

    # --------------------------------------------------
    # Create multimodal datasets
    # --------------------------------------------------

    train_dataset = TESSMultimodalDataset(
        args.train_manifest,
        args.full_manifest,
        args.feature_dir,
        vocab
    )

    val_dataset = TESSMultimodalDataset(
        args.val_manifest,
        args.full_manifest,
        args.feature_dir,
        vocab
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------
    # Create fusion model
    # --------------------------------------------------

    fusion_model = CrossModalFusion(
        d_model=128,
        num_heads=4,
        num_classes=len(EMOTIONS),
        dropout=0.3
    ).to(device)

    print(
        "Trainable fusion parameters:",
        sum(
            p.numel()
            for p in fusion_model.parameters()
            if p.requires_grad
        )
    )

    fusion_criterion = nn.CrossEntropyLoss()

    fusion_optimizer = optim.AdamW(
        fusion_model.parameters(),
        lr=0.0005,
        weight_decay=0.0001
    )

    fusion_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        fusion_optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    # --------------------------------------------------
    # Training configuration
    # --------------------------------------------------

    best_fusion_val_loss = float("inf")
    best_fusion_weights = None
    epochs_without_improvement = 0

    fusion_history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    # --------------------------------------------------
    # Fusion training loop
    # --------------------------------------------------

    for epoch in range(FUSION_EPOCHS):

        fusion_model.train()
        speech_encoder.eval()
        text_encoder.eval()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for audio, text, labels in train_loader:

            audio = audio.to(device)
            text = text.to(device)
            labels = labels.to(device)

            fusion_optimizer.zero_grad()

            # Extract frozen encoder representations
            with torch.no_grad():
                audio_features = speech_encoder(audio)

                text_features, text_padding_mask = text_encoder(text)

            # Cross-modal fusion
            outputs = fusion_model(
                audio_features,
                text_features,
                text_padding_mask
            )

            loss = fusion_criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                fusion_model.parameters(),
                max_norm=1.0
            )

            fusion_optimizer.step()

            train_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            train_correct += (
                predictions == labels
            ).sum().item()

            train_total += labels.size(0)

        train_loss /= train_total
        train_accuracy = train_correct / train_total

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        fusion_model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for audio, text, labels in val_loader:

                audio = audio.to(device)
                text = text.to(device)
                labels = labels.to(device)

                audio_features = speech_encoder(audio)

                text_features, text_padding_mask = text_encoder(text)

                outputs = fusion_model(
                    audio_features,
                    text_features,
                    text_padding_mask
                )

                loss = fusion_criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)

                predictions = outputs.argmax(dim=1)

                val_correct += (
                    predictions == labels
                ).sum().item()

                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        fusion_history["train_loss"].append(train_loss)
        fusion_history["val_loss"].append(val_loss)
        fusion_history["train_acc"].append(train_accuracy)
        fusion_history["val_acc"].append(val_accuracy)

        fusion_scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{FUSION_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # Early stopping
        if val_loss < best_fusion_val_loss - 1e-5:

            best_fusion_val_loss = val_loss

            best_fusion_weights = copy.deepcopy(
                fusion_model.state_dict()
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= FUSION_PATIENCE:
            print("Early stopping triggered.")
            break

    # --------------------------------------------------
    # Restore and save best fusion model
    # --------------------------------------------------

    if best_fusion_weights is None:
        raise RuntimeError("No best fusion weights were saved.")

    fusion_model.load_state_dict(best_fusion_weights)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = output_dir / "cross_modal_fusion_best.pth"

    torch.save(
        fusion_model.state_dict(),
        checkpoint_path
    )

    with open(
        output_dir / "fusion_training_history.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(fusion_history, f, indent=2)

    print("\nFusion training completed!")
    print("Best validation loss:", best_fusion_val_loss)
    print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train the TESS cross-modal fusion model."
    )

    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--full-manifest", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--vocab-file", required=True)

    parser.add_argument("--speech-checkpoint", required=True)
    parser.add_argument("--text-checkpoint", required=True)

    parser.add_argument(
        "--output-dir",
        default="Results/fusion_models"
    )

    parser.add_argument("--seed", type=int, default=42)

    train(parser.parse_args())
