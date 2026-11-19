import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from models.speech_pipeline.model import OriginalCNNTransformerSpeech
from models.speech_pipeline.train import TESSAudioDataset, EMOTIONS


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = TESSAudioDataset(
        args.test_manifest,
        args.full_manifest,
        args.feature_dir
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False
    )

    model = OriginalCNNTransformerSpeech().to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(checkpoint)
    model.eval()

    predictions = []
    true_labels = []
    confidences = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)

            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)

            predictions.extend(outputs.argmax(1).cpu().tolist())
            true_labels.extend(labels.tolist())
            confidences.extend(
                probabilities.max(dim=1).values.cpu().tolist()
            )

    accuracy = accuracy_score(true_labels, predictions)

    report = classification_report(
        true_labels,
        predictions,
        labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=list(range(len(EMOTIONS)))
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": "Speech CNN-Transformer",
        "test_samples": len(true_labels),
        "test_accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "emotion_labels": EMOTIONS
    }

    with open(output_dir / "speech_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(
        matrix,
        index=EMOTIONS,
        columns=EMOTIONS
    ).to_csv(output_dir / "speech_confusion_matrix.csv")

    predictions_df = test_ds.df.copy()
    predictions_df["true_emotion"] = [
        EMOTIONS[i] for i in true_labels
    ]
    predictions_df["predicted_emotion"] = [
        EMOTIONS[i] for i in predictions
    ]
    predictions_df["confidence"] = confidences

    predictions_df.to_csv(
        output_dir / "speech_test_predictions.csv",
        index=False
    )

    print(f"Test samples: {len(true_labels)}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(
        true_labels,
        predictions,
        labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS,
        zero_division=0
    ))
    print(f"\nResults saved in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--full-manifest", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="Results/tables")
    parser.add_argument("--batch-size", type=int, default=32)

    test(parser.parse_args())
