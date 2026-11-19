import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from models.text_pipeline.model import (
    TextBiLSTM,
    TransformerTextModel,
    UpdatedTextTransformer
)

from models.text_pipeline.train import (
    EMOTIONS,
    TESSTextDataset
)


def evaluate_model(model, loader, device):
    model.eval()

    predictions = []
    true_labels = []
    probabilities = []

    with torch.no_grad():
        for tokens, labels in loader:
            tokens = tokens.to(device)

            outputs = model(tokens)
            probs = torch.softmax(outputs, dim=1)

            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            true_labels.extend(labels.numpy())

    accuracy = accuracy_score(true_labels, predictions)

    return accuracy, true_labels, predictions, probabilities
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test-manifest",
        default="Results/test_manifest.csv"
    )
    parser.add_argument(
        "--vocab-file",
        default="Results/text_models/vocabularies.json"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="Results/text_models"
    )
    parser.add_argument(
        "--output-dir",
        default="Results/text_models/evaluation"
    )
    parser.add_argument("--batch-size", type=int, default=32)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    os.makedirs(args.output_dir, exist_ok=True)

    test_df = pd.read_csv(args.test_manifest)

    with open(args.vocab_file, "r") as f:
        vocab_data = json.load(f)

    original_vocab = vocab_data["original_vocab"]
    secondary_vocab = vocab_data["secondary_vocab"]

    models = {
        "text_bilstm": (
            TextBiLSTM(vocab_size=len(original_vocab)),
            original_vocab,
            "original"
        ),
        "original_transformer": (
            TransformerTextModel(
                vocab_size=len(original_vocab),
                num_classes=len(EMOTIONS)
            ),
            original_vocab,
            "original"
        ),
        "updated_transformer": (
            UpdatedTextTransformer(
                vocab_size=len(secondary_vocab),
                num_classes=len(EMOTIONS)
            ),
            secondary_vocab,
            "updated"
        )
    }

    all_results = {}

    for name, (model, vocab, encoding) in models.items():
        print(f"\nEvaluating {name}...")

        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"{'transformer_text' if name == 'original_transformer' else 'updated_text_transformer' if name == 'updated_transformer' else name}_best.pth"
        )

        if not os.path.exists(checkpoint_path):
            print("Checkpoint not found:", checkpoint_path)
            continue

        dataset = TESSTextDataset(
            test_df,
            vocab,
            encoding
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False
        )

        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True
            )
        )

        model.to(device)

        accuracy, labels, predictions, probabilities = evaluate_model(
            model, loader, device
        )

        report = classification_report(
            labels,
            predictions,
            labels=list(range(len(EMOTIONS))),
            target_names=EMOTIONS,
            output_dict=True,
            zero_division=0
        )

        matrix = confusion_matrix(
            labels,
            predictions,
            labels=list(range(len(EMOTIONS)))
        )

        print(f"Test Accuracy: {accuracy * 100:.2f}%")

        np.save(
            os.path.join(args.output_dir, f"{name}_confusion_matrix.npy"),
            matrix
        )

        pd.DataFrame(
            matrix,
            index=EMOTIONS,
            columns=EMOTIONS
        ).to_csv(
            os.path.join(args.output_dir, f"{name}_confusion_matrix.csv")
        )

        predictions_df = test_df.copy()
        predictions_df["true_emotion"] = [
            EMOTIONS[i] for i in labels
        ]
        predictions_df["predicted_emotion"] = [
            EMOTIONS[i] for i in predictions
        ]
        predictions_df["confidence"] = [
            float(probabilities[i][predictions[i]])
            for i in range(len(predictions))
        ]

        predictions_df.to_csv(
            os.path.join(args.output_dir, f"{name}_predictions.csv"),
            index=False
        )

        all_results[name] = {
            "test_accuracy": float(accuracy),
            "classification_report": report
        }

    with open(
        os.path.join(args.output_dir, "evaluation_results.json"),
        "w"
    ) as f:
        json.dump(all_results, f, indent=4)

    print("\nText evaluation completed.")
    print("Results saved to:", args.output_dir)


if __name__ == "__main__":
    main()
