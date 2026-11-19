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

from models.fusion_pipeline.train import (
    EMOTIONS,
    SpeechFeatureExtractor,
    TextFeatureExtractor,
    TESSMultimodalDataset,
    load_checkpoint
)
from models.speech_pipeline.model import OriginalCNNTransformerSpeech
from models.text_pipeline.model import TransformerTextModel
from models.fusion_pipeline.model import CrossModalFusion


def evaluate(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    with open(args.vocab_file, "r", encoding="utf-8") as f:
        vocabularies = json.load(f)

    vocab = vocabularies["original_vocab"]
    print("Vocabulary size:", len(vocab))

    speech_model = OriginalCNNTransformerSpeech().to(device)
    speech_model = load_checkpoint(
        speech_model, args.speech_checkpoint, device
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
        text_model, args.text_checkpoint, device
    )

    speech_encoder = SpeechFeatureExtractor(speech_model).to(device)
    text_encoder = TextFeatureExtractor(text_model).to(device)

    speech_encoder.eval()
    text_encoder.eval()

    fusion_model = CrossModalFusion(
        d_model=128,
        num_heads=4,
        num_classes=len(EMOTIONS),
        dropout=0.3
    ).to(device)

    fusion_model = load_checkpoint(
        fusion_model, args.fusion_checkpoint, device
    )
    fusion_model.eval()

    dataset = TESSMultimodalDataset(
        args.test_manifest,
        args.full_manifest,
        args.feature_dir,
        vocab
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for audio, text, labels in loader:
            audio = audio.to(device)
            text = text.to(device)

            audio_features = speech_encoder(audio)
            text_features, text_padding_mask = text_encoder(text)

            outputs = fusion_model(
                audio_features,
                text_features,
                text_padding_mask
            )

            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(EMOTIONS)))
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": "Bidirectional Cross-Modal Fusion",
        "test_samples": len(all_labels),
        "test_accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "emotion_order": EMOTIONS
    }

    with open(output_dir / "fusion_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(
        matrix,
        index=EMOTIONS,
        columns=EMOTIONS
    ).to_csv(output_dir / "fusion_confusion_matrix.csv")

    predictions_df = pd.DataFrame({
        "true_emotion": [EMOTIONS[i] for i in all_labels],
        "predicted_emotion": [EMOTIONS[i] for i in all_predictions],
        "confidence": [
            max(probability) for probability in all_probabilities
        ]
    })

    predictions_df.to_csv(
        output_dir / "fusion_predictions.csv",
        index=False
    )

    print("\nFusion Test Results")
    print("-------------------")
    print("Test samples:", len(all_labels))
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(EMOTIONS))),
        target_names=EMOTIONS,
        zero_division=0
    ))
    print("Confusion Matrix:")
    print(matrix)
    print("\nResults saved to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the TESS multimodal fusion model."
    )

    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--full-manifest", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--vocab-file", required=True)
    parser.add_argument("--speech-checkpoint", required=True)
    parser.add_argument("--text-checkpoint", required=True)
    parser.add_argument("--fusion-checkpoint", required=True)
    parser.add_argument("--output-dir", default="Results/fusion_models")
    parser.add_argument("--batch-size", type=int, default=32)

    evaluate(parser.parse_args())
