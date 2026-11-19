# Multimodal Emotion Recognition Using Speech and Text

## 1. Project Overview

This project explores emotion recognition using three approaches:

1. Speech-only emotion recognition
2. Text-only emotion recognition
3. Multimodal speech and text emotion recognition

The experiments use the Toronto Emotional Speech Set (TESS), containing 2,800 audio recordings across seven emotion classes.

## 2. Emotion Classes

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Pleasant Surprise
- Sad

## 3. Project Structure

Assignment_2_Emotion_Recognition/
│
├── models/
│   ├── speech_pipeline/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── test.py
│   │
│   ├── text_pipeline/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── test.py
│   │
│   └── fusion_pipeline/
│       ├── __init__.py
│       ├── model.py
│       ├── train.py
│       └── test.py
│
├── scripts/
│   ├── prepare_dataset.py
│   ├── create_splits.py
│   └── preprocess_audio.py
│
├── Results/
│   ├── plots/
│   │   ├── temporal_pca.png
│   │   ├── temporal_tsne.png
│   │   ├── contextual_pca.png
│   │   ├── contextual_tsne.png
│   │   ├── original_text_pca.png
│   │   ├── text_pca.png
│   │   ├── text_tsne.png
│   │   ├── updated_text_pca.png
│   │   ├── updated_text_tsne.png
│   │   ├── fusion_pca.png
│   │   └── fusion_tsne.png
│   │
│   ├── tables/
│   │   ├── speech_confusion_matrix.csv
│   │   ├── speech_test_predictions.csv
│   │   └── speech_test_results.json
│   │
│   ├── fusion_models/
│   │   ├── fusion_confusion_matrix.csv
│   │   ├── fusion_predictions.csv
│   │   └── fusion_test_results.json
│   │
│   ├── text_models/
│   │   └── vocabularies.json
│   │
│   ├── tess_manifest.csv
│   ├── train_manifest.csv
│   ├── validation_manifest.csv
│   ├── test_manifest.csv
│   └── cluster_separability_metrics.csv
│
├── Report/
│   └── TESS_Emotion_Recognition_Report.md
│
├── TESS_Emotion_Recognition.ipynb
├── requirements.txt
└── README.md

### Directory Description

- **models/** – Model architectures and independent training/testing pipelines.
- **speech_pipeline/** – Speech-only emotion recognition using CNN–Transformer and BiLSTM architectures.
- **text_pipeline/** – Text-only emotion recognition using BiLSTM and Transformer architectures.
- **fusion_pipeline/** – Bidirectional cross-modal attention and multimodal classification.
- **scripts/** – Dataset manifest generation and audio preprocessing.
- **Results/** – Evaluation outputs, embeddings, plots, checkpoints and dataset manifests.
- **Report/** – Detailed project report and experimental analysis.
- **TESS_Emotion_Recognition.ipynb** – Notebook containing the experiments.

## 4. Dataset and Preprocessing

TESS contains 2,800 recordings distributed equally across seven emotion classes.

Transcripts are derived from the target words encoded in the filenames and the common carrier phrase, "Say the word".

### Audio preprocessing

- Convert audio to mono.
- Resample to 16 kHz.
- Trim silence.
- Extract 64-bin log-Mel spectrograms.
- FFT size: 1024.
- Hop length: 256.
- Pad or truncate to 300 frames.

Final speech feature shape: 64 × 300.

### Text preprocessing

Text is lowercased, tokenized and converted into integer sequences using vocabulary mappings, with padding and unknown-token handling.

### Dataset split

| Split | Samples |
|---|---:|
| Training | 1,960 |
| Validation | 420 |
| Testing | 420 |
| Total | 2,800 |

The split is grouped by target word, preventing target-word overlap. Speakers are not held out, so this is not a speaker-independent evaluation.

## 5. Model Architectures

### Speech Pipeline

A CNN–Transformer extracts local acoustic patterns and models temporal context. Attention pooling generates a fixed-size representation for classification. A BiLSTM speech model was also evaluated.

### Text Pipeline

The text experiments include:
- Bidirectional LSTM
- Transformer encoder
- Updated Transformer with an expanded vocabulary

### Multimodal Fusion Pipeline

The fusion model uses pretrained speech and text encoders with bidirectional cross-modal attention. The enhanced representations are pooled and combined before classification. The pretrained encoders are frozen during fusion training.

## 6. Experimental Results

| Model | Test Accuracy |
|---|---:|
| Speech BiLSTM | 99.52% |
| Speech CNN–Transformer | 99.52% |
| Text BiLSTM | 14.29% |
| Original Text Transformer | 14.29% |
| Updated Text Transformer | 14.29% |
| Cross-modal Fusion | 100.00% |

The text-only models perform at chance level for seven classes. TESS uses the same transcript content across emotion categories, with each target-word transcript occurring equally often in every emotion class. Therefore, transcript identity does not provide a meaningful emotion-label signal in this experimental setup.

Although fusion achieves 100% test accuracy, shuffling transcript pairings also retains 100% accuracy. The experiments therefore do not establish that correctly paired text improves performance; the fusion model may rely primarily on speech.

## 7. Evaluation and Visualizations

The Results directory contains evaluation outputs, confusion matrices, prediction tables and representation visualizations.

The analysis includes:
- Temporal embeddings
- Contextual embeddings
- Text embedding
- Fusion embeddings
- PCA and t-SNE plots
- Cluster separability metrics

These visualizations are descriptive analyses of test-set representations, not independent evidence of generalization.


## 8. Running the Project

### Environment Setup

Use Python 3.10 or later. Install dependencies:

```bash
pip install -r requirements.txt
```

Run all commands from the `Assignment_2_Emotion_Recognition` directory.

### Data Requirements

The original TESS dataset and preprocessed audio features are not included in the public repository.

Ensure the following files and directory are available:

- `Results/tess_manifest.csv`
- `Results/train_manifest.csv`
- `Results/validation_manifest.csv`
- `Results/test_manifest.csv`
- `Results/audio_features/`

### 1. Speech Model Training

```bash
python -m models.speech_pipeline.train \
  --train-manifest Results/train_manifest.csv \
  --val-manifest Results/validation_manifest.csv \
  --full-manifest Results/tess_manifest.csv \
  --feature-dir Results/audio_features
```

Checkpoint: `Results/checkpoints/speech_cnn_transformer_best.pth`

### 2. Text Model Training

```bash
python -m models.text_pipeline.train
```

Trains the Text BiLSTM, Original Transformer and Updated Transformer.

### 3. Multimodal Fusion Training

Requires pretrained speech and original text Transformer checkpoints.

```bash
python -m models.fusion_pipeline.train \
  --train-manifest Results/train_manifest.csv \
  --val-manifest Results/validation_manifest.csv \
  --full-manifest Results/tess_manifest.csv \
  --feature-dir Results/audio_features \
  --vocab-file Results/text_models/vocabularies.json \
  --speech-checkpoint Results/checkpoints/speech_cnn_transformer_best.pth \
  --text-checkpoint Results/text_models/transformer_text_best.pth
```

### 4. Model Evaluation

Speech evaluation:

```bash
python -m models.speech_pipeline.test \
  --test-manifest Results/test_manifest.csv \
  --full-manifest Results/tess_manifest.csv \
  --feature-dir Results/audio_features \
  --checkpoint Results/checkpoints/speech_cnn_transformer_best.pth
```

Text evaluation:

```bash
python -m models.text_pipeline.test
```

Fusion evaluation:

```bash
python -m models.fusion_pipeline.test \
  --test-manifest Results/test_manifest.csv \
  --full-manifest Results/tess_manifest.csv \
  --feature-dir Results/audio_features \
  --vocab-file Results/text_models/vocabularies.json \
  --speech-checkpoint Results/checkpoints/speech_cnn_transformer_best.pth \
  --text-checkpoint Results/text_models/transformer_text_best.pth \
  --fusion-checkpoint Results/fusion_models/cross_modal_fusion_best.pth
```

### Reproducibility Notes

- Scripts automatically use CUDA when available, otherwise CPU.
- Training uses fixed random seeds.
- Dataset files, preprocessed features and model checkpoints are excluded from the public repository.
- The commands above assume the default output directories.


## 9. Limitations

- TESS is a controlled acted-emotion dataset and does not represent unrestricted natural conversation.
- Text content is balanced across emotion labels, limiting text-only classification.
- The evaluation split is target-word-disjoint but not speaker-independent.
- The shuffled-text experiment does not demonstrate a benefit from correct transcript alignment.
- High accuracy on this dataset should not be interpreted as evidence of equivalent real-world performance.

## 10. Detailed Report

See [TESS Emotion Recognition Report](Report/TESS_Emotion_Recognition_Report.pdf) for the methodology, architecture details, evaluation, failure analysis and discussion.
