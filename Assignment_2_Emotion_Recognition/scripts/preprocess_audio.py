from pathlib import Path
import numpy as np
import pandas as pd
import librosa

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "Assignment_2_Emotion_Recognition"
    / "Results"
    / "tess_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Assignment_2_Emotion_Recognition"
    / "Results"
    / "audio_features"
)

SAMPLE_RATE = 16000
N_MELS = 64
MAX_FRAMES = 300

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(MANIFEST)

successful = 0
failed = []

for index, row in data.iterrows():
    try:
        audio_path = PROJECT_ROOT / row["audio_path"]

        # Load audio and convert to mono, 16 kHz
        signal, sr = librosa.load(
            audio_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        # Trim leading and trailing silence
        signal, _ = librosa.effects.trim(signal, top_db=25)

        if len(signal) == 0:
            raise ValueError("Empty audio after silence trimming")

        # Extract log-Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=signal,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=1024,
            hop_length=256
        )

        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Normalize each recording
        mean = log_mel.mean()
        std = log_mel.std()
        log_mel = (log_mel - mean) / (std + 1e-8)

        # Standardize temporal length
        frames = log_mel.shape[1]

        if frames < MAX_FRAMES:
            log_mel = np.pad(
                log_mel,
                ((0, 0), (0, MAX_FRAMES - frames)),
                mode="constant"
            )
        else:
            log_mel = log_mel[:, :MAX_FRAMES]

        output_file = OUTPUT_DIR / f"{index:04d}.npy"
        np.save(output_file, log_mel.astype(np.float32))

        successful += 1

        if successful % 200 == 0:
            print(f"Processed {successful}/{len(data)} recordings")

    except Exception as error:
        failed.append((row["filename"], str(error)))

print("\nAudio preprocessing completed!")
print("Successful:", successful)
print("Failed:", len(failed))
print("Feature shape:", (N_MELS, MAX_FRAMES))
print("Saved features to:", OUTPUT_DIR)

if failed:
    print("\nFirst failed files:")
    for filename, error in failed[:10]:
        print(filename, "->", error)

if successful == len(data):
    print("\nSUCCESS: All audio recordings were processed.")
