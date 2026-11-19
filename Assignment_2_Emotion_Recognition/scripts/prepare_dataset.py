from pathlib import Path
import csv
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "Assignment_2_Emotion_Recognition"
    / "dataset"
    / "TESS Toronto emotional speech set data"
    / "TESS Toronto emotional speech set data"
)

OUTPUT_DIR = PROJECT_ROOT / "Assignment_2_Emotion_Recognition" / "Results"
OUTPUT_FILE = OUTPUT_DIR / "tess_manifest.csv"

SUFFIX_TO_EMOTION = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "ps": "pleasant_surprise",
}

rows = []
skipped = []

for folder in sorted(DATASET_DIR.iterdir()):
    if not folder.is_dir():
        continue

    folder_parts = folder.name.split("_", 1)
    if len(folder_parts) != 2:
        continue

    speaker, folder_emotion = folder_parts
    folder_emotion = folder_emotion.lower()

    if folder_emotion in ("pleasant_surprise", "pleasant_surprised"):
        expected_emotion = "pleasant_surprise"
    else:
        expected_emotion = folder_emotion

    for audio_file in sorted(folder.glob("*.wav")):
        parts = audio_file.stem.split("_")

        if len(parts) < 3:
            skipped.append(audio_file.name)
            continue

        file_speaker = parts[0]
        target_word = "_".join(parts[1:-1])
        suffix = parts[-1].lower()

        emotion = SUFFIX_TO_EMOTION.get(suffix)

        if (
            emotion is None
            or (file_speaker != speaker and not (speaker == "OAF" and file_speaker == "OA"))
            or emotion != expected_emotion
        ):
            skipped.append(audio_file.name)
            continue

        rows.append({
            "filename": audio_file.name,
            "audio_path": audio_file.relative_to(PROJECT_ROOT).as_posix(),
            "transcript": f"Say the word {target_word}",
            "emotion": emotion,
            "speaker": speaker,
            "target_word": target_word
        })

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

fieldnames = [
    "filename",
    "audio_path",
    "transcript",
    "emotion",
    "speaker",
    "target_word"
]

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("\nManifest generation completed!")
print("Total recordings indexed:", len(rows))
print("Files skipped:", len(skipped))
print("Skipped filenames:", skipped)
print("Unique speakers:", sorted(set(r["speaker"] for r in rows)))
print("Unique emotions:", sorted(set(r["emotion"] for r in rows)))

print("\nEmotion distribution:")
for emotion, count in sorted(Counter(r["emotion"] for r in rows).items()):
    print(f"{emotion}: {count}")

if len(rows) == 2800 and not skipped:
    print("\nSUCCESS: All 2800 recordings were indexed.")
else:
    print("\nWARNING: Dataset count needs further investigation.")

print("\nCSV saved at:", OUTPUT_FILE)
