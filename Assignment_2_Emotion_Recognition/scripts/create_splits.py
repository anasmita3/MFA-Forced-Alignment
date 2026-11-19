from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "Assignment_2_Emotion_Recognition" / "Results"

df = pd.read_csv(RESULTS / "tess_manifest.csv")

# First split: 70% training, 30% temporary
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.30, random_state=42
)

train_idx, temp_idx = next(
    split1.split(df, groups=df["target_word"])
)

train = df.iloc[train_idx].copy()
temp = df.iloc[temp_idx].copy()

# Second split: divide temporary data equally
split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.50, random_state=42
)

val_idx, test_idx = next(
    split2.split(temp, groups=temp["target_word"])
)

val = temp.iloc[val_idx].copy()
test = temp.iloc[test_idx].copy()

for name, part in [
    ("train", train),
    ("validation", val),
    ("test", test)
]:
    part.to_csv(RESULTS / f"{name}_manifest.csv", index=False)

    print(f"\n{name.upper()}: {len(part)} recordings")
    print(part["emotion"].value_counts().sort_index())

# Verify target-word separation
train_words = set(train["target_word"])
val_words = set(val["target_word"])
test_words = set(test["target_word"])

assert train_words.isdisjoint(val_words)
assert train_words.isdisjoint(test_words)
assert val_words.isdisjoint(test_words)

print("\nSUCCESS: No target-word overlap between splits.")
