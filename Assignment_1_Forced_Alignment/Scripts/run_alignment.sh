#!/bin/bash
set -e

# Montreal Forced Aligner - Forced Alignment Workflow

conda activate aligner

CORPUS="$HOME/Speech_Assignment/Assignment_2a/corpus"
OUTPUT="$HOME/Speech_Assignment/Assignment_2a/outputs"
DICTIONARY="english_us_arpa"
ACOUSTIC_MODEL="english_us_arpa"

echo "Checking corpus..."
mfa validate "$CORPUS" "$DICTIONARY"

echo "Running forced alignment..."
mfa align "$CORPUS" "$DICTIONARY" "$ACOUSTIC_MODEL" "$OUTPUT"

echo "Alignment completed. TextGrid files are saved in: $OUTPUT"
