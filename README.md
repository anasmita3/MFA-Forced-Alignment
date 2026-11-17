# Forced Alignment Using Montreal Forced Aligner (MFA)

## 1. Project Overview
This project performs forced alignment of English speech recordings with their corresponding transcripts using the Montreal Forced Aligner (MFA).

Forced alignment identifies the time boundaries of words and phonemes in an audio recording when the transcript is already known.

## 2. Tools and Models
- Montreal Forced Aligner (MFA) 3.4.3
- Python / Conda environment
- English US ARPA pronunciation dictionary
- English US ARPA acoustic model
- Praat for visualization and inspection

## 3. Dataset Preparation
The dataset contains six audio recordings and their corresponding transcript files.

Each audio file is paired with a transcript having the same filename.

Audio format:
- WAV
- Mono channel
- 16 kHz sampling rate
- 16-bit PCM

The files were organized into an MFA-compatible corpus structure.

## 4. Installation
Create and activate the Conda environment:

    conda create -n aligner python=3.10
    conda activate aligner

Install MFA:

    conda install -c conda-forge montreal-forced-aligner

Download the required models:

    mfa model download dictionary english_us_arpa
    mfa model download acoustic english_us_arpa

## 5. Validation and Alignment

Validate the corpus:

    mfa validate CORPUS_PATH english_us_arpa

Run forced alignment:

    mfa align CORPUS_PATH english_us_arpa english_us_arpa OUTPUT_PATH

## 6. Results
The alignment generated six TextGrid files and an alignment_analysis.csv file.

Each TextGrid contains:
- A words tier showing word-level intervals
- A phones tier showing phoneme-level intervals

## 7. Visualization and Inspection
The TextGrid files were opened with their corresponding audio recordings in Praat.

The word and phoneme boundaries were visually inspected to observe timing, segmentation, and possible alignment mismatches.

## 8. Repository Structure

    MFA_Final_Submission/
    ├── Audio/
    ├── TextGrids/
    ├── Praat_Pictures/
    ├── Scripts/
    │   ├── MFA_commands.txt
    │   └── run_alignment.sh
    ├── Report/
    └── README.md

## 9. Reproducibility
Place the provided WAV and transcript files in an MFA-compatible corpus directory before running the commands.

The original dataset is not included in this repository. Use the dataset provided with the assignment.
