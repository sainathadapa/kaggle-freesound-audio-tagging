#!/bin/bash
set -e

PIP=.env/bin/pip
PYTHON=.env/bin/python

# Create virtual environment
virtualenv .env -p python3

# Install dependencies
"$PIP" install -r requirements.txt

# Extract files
unzip -o -d data/ data/audio_train.zip
unzip -o -d data/ data/audio_test.zip

# Trim leading and trailing silence
"$PYTHON" 01-save-trimmed-wavs.py

# Compute Log Mel-Spectrograms
"$PYTHON" 02-compute-mel-specs.py

# Compute summary metrics of various spectral and time based features
"$PYTHON" 03-compute-summary-feats.py

# Compute PCA features over the summary metrics from previous script
"$PYTHON" 04-pca.py

# Divide the training data into 10 (stratified by label) folds
"$PYTHON" 05-ten-folds.py

# Train only the part of the model, that depends on the Log Mel-Spec features (10 folds)
for (( FOLD=0; FOLD<=9; FOLD+=1 )); do
  "$PYTHON" 06-train-model-only-mel.py "$FOLD"
done

# Train the full model, after loading weights from the mel-only model from previous script (10 folds)
for (( FOLD=0; FOLD<=9; FOLD+=1 )); do
  "$PYTHON" 07-train-model-mel-and-pca.py "$FOLD"
done

# Generate predictions
"$PYTHON" 08-generate-predictions.py
