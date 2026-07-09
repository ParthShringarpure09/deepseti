# DeepSeti

**Self-supervised anomaly detection for radio technosignatures.**

DeepSeti trains a Vision Transformer Masked Autoencoder (ViT-MAE) exclusively
on spectrograms of *normal* space radio noise from the [Breakthrough Listen](https://breakthroughlisten.org/)
public data archive. The model learns to reconstruct masked patches of
ordinary RFI-free noise; at inference time, spectrograms the model
reconstructs *poorly* are flagged as anomalous — "this looks weird" —
without ever having seen a labeled technosignature.

This label-free framing is the core contribution: most SETI signal
detection today (`turboSETI`, CNN classifiers) relies on hand-engineered
heuristics or supervised labels for signal shapes we expect. DeepSeti
instead asks a simpler question — *does this look like anything we've seen
before?* — which in principle generalizes to signal morphologies no one
has thought to label.

> **Status:** early development (Phase 1 — project scaffolding).
> Target venues: NeurIPS ML4PS workshop (primary), *Acta Astronautica* (backup).

## Motivation

Radio SETI searches process enormous volumes of spectrogram data from
telescopes like the Green Bank Telescope and Parkes. The dominant search
strategy (`turboSETI`) looks for narrowband drifting signals matching a
specific physical model, and supervised classifiers only recognize signal
shapes present in their training labels. Both approaches share a blind
spot: they can only find what they were told to look for.

DeepSeti explores a complementary, label-free strategy grounded in
self-supervised representation learning: train a reconstruction model on
the overwhelming majority-class signal (background noise + RFI), and treat
reconstruction error as an anomaly score.

## Method

1. **Data**: Filterbank/HDF5 cadences from Breakthrough Listen public
   archives are converted into fixed-size spectrogram patches.
2. **Pretraining**: A ViT-MAE (via Hugging Face `transformers`) is trained
   to reconstruct heavily masked (~75%) patches of "quiet" spectrograms.
3. **Anomaly scoring**: Reconstruction error is computed per spectrogram
   and calibrated into an anomaly score.
4. **Validation**: Synthetic narrowband, drifting, pulsed, and broadband
   signals are injected via `setigen` and detection is evaluated with
   AUROC/AUPRC and precision/recall at fixed false-positive budgets.

## Project structure
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
.mypy_cache/
.ruff_cache/
.pytest_cache/

# Virtual environments
.venv/
venv/
.conda/

# Jupyter
.ipynb_checkpoints/

# Data (never commit raw or processed data)
data/raw/*
data/processed/*
data/synthetic/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/synthetic/.gitkeep
*.h5
*.hdf5
*.fil
*.npy
*.npz

# Model checkpoints & experiment artifacts
experiments/*
!experiments/.gitkeep
*.pt
*.pth
*.ckpt

# Weights & Biases
wandb/

# Environment / secrets
.env
*.key

# OS / editor
.DS_Store
.vscode/
.idea/
