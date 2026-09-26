# DeepSeti

**Self-Supervised Masked Spectrogram Modeling for Anomaly Detection in Radio Technosignature Searches**

DeepSeti is a research project investigating whether self-supervised models trained only on ordinary radio observations can identify previously unseen technosignature-like signals as anomalies.

## Research Question

Can a masked autoencoder trained exclusively on normal radio spectrograms detect unseen technosignature-like signals without synthetic technosignatures being used during training?

## Planned Pipeline

Breakthrough Listen observations  
→ preprocessing  
→ spectrogram frames  
→ normal-only self-supervised training  
→ anomaly scoring  
→ synthetic injection benchmark  
→ real-data candidate ranking

## Model Comparison

The project will compare:

1. PCA reconstruction
2. Convolutional autoencoder
3. Vision Transformer Masked Autoencoder (ViT-MAE)

## Scientific Principle

Synthetic technosignatures are used only for validation and testing.

They are **never used to train the anomaly-detection models**.

Where possible, train, validation, and test splits will be created at the observation level to prevent leakage between chunks from the same observation.

## Evaluation

Primary evaluation will focus on anomaly-detection metrics rather than classification accuracy, including:

- AUROC
- AUPRC
- recall / detection rate
- false-positive rate
- recall at low false-positive rates
- anomaly-score distributions
- detection performance versus SNR
- detection performance versus drift rate

## Project Structure

```text
deepseti/
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── notebooks/
├── scripts/
├── src/deepseti/
│   ├── data/
│   ├── preprocessing/
│   ├── injection/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── tests/
├── results/
├── docs/
└── paper/
```

## Environment

- Python 3.12
- PyTorch
- Apple Silicon MPS / CUDA / CPU device support
- Dependency management with `uv`

## Project Status

Current milestone: **M0 — Repository and research environment setup**

## Research Scope

The initial research prototype will use public Breakthrough Listen data and controlled synthetic technosignature injections.

The synthetic benchmark will include signal morphologies such as:

- stationary narrowband signals
- linear drifting signals
- nonlinear or chirped signals
- intermittent signals
- multiple carriers
- burst-like signals

Synthetic signals are reserved for validation and testing only.

## Reproducibility

The repository is designed to support reproducible experiments using:

- fixed random seeds
- configuration files
- observation-level dataset splits
- version-controlled source code
- locked Python dependencies
- documented experiment outputs

Large raw datasets, model checkpoints, temporary files, and secrets are intentionally excluded from version control.

## License

License to be determined before public release.