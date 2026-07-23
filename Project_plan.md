# DeepSeti — Project Plan

## Phase 1 — Project Scaffolding ✅
- [x] Repo structure (src/, data/, config/, experiments/, notebooks/, tests/)
- [x] README, environment.yml, requirements.txt
- [x] .gitignore for data/checkpoints
- [x] Initial commit + push

## Phase 2 — Synthetic "Quiet" Data Generation (in progress)
- [x] `generate_quiet_frame()` using setigen
- [x] Fix seed reproducibility bug (pass seed into `stg.Frame`, not `np.random.seed`)
- [ ] Explore output distribution/shape in notebook (done informally — formalize in notebook)
- [ ] Batch-generate N quiet frames, save to `data/synthetic/`
- [ ] Commit + push Phase 2

## Phase 3 — Synthetic Signal Injection
- [ ] `inject_synthetic_signal()` using setigen (`add_signal`, drift-rate, SNR control)
- [ ] Generate frames spanning a range of SNR (5–20 dB) and drift rates
- [ ] Visual sanity check: quiet vs. injected frame side by side
- [ ] Save labeled synthetic dataset (frame + ground-truth signal params)
- [ ] Commit + push Phase 3

## Phase 4 — Real Breakthrough Listen Data Ingestion
- [ ] Download sample GBT/Parkes cadence (.h5 / filterbank)
- [ ] Parse with `blimpy`, extract spectrogram chunks
- [ ] Normalize/preprocess to match synthetic data format
- [ ] Commit + push Phase 4

## Phase 5 — Spectrogram Preprocessing Pipeline
- [ ] Unified data loader (real + synthetic, consistent shape/scaling)
- [ ] Train/val/test split logic
- [ ] PyTorch `Dataset` / `DataLoader` classes
- [ ] Commit + push Phase 5

## Phase 6 — ViT-MAE Model
- [ ] Patchify spectrograms
- [ ] ViT-MAE architecture (HuggingFace `transformers` or custom)
- [ ] Masking strategy (~75% random patches)
- [ ] Commit + push Phase 6

## Phase 7 — Self-Supervised Pretraining
- [ ] Training loop (loss, optimizer, logging via wandb)
- [ ] Train on quiet/real background data (Colab, GPU)
- [ ] Checkpointing
- [ ] Commit + push Phase 7

## Phase 8 — Reconstruction Error / Anomaly Scoring
- [ ] Compute per-spectrogram reconstruction error (MSE)
- [ ] Calibrate error → anomaly score
- [ ] Commit + push Phase 8

## Phase 9 — Validation via Synthetic Injection
- [ ] Run model on labeled synthetic (Phase 3) dataset
- [ ] Compute AUROC / AUPRC, precision/recall at fixed FP budget
- [ ] Sensitivity curves across SNR/drift-rate
- [ ] Commit + push Phase 9

## Phase 10 — Results Analysis
- [ ] Rank real data by anomaly score
- [ ] Manual RFI triage on top candidates
- [ ] Compile results tables/figures
- [ ] Commit + push Phase 10

## Phase 11 — Paper Writing
- [ ] Draft manuscript sections
- [ ] Finalize figures (pipeline diagram, Venn diagram, results plots)
- [ ] Internal review
- [ ] Commit + push Phase 11

## Phase 12 — Submission
- [ ] Target venue formatting (NeurIPS ML4PS / Acta Astronautica)
- [ ] Submit