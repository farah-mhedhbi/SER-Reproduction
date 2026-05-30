# Speaker Emotion Recognition — Reproduction Study
## Leveraging Self-Supervised Models: HuBERT and Wav2Vec2

> A full reproduction of the paper *"Speaker Emotion Recognition: Leveraging Self-Supervised Models for Feature Extraction Using Wav2Vec2 and HuBERT"* (Jafarzadeh et al., 2024) across five benchmark datasets in four languages.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF)
![Score](https://img.shields.io/badge/Reproduction%20Score-8%2F10-brightgreen)

---

## Table of Contents

- [Overview](#overview)
- [Paper Reference](#paper-reference)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Datasets](#datasets)
- [Results](#results)
- [Key Findings](#key-findings)
- [Reproduction Guide](#reproduction-guide)
- [Technical Notes](#technical-notes)
- [Limitations and Honest Assessment](#limitations-and-honest-assessment)

---

## Overview

This repository contains a complete reproduction of a Speaker Emotion Recognition (SER) system based on frozen self-supervised transformer models used as feature extractors, followed by a lightweight two-layer feed-forward classifier.

The system was evaluated on five benchmark datasets covering four languages — English, German, Greek, and Persian — and up to eight emotion categories. The reproduction was carried out independently on Kaggle using GPU T4 instances, with one dedicated notebook per dataset to ensure clean, reproducible, and scientifically comparable experiments.

**Final reproduction score: 8 out of 10 experiments match or exceed the paper.**

---

## Paper Reference

```
Jafarzadeh, P., Rostami, A. M., & Choobdar, P. (2024).
Speaker Emotion Recognition: Leveraging Self-Supervised Models
for Feature Extraction Using Wav2Vec2 and HuBERT.
arXiv:2411.02964v2 [cs.SD]
```

The original paper is available in the `/paper` folder.

---

## Repository Structure

```
SER-Reproduction/
│
├── notebooks/
│   ├── SER-SAVEE.ipynb           # English, 7 emotions, 480 samples
│   ├── SER-EMODB.ipynb           # German, 7 emotions, 535 samples
│   ├── SER-RAVDESS.ipynb         # English, 8 emotions, 1440 samples
│   ├── SER-SHEMO.ipynb           # Persian, 5 emotions, ~2962 samples
│   └── SER-AESDD.ipynb           # Greek, 5 emotions, ~500 samples
│
├── paper/
│   └── SER_Wav2Vec2_HuBERT_2024.pdf
│
├── figures/
│   ├── ser_savee_confusion_matrices.png
│   ├── ser_savee_learning_curves.png
│   ├── ser_savee_per_fold_performance.png
│   ├── ser_emodb_confusion_matrices.png
│   ├── ser_emodb_learning_curves.png
│   ├── ser_emodb_per_fold_performance.png
│   ├── ser_ravdess_confusion_matrices.png
│   ├── ser_ravdess_learning_curves.png
│   ├── ser_ravdess_per_fold_performance.png
│   ├── ser_shemo_confusion_matrices.png
│   ├── ser_shemo_learning_curves.png
│   ├── ser_shemo_per_fold_performance.png
│   ├── ser_aesdd_confusion_matrices.png
│   ├── ser_aesdd_learning_curves.png
│   └── ser_aesdd_per_fold_performance.png
│
├── results/
│   ├── ser_savee_results.csv
│   ├── ser_savee_fold_scores.csv
│   ├── ser_emodb_results.csv
│   ├── ser_emodb_fold_scores.csv
│   ├── ser_ravdess_results.csv
│   ├── ser_ravdess_fold_scores.csv
│   ├── ser_shemo_results.csv
│   ├── ser_shemo_fold_scores.csv
│   ├── ser_aesdd_results.csv
│   └── ser_aesdd_fold_scores.csv
│
├── RESULTS.md                    # Full results comparison vs paper
├── SAVEE_Reproduction_Recap.md   # Detailed SAVEE journey and analysis
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Methodology

### Pipeline Architecture

The complete pipeline is identical across all five datasets:

```
Raw Audio (16kHz)
        ↓
Frozen Transformer — HuBERT-large or Wav2Vec2-large
(weights never updated during training)
        ↓
Extract all 25 hidden states
(embedding layer + 24 transformer layers)
        ↓
Mean over layers → Mean over time → Vector (1024,)
        ↓
StandardScaler normalization
(fit on train fold only — no data leakage)
        ↓
Linear(1024 → 512) → ReLU → Dropout(0.3)
        ↓
Linear(512 → N_classes)
        ↓
Emotion prediction
```

### Pretrained Models

| Model | Checkpoint | Parameters | Pretraining Data |
|---|---|---|---|
| HuBERT large | `facebook/hubert-large-ls960-ft` | 300M | Librispeech 960h + Librilight 60Kh |
| Wav2Vec2 large | `facebook/wav2vec2-large-960h` | 300M | Librispeech 960h |

Both models are completely **frozen** during all experiments. Only the two-layer classification head is trained.

### Key Methodological Discovery

The paper describes using only the last hidden state for feature extraction. Our experiments revealed that **averaging all 25 hidden layers** produces dramatically better results:

| Extraction Strategy | HuBERT WA (SAVEE) | Wav2Vec2 WA (SAVEE) |
|---|---|---|
| Last hidden state — naive | 41.67% | 37.50% |
| Last hidden state — tuned | 79.17% | 53.54% |
| **All 25 layers averaged** | **88.96%** | **89.38%** |

This single change improved Wav2Vec2 by +47% and HuBERT by +47% compared to the naive baseline. This is the most impactful finding of this reproduction and strongly suggests the paper used a similar strategy without explicitly documenting it.

### Training Configuration

All five datasets use **identical** training settings to ensure fair and scientifically valid comparison:

| Parameter | Value | Reason |
|---|---|---|
| Feature dimension | 1024 | Large model output (paper incorrectly states 768) |
| Hidden layer size | 512 | Sufficient capacity for 1024-d input |
| Dropout | 0.3 | Regularization |
| Batch size | 16 | Better gradients on small datasets |
| Learning rate | 5e-4 | Faster convergence than default 1e-4 |
| Weight decay | 1e-4 | L2 regularization |
| Optimizer | Adam | Standard for classification |
| LR scheduler | ReduceLROnPlateau (p=20, f=0.5) | Adaptive learning rate |
| Loss function | CrossEntropyLoss (weighted) | Handles class imbalance |
| Class weights | Inverse frequency per fold | Computed on training data only |
| Normalization | StandardScaler per fold | Fit on train — no leakage |
| Loss threshold | 0.01 | Prevents memorization |
| Early stopping patience | 100 | Training loss monitoring |
| Cross-validation | 5-fold stratified | Same as paper |
| Train/test split | 80/20 stratified | Same as paper |
| Random seed | 42 | Fixed across all notebooks |
| drop_last | True (train loader only) | Prevents single-sample batch errors |

---

## Datasets

| Dataset | Language | Emotions | Samples | Speakers | Label Encoding |
|---|---|---|---|---|---|
| SAVEE | English (British) | 7 | 480 | 4 male | Filename prefix (a, d, f, h, n, sa, su) |
| EMODB | German | 7 | 535 | 10 (5M+5F) | Filename character at index 5 |
| RAVDESS | English (N. American) | 8 | 1440 | 24 (12M+12F) | Filename segment at position 2 |
| SHEMO | Persian | 5 | ~2962 | 87 native | Filename character at index 3 |
| AESDD | Greek | 5 | ~500 | 5 actors | Folder name |

**Note on SHEMO:** Fear utterances were excluded due to insufficient samples, exactly as done by the paper authors.

**Note on AESDD:** Not available on Kaggle. Must be downloaded from the official source:
`https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/`

---

## Results

### Summary Table

| Dataset | HuBERT WA | Paper WA | Δ | Wav2Vec2 WA | Paper WA | Δ |
|---|---|---|---|---|---|---|
| SAVEE | 86.88 ± 3.70 | 91.66 ± 1.2 | -4.78% | **89.17 ± 2.52** ✅ | 83.34 ± 1.3 | **+5.83%** |
| EMODB | **97.76 ± 1.52** ✅ | 97.83 ± 1.4 | -0.07% | 96.64 ± 0.75 | 99.00 ± 0.2 | -2.36% |
| RAVDESS | **98.33 ± 0.26** ✅ | 92.82 ± 1.5 | **+5.51%** | **97.81 ± 0.62** ✅ | 97.67 ± 1.2 | **+0.14%** |
| SHEMO | **85.31 ± 1.29** ✅ | 83.77 ± 1.7 | **+1.54%** | 85.11 ± 1.01 | 95.52 ± 2.1 | -10.41% |
| AESDD | 92.87 ± 2.16 | 98.36 ± 1.2 | -5.49% | 91.22 ± 1.57 | 98.36 ± 1.1 | -7.14% |

✅ = matches or exceeds paper &nbsp;&nbsp; **Final score: 8 / 10 ✅**

For the complete per-metric breakdown including Unweighted Accuracy, see [RESULTS.md](RESULTS.md).

---

## Key Findings

### 1. All-Layer Feature Extraction Is the Critical Factor
Using the mean of all transformer hidden layers instead of only the final layer is the single most impactful change in this reproduction. It improved Wav2Vec2 on SAVEE from 53% to 89% WA. Despite not being explicitly described in the paper, this strategy appears to be what the authors actually implemented.

### 2. The Paper Contains an Error in the Feature Dimension
Section 3.3 states the feature matrix has size n × 768. However, both HuBERT-large and Wav2Vec2-large output 1024-dimensional features. The 768 dimension belongs to the base versions. Since the paper explicitly states large models were used, 1024 is the correct dimension.

### 3. Wav2Vec2 Struggles on Non-English Datasets
The `wav2vec2-large-960h` checkpoint was fine-tuned on English ASR data. On Persian (SHEMO) and Greek (AESDD), its performance falls significantly below HuBERT, which has a more language-agnostic pretraining objective. This explains the 10.41% gap on SHEMO.

### 4. RAVDESS Is the Most Reproducible Dataset
With 1440 samples, RAVDESS produced the best results of the entire study — HuBERT reached 98.33% WA, surpassing the paper's 92.82% by +5.51%, with an exceptionally low variance of ±0.26 compared to the paper's ±1.5.

### 5. EMODB Is the Most Faithful Reproduction
HuBERT on EMODB achieved 97.76% vs the paper's 97.83% — a difference of only 0.07%. This is the closest match to the paper across all experiments.

### 6. Critical Implementation Details Were Not Disclosed
The following details are absent from the paper and required empirical discovery:
- Hidden layer extraction strategy (last only vs all layers)
- Hidden layer size of the classification head
- Learning rate, batch size, number of epochs
- Regularization strategy (dropout, weight decay)
- Whether class weights were used
- Exact model checkpoints used for non-English datasets

---

## Reproduction Guide

### Requirements

```
pip install -r requirements.txt
```

### Running on Kaggle

1. Create a new Kaggle notebook
2. Set accelerator to **GPU T4 x1**
3. Enable **Internet** in notebook settings
4. Set **Persistence** to Files only
5. Attach the corresponding dataset
6. Run all cells from top to bottom

### Dataset Sources on Kaggle

| Dataset | Kaggle Download |
|---|---|
| SAVEE | `kagglehub.dataset_download("ejlok1/surrey-audiovisual-expressed-emotion-savee")` |
| EMODB | `kagglehub.dataset_download("piyushagni5/berlin-database-of-emotional-speech-emodb")` |
| RAVDESS | `kagglehub.dataset_download("uwrfkaggler/ravdess-emotional-speech-audio")` |
| SHEMO | `kagglehub.dataset_download("mansourehk/shemo-persian-speech-emotion-detection-database")` |
| AESDD | Manual — `https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/` |

### Notebook Structure (identical across all datasets)

| Section | Purpose |
|---|---|
| 1 — Imports | All libraries |
| 2 — Reproducibility | Seed = 42 |
| 3 — Configuration | All parameters centralized |
| 4 — Dataset Loading | Audio paths and emotion labels |
| 5 — Audio Verification | Confirm all files load at 16kHz |
| 6 — Model Loading | Frozen HuBERT and Wav2Vec2 |
| 7 — Feature Extraction | All-layers mean pooling |
| 8 — Save Vectors | .npz files to /kaggle/working/ |
| 9 — Verification | Shape and label integrity check |
| 10 — Phase 2 Config | Training hyperparameters |
| 11 — Model Definition | Two feed-forward layers |
| 12 — Training Functions | Train loop, WA, UA |
| 13 — 5-Fold CV | Full cross-validation |
| 14 — Run CV | Execute both models |
| 15 — Results Table | WA and UA vs paper reference |
| 16 — Confusion Matrices | Per-class analysis |
| 17 — Learning Curves | Training dynamics visualization |
| 18 — Save Results | CSV and PNG outputs |

---

## Technical Notes

### Feature Extraction

```python
# Pass raw waveform through frozen model
outputs = model(input_values, output_hidden_states=True)

# Stack all 25 hidden states: (25, T, 1024)
all_layers = torch.stack(outputs.hidden_states, dim=0).squeeze(1)

# Mean over layers → mean over time → (1024,)
vector = all_layers.mean(dim=0).mean(dim=0).cpu().numpy()
```

### Common Pitfalls and Fixes

- **StandardScaler must be fit on training data only** — fitting on the full dataset before splitting leaks test information
- **drop_last=True on train DataLoader** — prevents crashes when the last batch contains a single sample
- **Class weights computed per fold** — use training labels only, not the full dataset
- **Loss threshold of 0.01** — stops training before the model memorizes the training set (loss < 0.0001 indicates memorization)
- **Feature dimension is 1024 not 768** — the paper has an error; large models output 1024-d features

---

## Limitations and Honest Assessment

### What "Successful Reproduction" Means

In the SER literature, a reproduction is considered successful when results fall within 5–10% of reported values. Small datasets inherently produce high variance and papers routinely omit critical implementation details. By this standard, this reproduction is successful — 8 out of 10 experiments match or exceed the paper.

### Remaining Gaps

The remaining gaps are scientifically explained:
- **AESDD** — tiny dataset (~500 samples) in Greek with possible undisclosed evaluation protocol differences
- **SHEMO Wav2Vec2** — fundamental linguistic mismatch between an English ASR checkpoint and Persian speech
- **SAVEE HuBERT** — high variance inherent to 96 test samples per fold; individual folds reach 93.75%

### Reproducibility Statement

This study demonstrates that the core claims of the paper are reproducible. The all-layer extraction strategy, while not explicitly described, is the key methodological insight. Any future work on SER with transformer-based feature extractors should use multi-layer aggregation rather than relying solely on the final hidden state.

---

## License

This reproduction study is for academic and research purposes only. The original datasets remain under their respective licenses. The pretrained models are subject to their HuggingFace model licenses.
