# Music Style Transfer: Chinese Folk → Jazz

Final project for **CIS 5200 (Machine Learning), University of Pennsylvania, Fall 2025**.
Team "Survival": Xiao Feng, Lebin Hu, Yiqun Su.

📄 **[Read the full report (PDF)](report/CIS5200_Final_Report.pdf)**

## Overview

Can a Chinese folk clip be turned into jazz by editing its features, without any paired examples of the same piece in both styles?

We tested the hypothesis that different musical styles share a common structure, plus style-specific components that can be separated and swapped. We compared five approaches, from a simple mean-shift baseline to a gradient-based CNN-LSTM transformation.

## Data

- 119 jazz songs and 119 Chinese folk songs from a public music platform (Pixabay), cut into 10-second clips
- **4,273 samples** in total (2,066 jazz / 2,207 folk), roughly balanced
- Each clip converted with LibROSA into an **80-band log-Mel spectrogram** (80 × 616 frames)

The audio files and the `.pt` feature files are not included in this repo.

## Methods

| Method | Idea |
|---|---|
| **Baseline (mean shift)** | Add the difference between the jazz mean and the folk mean to each folk sample |
| **PCA** | Keep the top-2 principal components as "content" and replace the rest with the average jazz style vector |
| **KNN** | After the mean shift, take the average of the 5 nearest real jazz samples |
| **PCA + GMM** (ours) | Cut spectrograms into overlapping patches, project them into a shared 64-dim PCA space, fit an 8-component GMM per style, match clusters across styles (Hungarian algorithm), shift each patch toward its matched jazz cluster, then rebuild with overlap-add |
| **CNN-LSTM-Attention** | Train a jazz/folk classifier (99.5% test accuracy), freeze it, then run gradient descent on the input spectrogram, trading off a style loss against a content loss |

We chose the number of PCA components from the reconstruction-error elbow and the number of GMM components with BIC.

## Results

We measured **conversion precision**: the share of converted folk clips that an independently trained neural classifier labels as jazz.

| Method | Conversion precision |
|---|---|
| **PCA + GMM** | **97.0%** |
| CNN-LSTM | 93.4% |
| KNN | 91.5% |
| PCA only | 13.6% |
| Baseline (unchanged reconstruction) | 8.7% |

### What we learned

PCA+GMM scored highest on the metric, but in listening tests the converted audio often had artifacts or sounded almost unchanged. Three takeaways:

- A classifier-based metric measures **distribution alignment**, not perceived style.
- Mel spectrograms encode pitch, timbre and rhythm in nonlinear ways, while linear and Gaussian methods (PCA, GMM) cannot cleanly separate style from content in them.
- Modern music generators (for example, systems built on neural audio codecs such as EnCodec, combined with Transformer or diffusion models) follow the same encode → transform → decode workflow. They succeed because their latent spaces capture nonlinear structure.

## Repository structure

```
notebooks/
  01_mel_feature_extraction.ipynb        # audio → 80-band log-Mel tensors
  02_baseline_knn_pca.ipynb              # baseline, PCA and KNN methods
  03_pca_gmm.ipynb                       # PCA + GMM method, with PCA/BIC selection plots
  04_cnn_lstm_classifier_training.ipynb  # CNN-LSTM-Attention jazz/folk classifier
  05_cnn_lstm_style_transfer.ipynb       # gradient-based transformation using the frozen classifier
report/
  CIS5200_Final_Report.pdf
```

The notebooks were run in Google Colab and on local machines, so file paths point to the original environments. Change the dataset and model paths before rerunning.

**Tech:** Python, PyTorch, scikit-learn, LibROSA, NumPy, SciPy, Matplotlib
