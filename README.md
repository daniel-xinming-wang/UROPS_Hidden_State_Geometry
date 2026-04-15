# UROPS Hidden State Geometry

This repository contains the code for the UROP project on the **geometric structure of hidden states in large vision-language models**.

The goal of this project is to understand how visual information influences internal representations by comparing hidden states under **image-present** and **image-absent** conditions while keeping textual prompts fixed.

We analyze representation geometry using PCA-based methods, including:

- intrinsic dimensionality
- cross-condition projection overlap
- principal subspace alignment (principal angles)

---

# Overview

The pipeline consists of two main stages:

1. Extract hidden states from multimodal models
2. Analyze representation geometry using PCA-based metrics

---

# Pipeline

## Step 1: Hidden state extraction

Hidden states are generated using:

MMMU:
```scripts/embed_mmmu.sh
```

MathVision:
```bash scripts/embed.sh
```

## Step 2: Representation geometry analysis

After hidden states are generated, run:

MMMU:
```python AnalyzeRank.py
```

MathVision:
```
python AnalyzeRank_mathv.py
```

These scripts load the saved hidden states and compute geometric statistics across layers, including:

- intrinsic dimensionality (number of principal components needed to explain 90% variance)
- projection overlap (how much variance of image-present representations lies in text-derived principal subspaces)
- principal angle similarity (alignment between dominant subspaces)

The scripts also generate plots showing how representation geometry changes across transformer depth.
