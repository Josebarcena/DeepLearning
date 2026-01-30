# ECG Anomaly Detection (PTB-XL) — ML & Deep Learning

Automatic ECG anomaly detection system using classic machine learning models (**SVM**, **kNN**, **Decision Trees**) and deep learning (**CNNs**).  
Trained and validated on the **PTB-XL** dataset, achieving **>90% F1-score and precision** for detecting cardiac anomalies, demonstrating potential for **clinical decision support**.

---

## Project Overview

This project focuses on building an end-to-end pipeline for ECG anomaly detection:
- ECG signal preprocessing and feature handling
- Training and evaluation of multiple classification approaches
- Reproducible experimentation (scripts + results export)

Models implemented:
- **Support Vector Machines (SVM)**
- **k-Nearest Neighbors (kNN)**
- **Decision Trees**
- **Convolutional Neural Networks (CNNs)**

Dataset:
- **PTB-XL**: a large publicly available ECG dataset containing 12-lead ECG recordings and diagnostic labels.

> ⚠️ Disclaimer: This repository is for research/educational purposes only and is **not** a medical device. It must not be used for real clinical decisions without proper validation and regulatory approval.

---

## Repository Structure

Based on the current repo layout:
## Repository Structure

```text
.
├── BDD/                 # Data storage / dataset-related files (raw or processed)
├── Scripts/             # Training, preprocessing, evaluation scripts
├── aa/
│   └── Funciones/       # Utility functions/helpers used across scripts
├── e0106/               # Experiment folder (run/config/results for a specific experiment)
├── e0208/               # Experiment folder (run/config/results for a specific experiment)
├── results/             # Consolidated outputs: metrics, figures, predictions, logs
└── e0208-2(1).xlsx      # Spreadsheet report (metrics/experiments summary)
