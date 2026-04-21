# Credit Card Fraud Detection using Gaussian Mixture Models

## Overview

This project focuses on detecting fraudulent credit card transactions using **unsupervised anomaly detection techniques**, specifically **Gaussian Mixture Models (GMMs)**.

Unlike traditional classification problems, fraud detection is highly challenging due to **extreme class imbalance**, where fraudulent transactions make up less than 1% of the data. This project explores how probabilistic models can identify such rare events effectively.

The goal was not just to train a model, but to:

* Understand anomaly detection in imbalanced datasets
* Compare different modeling strategies using GMM
* Optimize decision thresholds for real-world performance
* Evaluate models using appropriate metrics like **F1-score and PR-AUC**

---

## Problem Statement

Credit card fraud detection is a real-world problem where:

* Fraudulent transactions are extremely rare (~0.17%)
* Traditional accuracy-based models fail due to imbalance
* Missing fraud cases can have serious financial consequences

This project approaches the problem as an **anomaly detection task**, where fraudulent transactions are treated as deviations from normal behavior.

---

## Dataset

* Source: European cardholders transaction dataset (September 2013)
* Total samples: ~284,000
* Fraud cases: ~0.17%

Features:

* `V1–V28`: PCA-transformed numerical features
* `Class`: Target variable (1 = Fraud, 0 = Normal)

For this project, a subset of features (`V11–V20`) was used to analyze model behavior.

---

## Key Challenges

* **Severe class imbalance**
* Overlapping feature distributions
* Selecting meaningful evaluation metrics
* Choosing appropriate threshold for classification

---

## Approach

### 1. Exploratory Data Analysis

* Compared feature distributions for fraud vs normal transactions
* Identified features with strong separability

### 2. Single Feature Gaussian Models

* Modeled each feature using a single Gaussian
* Evaluated performance using ROC-AUC and PR-AUC
* Identified top discriminative features

### 3. Threshold Optimization

* Designed an efficient method to find optimal threshold
* Optimized for **F1-score** on validation set
* Avoided brute-force search

### 4. Multivariate Gaussian Models

* Extended to multiple features
* Used **Gaussian Mixture Models (GMMs)** to capture data distribution
* Compared different feature combinations

### 5. Two-Model Approach (Normal vs Fraud)

* Trained separate GMMs for:

  * Normal transactions
  * Fraud transactions
* Used **log-likelihood difference** as anomaly score

### 6. Model Selection

* Experimented with:

  * Number of features
  * Number of Gaussian components
* Selected best model based on validation F1-score

---

## Results

### Best Model Configuration

* Features: `V11, V12, V14, V16, V17`
* Normal Model: 2 Gaussian components
* Fraud Model: 3 Gaussian components

### Test Set Performance

* **F1 Score:** 0.736
* **Precision:** 0.757
* **Recall:** 0.716

---

## Key Insights

* **PR-AUC is more informative than ROC-AUC** in highly imbalanced datasets
* Modeling **fraud and normal distributions separately** improves performance
* Fraud transactions exhibit **higher variability**, requiring multiple components
* Using too many features can lead to **diminishing returns or overfitting**
* Threshold selection plays a **critical role** in anomaly detection systems

---

## Project Structure

```
fraud-detection-gmm/
│
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── evaluation/
│   └── utils/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── tables/
├── README.md
├── requirements.txt
```

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run notebooks:

```bash
jupyter notebook
```

3. (Optional) Run scripts:

```bash
python -m src.models.gmm_detector
```

---

## Limitations

* PCA-transformed features limit interpretability
* Fraud class is extremely small → sensitive to threshold choice
* Unsupervised methods may underperform compared to supervised models when labels are available

---

## Future Improvements

* Compare with models like Isolation Forest and One-Class SVM
* Use full feature set instead of subset
* Explore deep learning-based anomaly detection
* Incorporate cost-sensitive learning

---

## References

* Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*
* Chandola et al. (2009). *Anomaly Detection: A Survey*
* Scikit-learn Documentation
