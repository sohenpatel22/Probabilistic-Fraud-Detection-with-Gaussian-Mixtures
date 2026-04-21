# Dataset

This project uses the **Credit Card Fraud Detection Dataset**

## Source

* Dataset: Credit Card Fraud Detection
* Source: Kaggle
* Link: https://raw.githubusercontent.com/chyr98/Dataset/main/creditcard.csv

## Description

The dataset contains transactions made by European cardholders in September 2013.

* Total transactions: ~284,000
* Fraud cases: ~0.17% (highly imbalanced)
* Features:

  * `V1–V28`: PCA-transformed numerical features
  * `Time`, `Amount`
  * `Class`: Target variable (1 = Fraud, 0 = Normal)

## Usage in this Project

For this project:

* A subset of features (`V11–V20`) is used for modeling
* The task is treated as an **anomaly detection problem**

## File Location

Place the dataset file as:

```
data/creditcard.csv
```
