# Week 04 Day 22 — ML Assignment (PM Session)

**PG Diploma · AI-ML & Agentic AI Engineering · IIT Gandhinagar**
Gitlink: https://github.com/shubhamrangdal2001-hash/Assignment_Day22PM.git
## Overview

This assignment covers:
- Types of Machine Learning (Supervised, Unsupervised, Reinforcement Learning)
- Regression using Linear Regression
- Classification using Logistic Regression
- Data handling with Pandas
- Feature selection and correlation analysis

---

## Dataset

A synthetic housing dataset (`housing.csv`) with 1000 rows and 10 columns:

| Column | Type | Description |
|---|---|---|
| MedInc | float | Median income of block |
| HouseAge | float | Median house age |
| AveRooms | float | Average number of rooms |
| AveBedrms | float | Average number of bedrooms |
| Population | float | Block population |
| AveOccup | float | Average occupancy |
| Latitude | float | Geographic latitude |
| Longitude | float | Geographic longitude |
| MedHouseVal | float | **Regression target** — median house value |
| PriceCategory | string | **Classification target** — Low / Medium / High |

---

## Project Structure

```
.
├── data_setup.py               # Generates housing.csv
├── part_a_ml_analysis.py       # Part A: ML type, data cleaning, regression, classification
├── part_b_feature_analysis.py  # Part B: Feature correlation and selection
├── part_c_interview.py         # Part C: Pandas filter + aggregate coding
├── part_d_ai_task.py           # Part D: AI-augmented code example
├── housing.csv                 # Generated dataset (run data_setup.py first)
└── README.md
```

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn
```

### 2. Generate the dataset

```bash
python data_setup.py
```

This creates `housing.csv` in the current directory.

### 3. Run each part

```bash
python part_a_ml_analysis.py
python part_b_feature_analysis.py
python part_c_interview.py
python part_d_ai_task.py
```

---

## Results Summary

| Task | Model | Metric | Result |
|---|---|---|---|
| Regression | Linear Regression | MSE | 0.2389 |
| Classification | Logistic Regression | Accuracy | 93.0% |
| Feature selection (regression) | Linear Regression | MSE | 0.2423 |
| Feature selection (classification) | Logistic Regression | Accuracy | 93.0% |

---

## Key Findings

- `MedInc` (median income) is the dominant predictor of house value with a Pearson correlation of **0.95**.
- Feature selection confirmed that most other features add noise rather than signal.
- Logistic Regression with StandardScaler gave a clean 93% accuracy without overfitting.

---

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
