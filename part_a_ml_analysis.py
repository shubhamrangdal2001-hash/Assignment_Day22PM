"""
part_a_ml_analysis.py
---------------------
Part A: Concept Application
- Identify ML Problem Type
- Data Handling with Pandas
- Regression Task (predict MedHouseVal)
- Classification Task (predict PriceCategory)
- Comparison of Regression vs Classification
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── 1. Load dataset ────────────────────────────────────────────────────────
df = pd.read_csv('housing.csv')
print("=== Dataset Info ===")
print(f"Shape: {df.shape}")
print(df.dtypes)
print()

# ─── 2. Identify ML Problem Type ────────────────────────────────────────────
# The dataset has labelled target columns (MedHouseVal, PriceCategory)
# → Supervised Learning
# MedHouseVal is continuous → Regression
# PriceCategory is categorical → Classification
print("=== ML Problem Type ===")
print("Learning Type  : Supervised Learning")
print("Regression     : Predict MedHouseVal (continuous house value)")
print("Classification : Predict PriceCategory (Low / Medium / High)")
print()

# ─── 3. Data Handling with Pandas ───────────────────────────────────────────
print("=== Missing Values Before Cleaning ===")
print(df.isnull().sum())
print()

# Fill missing numerical values with column median
df['HouseAge'] = df['HouseAge'].fillna(df['HouseAge'].median())
df['AveRooms'] = df['AveRooms'].fillna(df['AveRooms'].median())

# Drop rows where the target itself is missing
df = df.dropna(subset=['MedHouseVal', 'PriceCategory'])

print("=== Missing Values After Cleaning ===")
print(df.isnull().sum())
print(f"\nRows after cleaning: {len(df)}")
print()

# Select relevant features
features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
df_clean = df[features + ['MedHouseVal', 'PriceCategory']].copy()

# ─── 4. Regression Task ─────────────────────────────────────────────────────
print("=== Regression Task ===")
X_reg = df_clean[features]
y_reg = df_clean['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_reg = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_reg)
print(f"Model          : Linear Regression")
print(f"Test MSE       : {mse:.4f}")
print(f"Sample Actual  : {y_test.values[:5]}")
print(f"Sample Predicted: {y_pred_reg[:5].round(4)}")
print()

# ─── 5. Classification Task ──────────────────────────────────────────────────
print("=== Classification Task ===")
X_clf = df_clean[features]
y_clf = df_clean['PriceCategory'].astype(str)

le = LabelEncoder()
y_enc = le.fit_transform(y_clf)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_clf, y_enc, test_size=0.2, random_state=42
)

# Scale features for logistic regression
scaler = StandardScaler()
X_train_c = scaler.fit_transform(X_train_c)
X_test_c  = scaler.transform(X_test_c)

clf_model = LogisticRegression(max_iter=1000, random_state=42)
clf_model.fit(X_train_c, y_train_c)
y_pred_clf = clf_model.predict(X_test_c)

acc = accuracy_score(y_test_c, y_pred_clf)
print(f"Model          : Logistic Regression")
print(f"Test Accuracy  : {acc:.4f}")
print(f"Classes        : {le.classes_}")
print()

# ─── 6. Comparison ───────────────────────────────────────────────────────────
print("=== Regression vs Classification Comparison ===")
comparison = {
    "Aspect": ["Output Type", "Use Case", "Evaluation Metric"],
    "Regression": [
        "Continuous number (e.g. 3.45)",
        "Predict house price, temperature, salary",
        "MSE, RMSE, R² Score"
    ],
    "Classification": [
        "Discrete label (e.g. Low / Medium / High)",
        "Spam detection, disease diagnosis, category prediction",
        "Accuracy, Precision, Recall, F1-Score"
    ]
}
comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))
