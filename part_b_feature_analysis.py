"""
part_b_feature_analysis.py
--------------------------
Part B: Stretch Problem
- Correlation between numerical features
- Identify important features
- Improve model by feature selection
- Explain how feature selection impacts models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── Load & clean data ───────────────────────────────────────────────────────
df = pd.read_csv('housing.csv')
df['HouseAge'] = df['HouseAge'].fillna(df['HouseAge'].median())
df['AveRooms'] = df['AveRooms'].fillna(df['AveRooms'].median())
df = df.dropna(subset=['MedHouseVal', 'PriceCategory'])

features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

# ─── 1. Correlation Analysis ─────────────────────────────────────────────────
print("=== Correlation Matrix (numerical features vs MedHouseVal) ===")
corr = df[features + ['MedHouseVal']].corr()
print(corr['MedHouseVal'].sort_values(ascending=False).round(4))
print()

# ─── 2. Identify Important Features ─────────────────────────────────────────
# Features with |correlation| > 0.1 with target are considered relevant
threshold = 0.1
important_feats = corr['MedHouseVal'].drop('MedHouseVal')
important_feats = important_feats[important_feats.abs() > threshold].index.tolist()
weak_feats = [f for f in features if f not in important_feats]

print(f"Important features (|corr| > {threshold}): {important_feats}")
print(f"Weak/irrelevant features              : {weak_feats}")
print()

# ─── 3. Regression: All features vs Selected features ────────────────────────
print("=== Regression: All Features vs Selected Features ===")
y_reg = df['MedHouseVal']

for label, feat_set in [("All Features", features), ("Selected Features", important_feats)]:
    X = df[feat_set]
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"  {label:25s} → MSE: {mse:.4f}")
print()

# ─── 4. Classification: All features vs Selected features ────────────────────
print("=== Classification: All Features vs Selected Features ===")
le = LabelEncoder()
y_clf = le.fit_transform(df['PriceCategory'].astype(str))

for label, feat_set in [("All Features", features), ("Selected Features", important_feats)]:
    X = df[feat_set]
    X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  {label:25s} → Accuracy: {acc:.4f}")
print()

# ─── 5. Explanation ──────────────────────────────────────────────────────────
print("=== How Feature Selection Impacts Models ===")
print("""
Regression:
  - Removing features with near-zero correlation reduces noise.
  - Fewer irrelevant features → simpler model → often lower MSE.
  - Too many weak features can cause overfitting on train data.

Classification:
  - Weak features add noise that confuses the classifier.
  - Keeping only correlated features helps the model draw cleaner boundaries.
  - Generally leads to better accuracy and faster training.
""")
