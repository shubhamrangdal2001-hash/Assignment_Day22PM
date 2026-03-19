"""
part_d_ai_task.py
-----------------
Part D: AI-Augmented Task
Prompt used:
  "Explain types of machine learning, regression, and classification
   with Python examples using Pandas."

This file contains the AI-generated code (cleaned and validated by me).
I verified that:
  1. The ML type descriptions are correct
  2. All code runs without errors
  3. The examples are meaningful and match real use cases
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# ────────────────────────────────────────────────────────────────────────────
# TYPES OF MACHINE LEARNING (from AI output — verified correct)
# ────────────────────────────────────────────────────────────────────────────
# 1. Supervised Learning   – learns from labelled data (input → output pairs)
#    Examples: email spam detection, house price prediction
#
# 2. Unsupervised Learning – finds hidden patterns in unlabelled data
#    Examples: customer segmentation, anomaly detection
#
# 3. Reinforcement Learning – agent learns by trial and error using rewards
#    Examples: game playing (AlphaGo), robot navigation
# ────────────────────────────────────────────────────────────────────────────

# ─── Small demo dataset ──────────────────────────────────────────────────────
np.random.seed(0)
data = pd.DataFrame({
    'area_sqft':   np.random.randint(500, 3000, 100),
    'num_rooms':   np.random.randint(1, 6, 100),
    'price_lakhs': np.random.uniform(20, 150, 100).round(2)
})
data['category'] = data['price_lakhs'].apply(
    lambda x: 'Affordable' if x < 60 else ('Mid-range' if x < 100 else 'Luxury')
)

print("=== Sample Dataset ===")
print(data.head())
print()

# ─── Regression Example ──────────────────────────────────────────────────────
print("=== Regression: Predict House Price ===")
X = data[['area_sqft', 'num_rooms']]
y = data['price_lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
reg = LinearRegression()
reg.fit(X_train, y_train)
mse = mean_squared_error(y_test, reg.predict(X_test))
print(f"MSE: {mse:.2f}")
print()

# ─── Classification Example ──────────────────────────────────────────────────
print("=== Classification: Predict House Category ===")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_cat = le.fit_transform(data['category'])

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=0)
clf = LogisticRegression(max_iter=300)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"Accuracy: {acc:.2f}")
print(f"Classes : {le.classes_}")

# ─── Evaluation ──────────────────────────────────────────────────────────────
print("""
=== AI Output Evaluation ===
Q: Are concepts correctly explained?
A: Yes. The AI correctly described supervised, unsupervised, and reinforcement
   learning with proper examples. Regression and classification were distinguished
   by output type (continuous vs categorical).

Q: Is the code runnable and meaningful?
A: Yes, after minor cleanup (fixed a deprecated fillna call and added
   LabelEncoder import). The examples demonstrate real workflows.
""")
