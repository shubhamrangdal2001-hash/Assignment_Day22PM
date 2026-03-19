"""
part_c_interview.py
-------------------
Part C: Interview Ready
Q2 (Coding) — Using Pandas:
  - Filter dataset where target variable meets a condition
  - Compute average of a feature for that subset
"""

import pandas as pd

# Load dataset
df = pd.read_csv('housing.csv')
df['HouseAge'] = df['HouseAge'].fillna(df['HouseAge'].median())
df['AveRooms'] = df['AveRooms'].fillna(df['AveRooms'].median())
df = df.dropna(subset=['MedHouseVal'])

# ─── Filter: houses where MedHouseVal > 3.0 (high-value homes) ──────────────
high_value = df[df['MedHouseVal'] > 3.0]
print(f"Total rows          : {len(df)}")
print(f"High-value homes    : {len(high_value)}")
print()

# ─── Compute average MedInc for that subset ──────────────────────────────────
avg_income = high_value['MedInc'].mean()
print(f"Average Median Income for high-value homes (MedHouseVal > 3.0): {avg_income:.4f}")
print()

# ─── One more example: low-value homes ───────────────────────────────────────
low_value = df[df['MedHouseVal'] <= 1.8]
print(f"Low-value homes     : {len(low_value)}")
avg_rooms_low = low_value['AveRooms'].mean()
print(f"Average Rooms for low-value homes (MedHouseVal <= 1.8): {avg_rooms_low:.4f}")
