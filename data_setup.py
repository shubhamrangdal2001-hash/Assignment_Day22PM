"""
data_setup.py
-------------
Generates the housing dataset used for this assignment.
Run this first before the other scripts.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
n = 1000

# Create features similar to California housing data
df = pd.DataFrame({
    'MedInc':     np.random.uniform(1.0, 15.0, n).round(4),
    'HouseAge':   np.random.randint(1, 52, n).astype(float),
    'AveRooms':   np.random.uniform(2.0, 10.0, n).round(4),
    'AveBedrms':  np.random.uniform(0.8, 3.0, n).round(4),
    'Population': np.random.randint(100, 5000, n).astype(float),
    'AveOccup':   np.random.uniform(1.5, 6.0, n).round(4),
    'Latitude':   np.random.uniform(32.0, 42.0, n).round(4),
    'Longitude':  np.random.uniform(-124.0, -114.0, n).round(4),
})

# Introduce some missing values to make it realistic
df.loc[np.random.choice(n, 20, replace=False), 'HouseAge'] = np.nan
df.loc[np.random.choice(n, 15, replace=False), 'AveRooms'] = np.nan

# Regression target: median house value (continuous)
noise = np.random.normal(0, 0.3, n)
df['MedHouseVal'] = (
    0.5 * df['MedInc'] + 0.01 * df['AveRooms'].fillna(df['AveRooms'].median())
    - 0.005 * df['HouseAge'].fillna(df['HouseAge'].median())
    + noise
).clip(0.5, 5.0).round(4)

# Classification target: price category (categorical)
df['PriceCategory'] = pd.cut(
    df['MedHouseVal'],
    bins=[0, 1.8, 3.2, 5.1],
    labels=['Low', 'Medium', 'High']
)

df.to_csv('housing.csv', index=False)
print("Dataset saved as housing.csv")
print(f"Shape: {df.shape}")
print(df.head())
