import pandas as pd
import numpy as np
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
print("=== 1. DATASET REPORTING ===")

print()

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
feature_names = iris.feature_names

# 1. Missing Values Check
df = pd.DataFrame(X, columns=feature_names)
missing = df.isnull().sum()
print("--- Missing Values Check ---")
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found.")
print()

# 2. Statistical Summary
print("--- Statistical Summary ---")
summary = df.describe()
print(summary)
print()

# 3. Correlation Matrix
print("--- Correlation Matrix ---")
corr = df.corr()
print(corr)
print()

print("Dataset Reporting completed!")
print("Missing values check: Successful")
print("Statistical summary: Successful")
print("Correlation matrix: Successful")
print()
print("=== Step 1 (Dataset Reporting) completed! ===")