import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("=== 3. EXPLORATORY DATA ANALYSIS ===")
print()

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = y

# 1. Class Distribution
print("--- Class Distribution ---")
class_counts = df['species'].value_counts().sort_index()
for i, count in enumerate(class_counts):
    print(f"{target_names[i]}: {count} samples")
print()

# 2. Outlier Detection using IQR
print("--- Outlier Detection ---")
outlier_counts = {}
for col in feature_names:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_counts[col] = len(outliers)
    print(f"{col}: {len(outliers)} outliers")
print()

# 3. Inter-species Comparison
print("--- Inter-species Feature Means ---")
species_means = df.groupby('species')[feature_names].mean()
for i, species in enumerate(target_names):
    print(f"{species}:")
    for feature in feature_names:
        print(f"  {feature}: {species_means.iloc[i][feature]:.2f}")
    print()

# 4. Key Insights
print("--- Key EDA Insights ---")
print("1. Balanced dataset: Each species has exactly 50 samples")
print("2. Petal measurements show stronger species separation than sepal")
print("3. Setosa is clearly distinguishable from others")
print("4. Versicolor and Virginica have overlapping characteristics")
print("5. Low outlier presence indicates good data quality")
print()

print("=== Step 3 (EDA) completed! ===") 