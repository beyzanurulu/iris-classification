import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

print("=== 4. FEATURE ENGINEERING & PREPROCESSING ===")
print()

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

print(f"Original features: {len(feature_names)}")

# Create DataFrame
df = pd.DataFrame(X, columns=feature_names)

# 1. Feature Creation - Create 14 new features
print("Creating new features...")

# Area features
df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']

# Ratio features
df['length_ratio'] = df['petal length (cm)'] / df['sepal length (cm)']
df['width_ratio'] = df['petal width (cm)'] / df['sepal width (cm)']

# Difference features
df['length_difference'] = df['sepal length (cm)'] - df['petal length (cm)']
df['width_difference'] = df['sepal width (cm)'] - df['petal width (cm)']

# Combined features
df['total_length'] = df['sepal length (cm)'] + df['petal length (cm)']
df['total_width'] = df['sepal width (cm)'] + df['petal width (cm)']

# Ratio combinations
df['petal_length_to_width_ratio'] = df['petal length (cm)'] / df['petal width (cm)']
df['sepal_length_to_width_ratio'] = df['sepal length (cm)'] / df['sepal width (cm)']

# Statistical features
df['features_mean'] = df[feature_names].mean(axis=1)
df['features_std'] = df[feature_names].std(axis=1)

# Interaction features
df['petal_sepal_length_interaction'] = df['petal length (cm)'] * df['sepal length (cm)']

# Complex combination
df['petal_combination'] = (df['petal length (cm)'] * df['petal width (cm)']) / (df['sepal length (cm)'] + df['sepal width (cm)'])

print(f"Created 14 new features")
print(f"Total features: {len(df.columns)}")
print()

# 2. Feature Importance (using Random Forest)
print("=== FEATURE IMPORTANCE (Top 8) ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(df, y)
feature_importance = pd.DataFrame({
    'feature': df.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(8).iterrows():
    feature_type = "[NEW]" if row['feature'] not in feature_names else "[ORIG]"
    print(f"{i+1}. {feature_type} {row['feature']}: {row['importance']:.3f}")
print()

# 3. Normalization
print("=== NORMALIZATION ===")
scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

X_scaled = scaler_standard.fit_transform(df)
X_normalized = scaler_minmax.fit_transform(df)

print("Applied StandardScaler and MinMaxScaler")
print()

# 4. Feature Selection
print("=== FEATURE SELECTION ===")

# SelectKBest with F-scores
selector_k = SelectKBest(score_func=f_classif, k=8)
X_selected_k = selector_k.fit_transform(X_scaled, y)

print("SelectKBest F-scores (all features):")
scores = selector_k.scores_
feature_scores = list(zip(df.columns, scores))
feature_scores.sort(key=lambda x: x[1], reverse=True)

for i, (feature, score) in enumerate(feature_scores, 1):
    feature_type = "[NEW]" if feature not in feature_names else "[ORIG]"
    selected = "SELECTED" if i <= 8 else "NOT_SELECTED"
    print(f"{i:2d}. {feature_type} {feature:<35} F-score: {score:8.3f} {selected}")
print()

print("SelectKBest (K=8) selected features:")
selected_features_k = df.columns[selector_k.get_support()]
for i, feature in enumerate(selected_features_k, 1):
    feature_type = "[NEW]" if feature not in feature_names else "[ORIG]"
    print(f"  {i}. {feature_type} {feature}")
print()

# RFE with rankings
estimator = RandomForestClassifier(n_estimators=50, random_state=42)
selector_rfe = RFE(estimator, n_features_to_select=6)
X_selected_rfe = selector_rfe.fit_transform(X_scaled, y)

print("RFE Rankings (all features):")
rankings = selector_rfe.ranking_
support = selector_rfe.support_
feature_rankings = list(zip(df.columns, rankings, support))
feature_rankings.sort(key=lambda x: (x[1], x[0]))

for i, (feature, rank, selected) in enumerate(feature_rankings, 1):
    feature_type = "[NEW]" if feature not in feature_names else "[ORIG]"
    status = "SELECTED" if selected else f"RANK {rank:2d}"
    print(f"{i:2d}. {feature_type} {feature:<35} {status}")
print()

print("RFE (6 features) selected features:")
selected_features_rfe = df.columns[selector_rfe.get_support()]
for i, feature in enumerate(selected_features_rfe, 1):
    feature_type = "[NEW]" if feature not in feature_names else "[ORIG]"
    print(f"  {i}. {feature_type} {feature}")
print()

# Summary
print("=== SUMMARY ===")
print(f"Original features: {len(feature_names)}")
print(f"Total features after engineering: {len(df.columns)}")
print(f"Features after SelectKBest: {len(selected_features_k)}")
print(f"Features after RFE: {len(selected_features_rfe)}")
print()

new_features = [f for f in df.columns if f not in feature_names]
successful_new_features = [f for f in new_features if f in list(selected_features_k) + list(selected_features_rfe)]

print(f"Successful new features: {len(successful_new_features)}")
for feature in successful_new_features:
    print(f"  - {feature}")
print()

print("Feature engineering completed. Ready for model development.") 