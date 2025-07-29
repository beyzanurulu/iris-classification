import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=== 5. MODEL DEVELOPMENT ===")
print()

# Load and prepare data
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Create enhanced features (same as feature_engineering.py)
df = pd.DataFrame(X, columns=feature_names)

# Add new features
df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
df['length_ratio'] = df['petal length (cm)'] / df['sepal length (cm)']
df['width_ratio'] = df['petal width (cm)'] / df['sepal width (cm)']
df['length_difference'] = df['sepal length (cm)'] - df['petal length (cm)']
df['width_difference'] = df['sepal width (cm)'] - df['petal width (cm)']
df['total_length'] = df['sepal length (cm)'] + df['petal length (cm)']
df['total_width'] = df['sepal width (cm)'] + df['petal width (cm)']
df['petal_length_to_width_ratio'] = df['petal length (cm)'] / df['petal width (cm)']
df['sepal_length_to_width_ratio'] = df['sepal length (cm)'] / df['sepal width (cm)']
df['features_mean'] = df[feature_names].mean(axis=1)
df['features_std'] = df[feature_names].std(axis=1)
df['petal_sepal_length_interaction'] = df['petal length (cm)'] * df['sepal length (cm)']
df['petal_combination'] = (df['petal length (cm)'] * df['petal width (cm)']) / (df['sepal length (cm)'] + df['sepal width (cm)'])

print(f"Total features: {len(df.columns)}")

# Feature selection and preprocessing
selector = SelectKBest(score_func=f_classif, k=8)
X_selected = selector.fit_transform(df, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print()

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Neighbors': KNeighborsClassifier(n_neighbors=3),
    'SVM': SVC(random_state=42, probability=True),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
    'Gaussian NB': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(random_state=42)
}

print("=== TRAINING 9 CLASSIFICATION MODELS ===")
print()

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"  Accuracy: {accuracy:.4f}")

print()
print("=== MODEL TRAINING RESULTS ===")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

for i, (name, accuracy) in enumerate(sorted_results, 1):
    print(f"{i}. {name}: {accuracy:.4f}")

print()
print("All models trained successfully!")
print("Ready for model comparison and evaluation.")
print()
print("=== Step 5 (Model Development) completed! ===") 