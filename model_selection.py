import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print('=== 7. MODEL SELECTION ===')
print()

# Data preparation (from previous steps)
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Feature Engineering
df = pd.DataFrame(X, columns=iris.feature_names)
df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
df['petal_sepal_ratio'] = df['petal_area'] / df['sepal_area']
df['total_area'] = df['petal_area'] + df['sepal_area']
df['length_ratio'] = df['petal length (cm)'] / df['sepal length (cm)']
df['width_ratio'] = df['petal width (cm)'] / df['sepal width (cm)']

# Feature Selection and Normalization
selector = SelectKBest(score_func=f_classif, k=6)
X_selected = selector.fit_transform(df.values, y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

print('SVM Hyperparameter Optimization...')

# SVM hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

# GridSearchCV
svm = SVC(random_state=42, probability=True)
grid_search = GridSearchCV(
    svm, 
    param_grid, 
    cv=5, 
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

# Run grid search
grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best CV score: {grid_search.best_score_:.4f}')

# Final test with best model
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
y_pred = best_model.predict(X_test)

print(f'Final test accuracy: {test_accuracy:.4f}')
print()

# Detailed performance report
print('--- DETAILED PERFORMANCE REPORT ---')
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
print('--- CONFUSION MATRIX ---')
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Performance improvement
baseline_svm = SVC(random_state=42).fit(X_train, y_train)
baseline_accuracy = baseline_svm.score(X_test, y_test)
improvement = test_accuracy - baseline_accuracy

print(f'\nIMPROVEMENT ANALYSIS:')
print(f'   Baseline SVM: {baseline_accuracy:.4f}')
print(f'   Optimized SVM: {test_accuracy:.4f}')
print(f'   Improvement: {improvement:.4f} ({improvement*100:.2f}%)')

# Save final model
print('\nSaving model...')
joblib.dump(best_model, 'final_iris_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')
joblib.dump(selector, 'feature_selector.pkl')

# Model information
model_info = {
    'model': 'Support Vector Machine',
    'best_params': grid_search.best_params_,
    'test_accuracy': test_accuracy,
    'cv_score': grid_search.best_score_,
    'feature_count': X_selected.shape[1],
    'created_features': ['petal_area', 'sepal_area', 'petal_sepal_ratio', 'total_area', 'length_ratio', 'width_ratio']
}

print('\nFINAL MODEL INFORMATION:')
for key, value in model_info.items():
    print(f'   {key}: {value}')

print('\nModel Selection completed!')
print('SVM hyperparameter optimization')
print('Final model ready')
print('Saved files:')
print('   - final_iris_model.pkl')
print('   - feature_scaler.pkl') 
print('   - feature_selector.pkl') 