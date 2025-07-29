import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

print('=== 6. MODEL COMPARISON ===')
print()

# Use prepared data from Iris dataset (from main.ipynb)
iris = datasets.load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

# Feature Engineering and Selection (from previous steps)
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

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# Model definitions
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(random_state=42, probability=True),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42, n_estimators=100),
    'Naive Bayes': GaussianNB(),
    'AdaBoost': AdaBoostClassifier(random_state=42, algorithm='SAMME')
}

# Model performance comparison
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    
    results.append({
        'Model': name,
        'Test Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })
    
    print(f'{name}: Accuracy={accuracy:.4f}, CV={cv_scores.mean():.4f}')

# Best model
results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)
print(f'\nBEST MODEL: {results_df.iloc[0]["Model"]} (Accuracy: {results_df.iloc[0]["Test Accuracy"]:.4f})')

# ROC Curves (One-vs-Rest)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray'])

plt.figure(figsize=(12, 8))
auc_scores = {}

for (name, model), color in zip(models.items(), colors):
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)
    
    # Macro-average ROC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Macro-average calculation
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 3
    
    macro_auc = auc(all_fpr, mean_tpr)
    auc_scores[name] = macro_auc
    
    plt.plot(all_fpr, mean_tpr, color=color, linewidth=2,
             label=f'{name} (AUC = {macro_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Model Comparison - ROC Curves (One-vs-Rest)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison_roc.png', dpi=300, bbox_inches='tight')
plt.show()

# Performance table
print('\n--- PERFORMANCE RANKING ---')
print(results_df.round(4))

print('\nModel Comparison completed!')
print('9 models compared')
print('ROC curves generated')
print('Chart saved: model_comparison_roc.png') 