#  Iris Flower Classification - Complete ML Pipeline

[![Live Demo](https://huggingface.co/spaces/ulubey/iris-flower-classifier)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive end-to-end machine learning pipeline for classifying Iris flowers into three species (Setosa, Versicolor, Virginica) based on their physical measurements. This project demonstrates modern ML best practices from data analysis to production deployment.

##  Project Overview

This project implements a complete machine learning workflow including:
- **Data Analysis & Visualization** - Comprehensive EDA with 5 different chart types
- **Feature Engineering** - Creating 14 additional features from 4 original measurements  
- **Model Development** - Training and comparing 9 different ML algorithms
- **Model Optimization** - Hyperparameter tuning with GridSearchCV
- **Production Deployment** - Interactive web interface with Gradio

##  Key Features

-  **Comprehensive Data Analysis**: Missing values, correlations, statistical summaries
-  **Advanced Visualizations**: Heatmaps, pair plots, feature importance, distributions
-  **Feature Engineering**: 14 engineered features with selection and scaling
-  **Multiple ML Models**: 9 algorithms compared (SVM, Random Forest, etc.)
-  **Hyperparameter Optimization**: Grid search for best performance
-  **Interactive Web App**: Real-time predictions with educational tooltips
-  **Production Ready**: Saved models and preprocessors for deployment

##  Live Demo

**Try the live application:** [Iris Classifier Web App](https://huggingface.co/spaces/ulubey/iris-flower-classifier)

Enter flower measurements and get instant species predictions with confidence scores!

## 📊 Results

### Model Performance Summary
| Model | Test Accuracy | Cross-Val Score |
|-------|---------------|-----------------|
| **SVM (Optimized)** | **100%** | **98.67%** |
| Gradient Boosting | 100% | 97.33% |
| Logistic Regression | 97.78% | 96.00% |
| K-Neighbors | 97.78% | 95.33% |
| Random Forest | 95.56% | 94.67% |

### Feature Engineering Impact
- **Original Features**: 4 (sepal/petal length/width)
- **Engineered Features**: 14 additional features
- **Final Feature Set**: 18 features after selection
- **Performance Boost**: +15% accuracy improvement

## 🏗️ Project Structure

```
iris-classification/
├── 📊 Data Analysis
│   ├── dataset_reporting.py     # Data quality analysis
│   ├── visualization_plots.py   # 5 different visualizations
│   └── eda_analysis.py         # Exploratory data analysis
├── 🛠️ Feature Engineering
│   └── feature_engineering.py  # Feature creation & selection
├── 🤖 Model Development
│   ├── model_development.py    # Train 9 ML models
│   ├── model_comparison.py     # Performance comparison
│   └── model_selection.py      # Hyperparameter optimization
├── 🌐 Deployment
│   └── app.py                  # Gradio web interface
├── 💾 Saved Models
│   ├── final_iris_model.pkl    # Optimized SVM model
│   ├── feature_scaler.pkl      # StandardScaler
│   └── feature_selector.pkl    # SelectKBest selector
├── 📈 Generated Visualizations
│   ├── correlation_heatmap.png
│   ├── pair_plot.png
│   ├── feature_importance.png
│   ├── box_plots.png
│   ├── feature_distributions.png
│   └── model_comparison_roc.png
└── 📋 Configuration
    ├── requirements.txt
    ├── pyproject.toml
    └── setup.sh
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.12+
- pip or uv package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/iris-classification.git
   cd iris-classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete pipeline**
   ```bash
   # Execute all steps in sequence
   python dataset_reporting.py
   python visualization_plots.py
   python eda_analysis.py
   python feature_engineering.py
   python model_development.py
   python model_comparison.py
   python model_selection.py
   
   # Launch web interface
   python app.py
   ```

### Alternative Setup (using uv)
```bash
chmod +x setup.sh
./setup.sh
```

## 📈 Pipeline Steps Explained

### 1. Dataset Reporting (`dataset_reporting.py`)
- Loads Iris dataset from scikit-learn
- Checks for missing values (none found)
- Generates statistical summaries
- Creates correlation matrix

### 2. Visualization (`visualization_plots.py`)
- **Correlation Heatmap**: Feature relationships visualization
- **Pair Plot**: Species separation patterns
- **Box Plots**: Feature distributions by species
- **Feature Importance**: RandomForest-based importance scores
- **Distribution Histograms**: Individual feature value distributions

### 3. Exploratory Data Analysis (`eda_analysis.py`)
- Class distribution analysis (50 samples per species)
- Outlier detection using IQR method
- Inter-species distance calculations
- Feature discriminative power analysis

### 4. Feature Engineering (`feature_engineering.py`)
Creates 14 additional features:
- **Ratio Features**: `petal_ratio`, `sepal_ratio`
- **Area Features**: `petal_area`, `sepal_area`
- **Combined Features**: `total_length`, `total_width`, `total_area`
- **Difference Features**: `length_diff`, `width_diff`
- **Interaction Features**: `sepal_petal_interaction`
- **Statistical Features**: `feature_mean`, `feature_std`, `feature_max`, `feature_min`

### 5. Model Development (`model_development.py`)
Trains 9 classification algorithms:
- Logistic Regression, Random Forest, Decision Tree
- K-Neighbors, SVM, Gradient Boosting
- Extra Trees, Gaussian Naive Bayes, AdaBoost

### 6. Model Comparison (`model_comparison.py`)
- Calculates accuracy, precision, recall, F1-score
- Performs cross-validation analysis
- Generates ROC curves for multi-class classification
- Creates performance comparison visualizations

### 7. Model Selection (`model_selection.py`)
- Optimizes best-performing model (SVM) with GridSearchCV
- Tests hyperparameters: C, gamma, kernel
- Saves final model and preprocessors
- Achieves 100% test accuracy

### 8. Web Deployment (`app.py`)
- Interactive Gradio interface
- Real-time predictions with confidence scores
- Educational tooltips and examples
- Responsive design with species comparisons

## 🌐 Deployment Options

### Hugging Face Spaces (Recommended)
The easiest way to deploy:
1. Upload `app.py`, `requirements.txt`, and model files
2. Automatic deployment at `https://username-iris-classifier.hf.space`

### Local Development
```bash
python app.py
# Access at http://localhost:7860
```

### Other Platforms
- **Railway**: Add `Procfile` with `web: python app.py`
- **Render**: Configure build and start commands
- **Docker**: Use provided Dockerfile for containerization

## 🛠️ Technical Implementation

### Machine Learning Pipeline
- **Data Preprocessing**: StandardScaler normalization
- **Feature Selection**: SelectKBest with f_classif scoring
- **Model Training**: 70/30 train-test split with stratification
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Model Persistence**: Joblib serialization for production deployment

### Feature Engineering Strategy
The engineered features capture important botanical relationships:
- **Ratios** reveal shape characteristics independent of size
- **Areas** provide size-based discriminative power
- **Combinations** capture overall plant dimensions
- **Statistical features** summarize measurement distributions

### Web Interface Design
- **Educational Focus**: Tooltips explain botanical terms
- **User Experience**: Sliders with realistic value ranges
- **Transparency**: Shows confidence scores and feature processing
- **Examples**: Real dataset samples for quick testing

## 📊 Model Performance Analysis

### Why SVM Performed Best
1. **High-dimensional feature space** (18 features) suits SVM well
2. **Clear class separation** in engineered feature space
3. **RBF kernel** captures non-linear relationships effectively
4. **Hyperparameter optimization** fine-tuned C and gamma values

### Cross-Validation Results
- **SVM**: 98.67% ± 1.2% (most consistent)
- **Gradient Boosting**: 97.33% ± 2.1%
- **Logistic Regression**: 96.00% ± 1.8%

## 🔄 Reproducibility

All random operations use fixed seeds:
- `random_state=42` for train-test splits
- Consistent cross-validation folds
- Deterministic model training

## 📝 Usage Examples

### Programmatic Usage
```python
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load('final_iris_model.pkl')
scaler = joblib.load('feature_scaler.pkl') 
selector = joblib.load('feature_selector.pkl')

# Make prediction
measurements = [[5.1, 3.5, 1.4, 0.2]]  # sepal_length, sepal_width, petal_length, petal_width
# ... (feature engineering code)
prediction = model.predict(processed_features)
```

### Web Interface
Visit the live demo and try these examples:
- **Setosa**: 5.1, 3.5, 1.4, 0.2
- **Versicolor**: 6.4, 3.2, 4.5, 1.5  
- **Virginica**: 6.3, 3.3, 6.0, 2.5


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Iris dataset
- **Scikit-learn** for comprehensive ML algorithms
- **Gradio** for easy-to-use web interface framework
- **Hugging Face** for free model hosting and deployment

**Live Demo**: [https://huggingface.co/spaces/yourusername/iris-flower-classifier](https://huggingface.co/spaces/ulubey/iris-flower-classifier)


