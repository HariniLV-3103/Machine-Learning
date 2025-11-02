# Linear Models: Classification and Regression

This repository contains two experiments demonstrating the application of **linear models** for classification and regression tasks using Python, NumPy, and related libraries.

---
## 1. Linear Regression — Mobile Price Prediction

### Aim
To implement and critically evaluate **matrix-based linear regression** (closed-form, gradient descent, and L2-regularized) for predicting mobile phone prices, and compare performance with and without standardization.

### Objectives
- Implement ordinary least squares (closed-form) and gradient descent regression.  
- Incorporate L2 regularization (ridge regression).  
- Compare model performance with and without feature standardization.  
- Visualize predicted vs. actual prices.  
- Analyze coefficient-based feature importance.

### Expected Output
- Comparison table for model performance (OLS, Gradient Descent, Ridge).  
- Plots for:
  - Predicted vs. Actual Prices.  
  - Gradient Descent convergence.  
- Insights on the effect of standardization and regularization.

---
## 2. Linear Classification — Bank Note Authentication

### Aim
To fit and evaluate a **Linear Classification Model** (single or multi-layer neural network) for the *Bank Note Authentication dataset*, and analyze the suitability of linear models for binary classification.

### Objectives
- Divide the dataset into training and testing sets.  
- Fit classification models using linear classifiers.  
- Evaluate performance using appropriate metrics (accuracy, confusion matrix, ROC curve).  
- Compare decision boundaries and analyze classification results.  
- Discuss whether linear models are adequate for this dataset.

### Methodology
- The dataset is preprocessed and split into train/test sets.  
- A **linear classifier** (e.g., logistic regression or single-layer perceptron) is trained.  
- Model parameters are learned via gradient descent or closed-form approximation.  
- Performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.  
- Visualization includes decision boundary plots and comparison of model predictions.

### Expected Output
- Classification accuracy and confusion matrix.  
- Visual comparison of predicted vs. actual classes.  
- Discussion on model performance and limitations of linear classifiers for complex data.

---

## ⚙️ Requirements
Install the necessary Python libraries before running:

```bash
pip install numpy pandas matplotlib scikit-learn seaborn
