# Experiment 4: Ensemble Prediction and Decision Tree Model Evaluation

## Objective
To build and evaluate classifiers such as **Decision Tree, AdaBoost, Gradient Boosting, XGBoost, Random Forest, and Stacking Models (SVM + Naïve Bayes + Decision Tree)**. Performance is measured through **5-Fold Cross-Validation** and **hyperparameter tuning**.

---

## Dataset
- **Source:** [Wisconsin Diagnostic Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)  
- **Size:** 569 samples, 30 numerical features  
- **Target Labels:** Binary (Malignant or Benign)  
- **Description:** Features represent cell nuclei characteristics from digitized images.

---

## Steps for Implementation
1. **Data Preprocessing**
   - Encode labels (Malignant = 1, Benign = 0)
   - Handle missing values
   - Standardize features (Z-score normalization)

2. **Exploratory Data Analysis (EDA)**
   - Class balance visualization (Malignant vs Benign)
   - Correlation heatmap of features
   - Distribution plots of important features

3. **Train-Test Split**
   - 80% Training, 20% Testing

4. **Model Training**
   - Decision Tree
   - AdaBoost
   - Gradient Boosting
   - XGBoost
   - Random Forest
   - Stacking Classifier (Base: SVM, Naïve Bayes, Decision Tree; Meta: Logistic Regression)

5. **Hyperparameter Tuning**
   - Use `GridSearchCV` / `RandomizedSearchCV`
   - Optimize important hyperparameters:
     - Decision Tree: `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`
     - AdaBoost: `n_estimators`, `learning_rate`, `base_estimator`
     - Gradient Boosting: `n_estimators`, `learning_rate`, `max_depth`, `subsample`
     - XGBoost: `n_estimators`, `learning_rate`, `max_depth`, `gamma`, `subsample`, `colsample_bytree`
     - Random Forest: `n_estimators`, `max_depth`, `criterion`, `max_features`, `min_samples_split`
     - Stacking: Base models + final estimator (Logistic Regression)

6. **Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - ROC Curve & AUC
   - Confusion Matrix
   - 5-Fold Cross-Validation Scores

7. **Comparison of Models**
   - Record best hyperparameters
   - Compare mean CV performance
   - Discuss advantages/disadvantages of each ensemble method

---

## Expected Results
- **Decision Tree**: Simple baseline model, prone to overfitting.  
- **Random Forest**: Strong accuracy, handles feature importance well.  
- **AdaBoost & Gradient Boosting**: Good balance, reduces bias.  
- **XGBoost**: High performance with careful tuning, robust to overfitting.  
- **Stacked Models**: Often outperform single models by combining strengths.  

---

## Libraries Used
- `pandas`, `numpy` – Data handling
- `matplotlib`, `seaborn` – Visualizations
- `scikit-learn` – Decision Tree, Ensemble Models, Evaluation, GridSearchCV
- `xgboost` – Extreme Gradient Boosting
- `scikit-learn`’s `StackingClassifier` – Model stacking

---

## Learning Outcomes
- Implemented tree-based and ensemble models for classification.
- Understood the effect of **hyperparameter tuning** using GridSearchCV and RandomizedSearchCV.
- Learned the advantages of **bagging, boosting, and stacking** in reducing bias/variance trade-offs.
- Compared performance across multiple models using **5-Fold Cross-Validation**.
- Gained experience interpreting ROC curves, AUC, and confusion matrices.

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo/Experiment3
