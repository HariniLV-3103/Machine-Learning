# ✉️ Experiment 3: Email Spam or Ham Classification

## Objective
To classify emails as **spam** or **ham** using three machine learning algorithms—Naïve Bayes, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM)—and evaluate their performance using accuracy metrics and K-Fold cross-validation.

---

## Dataset
- **Source:** [Spambase – Kaggle](https://www.kaggle.com)  
- **Description:** This dataset includes extracted features from emails along with a label indicating whether an email is spam or ham (not spam).

---

## Implementation Steps
1. **Load & Preprocess the Dataset**
   - Handle missing values
   - Normalize/standardize numerical features
   - Encode categorical data if necessary

2. **Exploratory Data Analysis (EDA)**
   - Check class balance (spam vs ham)
   - Visualize feature distributions (histograms, boxplots)
   - Plot correlations to detect important features

3. **Data Splitting**
   - Divide dataset into training and testing sets
   - Use stratified sampling to maintain class balance

4. **Model Training**
   - **Naïve Bayes**: Gaussian, Multinomial, Bernoulli variants
   - **KNN**: Vary `k`, explore KDTree and BallTree implementations
   - **SVM**: Train with Linear, Polynomial, RBF, and Sigmoid kernels

5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - ROC Curve and AUC

6. **Cross-Validation**
   - Perform **K-Fold Cross Validation (K = 5)**
   - Record average metrics across folds

7. **Comparison & Observations**
   - Compare model performance
   - Identify the most suitable algorithm for spam classification

---

## Expected Results
- Naïve Bayes is generally fast and effective for text classification.
- KNN performance depends on the choice of `k` and distance metric.
- SVM with RBF kernel usually provides strong accuracy but may be slower.

---

## Libraries Used
- `pandas` – Data handling
- `numpy` – Numerical computations
- `matplotlib`, `seaborn` – Visualizations
- `scikit-learn` – ML models, preprocessing, evaluation

---

## Learning Outcomes
- Learned how to preprocess and analyze text-based classification datasets.
- Applied **Naïve Bayes, KNN, and SVM** for email classification.
- Understood the role of **cross-validation** in ensuring model robustness.
- Evaluated models using multiple metrics beyond accuracy (Precision, Recall, F1, AUC).
- Compared strengths and weaknesses of probabilistic, instance-based, and kernel-based classifiers.

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo/Experiment2
