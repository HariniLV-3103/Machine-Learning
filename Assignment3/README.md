Assignment 3: Email Spam or Ham Classification using Naive Bayes, KNN, and SVM
Objective
To classify emails as Spam (1) or Ham (0) using Naive Bayes, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM) algorithms. The aim is to evaluate model performance through statistical metrics and visualizations, and compare the effectiveness of these classifiers.

Tools and Libraries Used
Python – programming language
Pandas – data handling and preprocessing
NumPy – numerical computations
SciPy – z-score for outlier detection
Matplotlib & Seaborn – visualization and EDA
Scikit-learn – ML models, preprocessing, and evaluation metrics

Dataset Explored
Dataset Name: spam_or_not_spam.csv
Features: Email text data (preprocessed into TF-IDF vectors)

Target Label:
0 → Ham (not spam)
1 → Spam

Tasks Performed
Data Preprocessing
Removed missing values and handled outliers using z-score.
Converted text data into TF-IDF features (1000 max features).
Scaled features for algorithms sensitive to distance (KNN, SVM).
Data Splitting
Train, Validation, and Test splits created.
Raw features → Naive Bayes.
Scaled features → KNN and SVM.

Model Building
Naive Bayes: GaussianNB, MultinomialNB, BernoulliNB.
KNN: k = 1, 3, 5, 7 and kd_tree, ball_tree algorithms.
SVM: Linear, Polynomial, RBF, and Sigmoid kernels.

Evaluation Metrics Used
Accuracy
Precision
Recall
F1 Score
F-beta Score (β = 0.5)
Matthews Correlation Coefficient (MCC)
Confusion Matrix (heatmap)
ROC & AUC Curve

Cross Validation
Applied 5-Fold Stratified Cross Validation for robust model performance comparison.

Observations
MultinomialNB outperformed other Naive Bayes models, showing strong precision and balanced recall.
KNN performance varied; higher k values stabilized predictions but reduced recall slightly.
SVM (RBF & Sigmoid kernels) achieved the best performance, with high precision and strong recall, making them ideal for spam detection.

Inference
MultinomialNB is best suited for text classification tasks using TF-IDF features.
KNN is less effective on high-dimensional sparse text data.
SVM with RBF kernel is the most reliable classifier, balancing accuracy, recall, and precision.

Learning Outcomes
Learned text preprocessing using TF-IDF vectorization.
Understood the importance of feature scaling in distance-based algorithms.
Applied and compared multiple classifiers (Naive Bayes, KNN, SVM) on the same dataset.
Learned to evaluate classifiers using precision, recall, F1, MCC, ROC-AUC.
Gained insights into model suitability for sparse high-dimensional data like text.

Notes
Complete Python code includes inline comments for clarity.
Outputs & visualizations (confusion matrices, ROC curves) are included.
Final comparison is presented in summary tables for easy interpretation.
