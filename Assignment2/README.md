# Assignment 2: Loan Amount Prediction using Linear Regression

## Objective
To build and evaluate a Linear Regression model to predict loan sanction amounts based on user and property attributes using a real-world dataset.

## Dataset Description
The dataset contains user details, property attributes, and loan-related features including:
- Income
- Age
- Credit Score
- Loan Amount Requested
- Property Price, Type, Age
- Employment and Profession
- Current Loan Expenses, Defaults

## Tools and Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn (LinearRegression, metrics, preprocessing)
- Matplotlib and Seaborn for plots

## Tasks Performed
- Handled missing values and encoded categorical features
- Performed standardization using `StandardScaler`
- Conducted Exploratory Data Analysis (EDA)
- Trained a Linear Regression model using Scikit-learn
- Evaluated the model using MAE, MSE, RMSE, and R² score
- Visualized predicted vs actual values and residuals
- Analyzed feature importance using model coefficients
- Applied 5-fold cross-validation to ensure model robustness

## Results Summary
- R² Score ≈ 0.83
- RMSE ≈ 340.0
- Most Influential Features: Income, Credit Score
- No significant overfitting observed

## Learning Outcomes
- Applied linear regression to real-world prediction tasks
- Gained insight into preprocessing steps like encoding and scaling
- Evaluated model using appropriate regression metrics
- Visualized model accuracy and error distribution
- Learned the role of cross-validation in reliable evaluation

## Notes
- Python code and visualizations included
- Summary tables and learning outcomes are documented
