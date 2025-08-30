Risk Classification Project

Overview

This project performs a multiclass classification task to predict risk levels (Low, Medium, High) using the Ministry of Revenue Risk Management dataset. The dataset contains 4,787 records and 29 attributes related to financial and operational metrics of companies. Three machine learning models are explored: Decision Tree, Support Vector Machine (SVM), and Feedforward Neural Network (FNN). The goal is to classify companies into risk categories based on their financial and audit-related features.

Dataset





Source: Ministry of Revenue Risk Management dataset



Size: 4,787 rows, 29 columns



Attributes: Financial ratios and indicators (e.g., _c1_loss_declaration, _c2_capital_to_turnover, _c24_vat_import_purchase, etc.)



Target Variable: risk_level (Low: 0, Medium: 1, High: 2)



Key Preprocessing Steps:





Dropped last_audit_year due to 3,171 missing values.



Dropped _c29_foreign_company due to constant values.



Removed final_score and risk_score to avoid data leakage.



Converted risk_level categorical values to numerical (Low: 0, Medium: 1, High: 2).



Applied SMOTE to balance the dataset, addressing class imbalance.



Split data into 80% training and 20% testing sets.

Methodology





Data Exploration:





Checked for missing values and outliers.



Analyzed class distribution to identify imbalance.



Computed correlation matrix to understand feature relationships with risk_level.



Data Preprocessing:





Removed columns with missing or constant values.



Balanced the dataset using SMOTE to ensure equal representation of Low, Medium, and High risk classes.



Scaled features implicitly through model requirements (e.g., SVM, FNN).



Models:





Decision Tree:





Tuned using GridSearchCV and RandomizedSearchCV.



Best parameters: {'max_depth': 10, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 5} (from RandomizedSearchCV).



SVM:





Tuned using GridSearchCV.



Best parameters: {'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}.



Feedforward Neural Network:





Architecture: 3 layers (26 neurons with ReLU, 32 neurons with ReLU, softmax output).



Dropout layers (0.3) to prevent overfitting.



Trained for 100 epochs with Adam optimizer and sparse categorical crossentropy loss.



Evaluation Metrics:





Accuracy, Precision, Recall, F1 Score (macro-averaged).



Confusion Matrix for test set performance.

Results





Decision Tree:





Test Accuracy: 0.85



Precision: 0.85



Recall: 0.86



F1 Score: 0.85



Train Accuracy: 0.92



SVM:





Train Accuracy: 0.96 (test metrics not fully shown in notebook).



Feedforward Neural Network:





Test Accuracy: 0.94



Train Accuracy: 0.95

The FNN outperformed the Decision Tree and SVM in test accuracy, achieving 94% accuracy on the test set.

Requirements

To run the notebook, install the following Python libraries:

pip install numpy pandas matplotlib seaborn scikit-learn imblearn tensorflow

How to Run





Clone the repository:

git clone https://github.com/ElshadayD/Risk_Classification.git
cd Risk_Classification



Ensure the dataset (Risk-Data - Sheet1.csv) is in the project directory.



Activate your Python environment (e.g., myenv):

source myenv/bin/activate



Install dependencies:

pip install -r requirements.txt



Launch Jupyter Notebook:

jupyter notebook



Open and run Risk_Classification.ipynb.

Notes





The dataset (Risk-Data - Sheet1.csv) is not included in the repository due to potential sensitivity. Ensure you have access to it.



The notebook includes visualizations (e.g., class distribution, correlation heatmap) that require a graphical environment to display.



For large datasets, consider clearing notebook outputs before committing to GitHub to reduce file size (use nbstripout or clear outputs manually).

Future Improvements





Explore additional models (e.g., Random Forest, XGBoost).



Perform feature selection to reduce dimensionality.



Investigate alternative balancing techniques (e.g., ADASYN).



Add cross-validation for more robust evaluation.

Author





ElshadayD
