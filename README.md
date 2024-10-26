# bank-customer-retention-analysis

Updated README
Bank Customer Retention Analysis
Overview
This project analyzes bank customer data to identify factors that influence customer retention. Using logistic regression, random forest, and other statistical methods in R, the project aims to predict whether a customer will remain with the bank or exit. The insights gained can help the bank develop strategies to improve customer retention rates.

Table of Contents
Overview
Dataset
Project Structure
Prerequisites
Installation
Usage
Analysis Steps
Results
Conclusions
Visualizations
License
Acknowledgments
Dataset
The dataset used in this project is bank_customer_retention.csv, which contains the following features for each customer:

RowNumber: Unique identifier for each row
CustomerId: Unique identifier for each customer
Surname: Customer's surname
CreditScore: Credit score of the customer
Geography: Country of residence (France, Germany, Spain)
Gender: Gender of the customer (Male, Female)
Age: Age of the customer
Tenure: Number of years the customer has been with the bank
Balance: Account balance
NumOfProducts: Number of bank products the customer uses
HasCrCard: Whether the customer has a credit card (0 = No, 1 = Yes)
IsActiveMember: Whether the customer is an active member (0 = No, 1 = Yes)
EstimatedSalary: Estimated annual salary
Exited: Whether the customer has exited the bank (0 = No, 1 = Yes)
Project Structure
kotlin
Copy code
bank-customer-retention-analysis/
├── data/
│   └── bank_customer_retention.csv
├── scripts/
│   └── bank_customer_retention_analysis.R
├── results/
│   ├── model_summary.txt
│   ├── anova_results.txt
│   ├── confusion_matrix.txt
│   ├── roc_curve.png
│   ├── variable_importance_logistic.png
│   ├── variable_importance_rf.png
│   ├── roc_curves_comparison.png
│   ├── partial_dependence_age.png
│   └── partial_dependence_balance.png
├── README.md
└── LICENSE
Prerequisites
R version 3.6 or higher
R packages:
rstudioapi
car
pROC
ggplot2
dplyr
caret
corrplot
pdp
randomForest
MASS
Installation
Clone the Repository

bash
Copy code
git clone https://github.com/yourusername/bank-customer-retention-analysis.git
Navigate to the Project Directory

bash
Copy code
cd bank-customer-retention-analysis
Install Required R Packages

Open R or RStudio and run:

R
Copy code
install.packages(c("rstudioapi", "car", "pROC", "ggplot2", "dplyr", "caret", "corrplot", "pdp", "randomForest", "MASS"))
Usage
Prepare the Data

Place the bank_customer_retention.csv dataset into the data/ directory.
Ensure the dataset is clean and formatted as described in the Dataset section.
Run the Analysis Script

Open bank_customer_retention_analysis.R located in the scripts/ directory in RStudio or your preferred R environment.

Set the working directory to the script's location:

R
Copy code
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
Source the script or run it line by line to execute the analysis.

View Results

The script will output results to the console and save figures and summary files in the results/ directory.
Analysis Steps
Data Preprocessing

Converted categorical variables to factors.
Created a new binary variable Retained (1 = Retained, 0 = Exited).
Checked for and handled missing values.
Performed random sampling to select 1,000 customers for analysis.
Data Exploration

Generated summary statistics.
Visualized distributions of key variables.
Created correlation matrix and visualizations.
Feature Engineering

Created interaction terms to capture combined effects of variables (e.g., Age_IsActive).
Model Building

Logistic Regression: Built a multiple logistic regression model using significant predictors.
Model Diagnostics: Checked for multicollinearity using Variance Inflation Factor (VIF).
Threshold Optimization: Determined the optimal classification threshold based on ROC curve.
Cross-Validation: Performed 10-fold cross-validation to validate model performance.
Alternative Models: Built a Random Forest model for comparison.
Feature Selection: Used stepwise regression to identify the most significant predictors.
Model Evaluation

Evaluated model performance using confusion matrix, accuracy, precision, recall, and F1 score.
Generated ROC curves and calculated AUC for all models.
Assessed variable importance for both logistic regression and random forest models.
Visualization

Plotted ROC curves for model comparison.
Created variable importance plots.
Generated partial dependence plots for significant variables.
Results
Model Performance

Logistic Regression:
Achieved a solid accuracy rate on the test dataset.
AUC indicates good model performance.
Random Forest:
Showed improved performance over logistic regression.
Higher AUC and better classification metrics.
Stepwise Logistic Regression:
Simplified model with key predictors.
Performance comparable to the full logistic regression model.
Significant Predictors

Variables like Age, Balance, IsActiveMember, and NumOfProducts were significant in predicting customer retention.
Interaction terms provided additional insights into combined effects.
Variable Importance

Logistic Regression:
Variable importance identified key factors influencing retention.
Random Forest:
Variable importance plot highlighted the most influential predictors.
Model Comparison

ROC curves demonstrated that the Random Forest model outperformed the logistic regression models.
Cross-validation confirmed the robustness of the models.
Conclusions
Random Forest Model: Provided better predictive performance and should be considered for deployment.
Significant Factors: Age, account balance, activity status, and number of products are critical for customer retention.
Strategic Focus: The bank should focus on personalized services for high-value and active customers to improve retention rates.
Model Application: The models can be used to predict at-risk customers and develop targeted retention strategies.
Visualizations
ROC Curves Comparison

The ROC curves compare the performance of logistic regression, random forest, and stepwise logistic regression models.
Variable Importance - Logistic Regression

Highlights the most significant predictors in the logistic regression model.
Variable Importance - Random Forest

Shows the importance of variables in the random forest model.
Partial Dependence Plots
Age

Illustrates the effect of age on the probability of customer retention.
Balance

Shows how account balance influences the retention probability.
