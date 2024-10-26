#Andrew Szigety
#CS 555 - Final Project
#Bank Customer Retention Analysis

#Load Necessary Libraries
library(rstudioapi)  #For setting the working directory
library(car)         #For ANOVA and VIF analysis
library(pROC)        #For ROC curves
library(ggplot2)     #For data visualization
library(dplyr)       #For data manipulation
library(caret)       #For model training and evaluation
library(corrplot)    #For correlation plots
library(pdp)         #For partial dependence plots
library(randomForest) #For Random Forest model
library(MASS)        #For stepwise regression

#Set Working Directory to Current Script's Location
setwd(dirname(getActiveDocumentContext()$path))

#Read the Data
bank_info <- read.csv("data/bank_customer_retention.csv", stringsAsFactors = FALSE)

#Data Preprocessing

#Create a 'Retained' Variable (1 = Retained, 0 = Exited)
bank_info$Retained <- ifelse(bank_info$Exited == 0, 1, 0)

#Convert Categorical Variables to Factors
bank_info$Geography <- factor(bank_info$Geography)
bank_info$Gender <- factor(bank_info$Gender)
bank_info$HasCrCard <- factor(bank_info$HasCrCard, labels = c("No", "Yes"))
bank_info$IsActiveMember <- factor(bank_info$IsActiveMember, labels = c("Inactive", "Active"))

#Check for Missing Values
missing_values <- sapply(bank_info, function(x) sum(is.na(x)))
print("Missing values in each column:")
print(missing_values)

#Handle Missing Values (if any)
#For simplicity, we'll remove rows with missing values
bank_info <- na.omit(bank_info)

#Data Exploration

#Summary Statistics
summary(bank_info)

#Plot Distribution of Retained vs. Exited Customers
ggplot(bank_info, aes(x = factor(Retained))) +
  geom_bar(fill = c("red", "green")) +
  labs(title = "Distribution of Retained vs. Exited Customers", x = "Retention Status", y = "Count") +
  scale_x_discrete(labels = c("Exited", "Retained"))

#Visualize Numeric Variables
numeric_vars <- c("CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary")

#Histograms for Numeric Variables
par(mfrow = c(2, 3))
for (var in numeric_vars) {
  hist(bank_info[[var]], main = paste("Histogram of", var), xlab = var, col = "skyblue")
}

#Correlation Matrix
numeric_data <- bank_info[, numeric_vars]
corr_matrix <- cor(numeric_data)
print("Correlation Matrix:")
print(corr_matrix)

#Visualize Correlation Matrix
corrplot(corr_matrix, method = "ellipse")

#Random Sampling (if dataset is large; here, sample 1,000 rows)
set.seed(1)
bank_sample <- bank_info %>% sample_n(1000)

#Feature Engineering

#Create Interaction Terms (if relevant)
#Example: Interaction between Age and IsActiveMember
bank_sample$Age_IsActive <- bank_sample$Age * as.numeric(bank_sample$IsActiveMember)

#Model Building

#Split the Data into Training and Testing Sets
set.seed(123)
train_index <- createDataPartition(bank_sample$Retained, p = 0.8, list = FALSE)
train_data <- bank_sample[train_index, ]
test_data <- bank_sample[-train_index, ]

#Multiple Logistic Regression Model
model_formula <- Retained ~ CreditScore + Age + Geography + Gender + Tenure + Balance +
  NumOfProducts + HasCrCard + IsActiveMember + EstimatedSalary + Age_IsActive

logistic_model <- glm(model_formula, data = train_data, family = binomial)

#Model Summary
summary(logistic_model)

#Model Diagnostics

#Check for Multicollinearity using VIF
vif_values <- vif(logistic_model)
print("VIF Values:")
print(vif_values)

#Threshold Optimization

#Predict Probabilities on Test Set
test_data$Predicted_Prob <- predict(logistic_model, newdata = test_data, type = "response")

#Determine Optimal Threshold based on ROC Curve
roc_curve <- roc(test_data$Retained ~ test_data$Predicted_Prob)
optimal_coords <- coords(roc_curve, "best", ret = "threshold", transpose = FALSE)
optimal_threshold <- optimal_coords$threshold
print(paste("Optimal Threshold:", optimal_threshold))

#Convert Probabilities to Class Labels using Optimal Threshold
test_data$Predicted_Class <- ifelse(test_data$Predicted_Prob >= optimal_threshold, 1, 0)

#Model Evaluation

#Confusion Matrix
conf_matrix <- confusionMatrix(factor(test_data$Predicted_Class), factor(test_data$Retained), positive = "1")
print("Confusion Matrix:")
print(conf_matrix)

#ROC Curve and AUC
auc_value <- auc(roc_curve)
print(paste("AUC:", auc_value))

#Plot ROC Curve
plot(roc_curve, col = "blue", main = "ROC Curve for Logistic Regression Model")

#Cross-Validation

#Perform 10-fold Cross-Validation
control <- trainControl(method = "cv", number = 10, classProbs = TRUE, summaryFunction = twoClassSummary)
cv_model <- train(model_formula, data = train_data, method = "glm", family = "binomial", trControl = control, metric = "ROC")
print("Cross-Validation Results:")
print(cv_model)

#Alternative Models

#Random Forest Model
set.seed(123)
rf_model <- randomForest(Retained ~ CreditScore + Age + Geography + Gender + Tenure + Balance +
                           NumOfProducts + HasCrCard + IsActiveMember + EstimatedSalary + Age_IsActive,
                         data = train_data, importance = TRUE, ntree = 500)

#Predict on Test Set
test_data$RF_Predicted_Prob <- predict(rf_model, newdata = test_data, type = "prob")[,2]
test_data$RF_Predicted_Class <- predict(rf_model, newdata = test_data)

#Evaluate Random Forest Model
rf_conf_matrix <- confusionMatrix(test_data$RF_Predicted_Class, factor(test_data$Retained), positive = "1")
print("Random Forest Confusion Matrix:")
print(rf_conf_matrix)

#ROC Curve for Random Forest
rf_roc_curve <- roc(test_data$Retained ~ test_data$RF_Predicted_Prob)
rf_auc_value <- auc(rf_roc_curve)
print(paste("Random Forest AUC:", rf_auc_value))

#Plot ROC Curve for Random Forest
plot(rf_roc_curve, col = "red", main = "ROC Curve for Random Forest Model")
legend("bottomright", legend = c("Logistic Regression", "Random Forest"), col = c("blue", "red"), lwd = 2)

#Variable Importance Plot for Random Forest
varImpPlot(rf_model, main = "Variable Importance - Random Forest")

#Feature Selection

#Stepwise Regression using AIC
step_model <- stepAIC(logistic_model, direction = "both")
summary(step_model)

#Predict with Stepwise Model
test_data$Step_Predicted_Prob <- predict(step_model, newdata = test_data, type = "response")
test_data$Step_Predicted_Class <- ifelse(test_data$Step_Predicted_Prob >= optimal_threshold, 1, 0)

#Evaluate Stepwise Model
step_conf_matrix <- confusionMatrix(factor(test_data$Step_Predicted_Class), factor(test_data$Retained), positive = "1")
print("Stepwise Logistic Regression Confusion Matrix:")
print(step_conf_matrix)

#ROC Curve for Stepwise Model
step_roc_curve <- roc(test_data$Retained ~ test_data$Step_Predicted_Prob)
step_auc_value <- auc(step_roc_curve)
print(paste("Stepwise Logistic Regression AUC:", step_auc_value))

#Plot ROC Curves for Comparison
plot(roc_curve, col = "blue", main = "ROC Curves Comparison")
plot(rf_roc_curve, col = "red", add = TRUE)
plot(step_roc_curve, col = "green", add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "Random Forest", "Stepwise Logistic Regression"),
       col = c("blue", "red", "green"), lwd = 2)

#ANOVA Analysis
anova_results <- Anova(logistic_model, type = 3)
print("ANOVA Results:")
print(anova_results)

#Additional Visualizations

#Variable Importance Plot for Logistic Regression
logit_var_importance <- varImp(cv_model, scale = FALSE)
print("Variable Importance - Logistic Regression:")
print(logit_var_importance)
plot(logit_var_importance, main = "Variable Importance - Logistic Regression")

#Partial Dependence Plots for Significant Variables
#For Logistic Regression
partial_plot_age <- partial(logistic_model, pred.var = "Age", plot = TRUE, rug = TRUE, main = "Partial Dependence Plot for Age")
partial_plot_balance <- partial(logistic_model, pred.var = "Balance", plot = TRUE, rug = TRUE, main = "Partial Dependence Plot for Balance")

#For Random Forest
partial_plot_rf_age <- partial(rf_model, pred.var = "Age", plot = TRUE, rug = TRUE, main = "Partial Dependence Plot for Age (Random Forest)")
partial_plot_rf_balance <- partial(rf_model, pred.var = "Balance", plot = TRUE, rug = TRUE, main = "Partial Dependence Plot for Balance (Random Forest)")
