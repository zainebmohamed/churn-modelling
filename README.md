# churn-modelling

This project aims to build a predictive machine learning model to identify Telco customers who are likely to churn.  The dataset includes a number of customer attributes including demographics, financial data and account activity indicators.

# Objective

To build and train a churn pipeline that evaluates several classification estimators for optimal recall, while handling the inherent class imbalance. The best performing model will be optimised through hyperparameter tuning and calibration. 

Recall is a critical metric in the telco industry, and so model evaluation efforts will focus exclusively on optimising recall.

### Why optimise recall?
- Higher cost of false negatives than false positives
- More expensive to replace a customer than to retain an existing one (up to 5-10 times) through targeted retention efforts

--- 

#### **`Goal: To achieve a recall of at least 0.8`**

--- 

# Dataset

The dataset contains the following columns:

1. **RowNumber**: Index of the row in the dataset (Not used in model)
2. **CustomerId**: Unique identifier for each customer (Not used in model)  
3. **Surname**: Customer’s last name (Not used in model) 
4. **CreditScore**: Credit score of the customer  
5. **Geography**: Customer’s country of residence  
6. **Gender**: Customer’s gender
7. **Age**: Customer’s age
8. **Tenure**: Number of years the customer has been with the company  
9. **Balance**: Account balance of the customer
10. **NumOfProducts**: Number of products the customer is using  
11. **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No)  
12. **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No)  
13. **EstimatedSalary**: Estimated yearly salary of the customer
14. **Exited**: Target variable indicating if the customer churned (1 = Yes, 0 = No)


# Model Workflow

The core logic of the model building, optimisation and evaluation steps is wrapped in a 'ChurnClassifier' class. 

Minimal preprocessing is conducted on the data, removing non-informative columns which will not be used in the model building stage.

## Pipeline Construction
- Encode categorical variables using OneHotEncoder
- Scale numerical variables using StandardScaler
- Handle class imbalance using model parameters e.g class_weight = 'balanced'  such that the weight applied is inversely proportional to the class frequency
- Build a modelling pipeline using sklearn's Pipeline class to prevent data leakage, applying data transformations through a ColumnTransformer

## Model Training
- Train multiple classifiers including LogisticRegression, Random Forest, SVM and XGBoost
- Perform cross-validation to evaluate recall

## Model Optimisation
- Perform hyperparameter tuning on the best-performing model using GridSearchCV
- Calibrate the final model using CalibratedClassifierCV to ensure reliable probability estimates

## Theshold Tuning
- Generate a precision recall curve using the calibrated predicted probabilities
- Identify the maximum threshold where recall >= 0.8
- Determine precision at the point of optimal recall

## Save Model
- Save the trained model using the joblib library


# Dependencies

The following libraries must be installed to run this project:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `xgboost`
- `joblib`

To install dependencies, use:

```bash
pip install -r requirements.txt
