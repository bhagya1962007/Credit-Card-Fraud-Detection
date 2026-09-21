# Credit Card Fraud Detection

## Synopsis
This project uses machine learning to identify potentially fraudulent credit card transactions. It analyzes transaction characteristics and classifies transactions as normal or potentially fraudulent.

## Objectives
- Understand transaction data.
- Explore normal and fraudulent transactions.
- Identify useful transaction features.
- Train a classification model.
- Predict potentially fraudulent transactions.
- Evaluate model performance.

## Tools
Python, Pandas, Matplotlib, Scikit-learn

## Dataset Features
- transaction_id
- amount
- hour
- distance_from_home_km
- merchant_risk_score
- transactions_last_24h
- card_present
- international
- online_transaction
- fraud

Target:
- 0 = Normal Transaction
- 1 = Fraudulent Transaction

## Data Preparation
- Loaded transaction data using Pandas.
- Checked for missing values.
- Filled missing numerical values with median values.
- Split the data into training and testing sets.
- Standardized numerical features using StandardScaler.
- Used class balancing in Logistic Regression because fraud detection commonly has class imbalance.

## Machine Learning Algorithm
Logistic Regression is used as a binary classification model.

## Evaluation
The program calculates:
- Accuracy
- Precision
- Recall
- Classification report
- Confusion matrix
- Fraud probability for a new transaction

Precision and recall are included because accuracy alone can be misleading for fraud-detection datasets.

The program also reports the model coefficients to identify influential transaction features and generates two visualizations.

## How to Run
1. Install Python.
2. Open a terminal in this project folder.
3. Install dependencies:
   pip install -r requirements.txt
4. Run:
   python credit_card_fraud_detection.py

## Dataset Note
The included transaction dataset is synthetic and created for educational/classroom machine-learning practice. It does not contain real credit card information.

## Important Note
This project is for educational purposes and should not be used as a production fraud-detection system.
