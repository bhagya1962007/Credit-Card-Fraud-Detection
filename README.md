# Credit-Card-Fraud-Detection
Credit Card Fraud Detection using Machine Learning is a Python-based machine learning project that identifies potentially fraudulent credit card transactions using transaction-level characteristics. The project uses Logistic Regression to classify transactions as either normal or fraudulent.
# Credit Card Fraud Detection Using Machine Learning

## Overview

Credit Card Fraud Detection is a machine learning project that identifies potentially fraudulent credit card transactions using transaction-related features.

The project uses **Logistic Regression** to classify transactions into two categories:

* `0` — Normal transaction
* `1` — Fraudulent transaction

The complete workflow includes data preprocessing, missing-value handling, feature scaling, train-test splitting, class balancing, model training, evaluation, fraud probability prediction, and feature analysis.

The dataset used in this project is synthetic and is intended for educational and machine learning purposes.

## Objectives

* Analyze credit card transaction data.
* Identify patterns associated with fraudulent transactions.
* Preprocess and prepare transaction data for machine learning.
* Handle missing values.
* Address class imbalance during model training.
* Train a Logistic Regression classification model.
* Evaluate model performance using multiple classification metrics.
* Predict the probability of fraud for new transactions.
* Identify important features influencing fraud predictions.
* Generate visualizations for transaction analysis.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

## Machine Learning Algorithm

### Logistic Regression

Logistic Regression is used as the primary classification algorithm.

The model predicts whether a transaction is normal or fraudulent and also provides the probability of a transaction being fraudulent.

The model is configured with balanced class weights to help address class imbalance:

```python
LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
```

## Dataset

The project uses a synthetic credit card transaction dataset.

### Features

| Feature                 | Description                                                      |
| ----------------------- | ---------------------------------------------------------------- |
| `transaction_id`        | Unique identifier for each transaction                           |
| `amount`                | Transaction amount                                               |
| `hour`                  | Hour at which the transaction occurred                           |
| `distance_from_home_km` | Distance of the transaction from the cardholder's home           |
| `merchant_risk_score`   | Risk score associated with the merchant                          |
| `transactions_last_24h` | Number of transactions made during the previous 24 hours         |
| `card_present`          | Indicates whether the physical card was present                  |
| `international`         | Indicates whether the transaction was international              |
| `online_transaction`    | Indicates whether the transaction was made online                |
| `fraud`                 | Target variable indicating whether the transaction is fraudulent |

### Target Variable

```text
0 = Normal Transaction
1 = Fraudulent Transaction
```

## Project Workflow

### 1. Data Loading

The transaction dataset is loaded using Pandas.

```python
data = pd.read_csv("credit_card_transactions.csv")
```

### 2. Data Exploration

The project examines:

* Dataset dimensions
* Sample records
* Missing values
* Fraud and non-fraud transaction counts
* Fraud percentage
* Feature distributions

### 3. Data Preprocessing

Missing numerical values are handled using median imputation.

The `transaction_id` column is removed because it is an identifier and does not provide meaningful predictive information.

### 4. Train-Test Split

The dataset is divided into:

* 80% training data
* 20% testing data

A stratified split is used to maintain the class distribution between training and testing datasets.

### 5. Feature Scaling

Numerical features are standardized using `StandardScaler`.

Feature scaling ensures that variables with different numerical ranges can be used effectively by the Logistic Regression model.

### 6. Model Training

The Logistic Regression model is trained using the processed training data.

Class balancing is enabled using:

```python
class_weight="balanced"
```

This helps give greater consideration to the minority fraud class.

### 7. Model Evaluation

The model is evaluated on the test dataset using:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix

Accuracy measures the overall percentage of correctly classified transactions.

Precision measures how many transactions predicted as fraudulent were actually fraudulent.

Recall measures how many actual fraudulent transactions were successfully identified.

F1-score provides a balance between precision and recall.

## Fraud Prediction

The trained model can be used to predict whether a new transaction is potentially fraudulent.

For example, a new transaction can be evaluated using characteristics such as:

```text
Amount: 950.00
Hour: 2
Distance from home: 120 km
Merchant risk score: 0.90
Transactions in last 24 hours: 9
Card present: No
International: Yes
Online transaction: Yes
```

The model returns both the predicted class and the estimated probability of fraud.

Example output:

```text
New Transaction Prediction: Potential Fraud
Fraud Probability: XX.XX%
```

The actual probability depends on the trained model and dataset.

## Feature Analysis

The Logistic Regression coefficients are analyzed to identify features that have a stronger influence on the model's predictions.

The absolute coefficient values are used to rank the features according to their relative importance.

This provides an interpretable view of which transaction characteristics contribute most to the classification decision.

## Visualizations

The project generates visualizations to better understand the transaction data.

### Transaction Amount vs Merchant Risk Score

A scatter plot is used to examine the relationship between transaction amount and merchant risk score while distinguishing between normal and fraudulent transactions.

Output file:

```text
fraud_transaction_analysis.png
```

### Transaction Class Distribution

A bar chart displays the number of normal and fraudulent transactions.

Output file:

```text
fraud_class_distribution.png
```

## Project Structure

```text
Credit_Card_Fraud_Detection/
│
├── credit_card_fraud_detection.py
├── credit_card_transactions.csv
├── requirements.txt
├── README.md
│
├── fraud_transaction_analysis.png
└── fraud_class_distribution.png
```

## Installation

### Prerequisites

Make sure Python 3.x is installed.

Check the installed Python version:

```bash
python --version
```

### Install Dependencies

Clone the repository:

```bash
git clone <repository-url>
```

Navigate to the project directory:

```bash
cd Credit_Card_Fraud_Detection
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the main Python script:

```bash
python credit_card_fraud_detection.py
```

The program will:

1. Load the transaction dataset.
2. Display basic dataset information.
3. Check for missing values.
4. Analyze fraud class distribution.
5. Preprocess the data.
6. Split the data into training and testing sets.
7. Scale the features.
8. Train the Logistic Regression model.
9. Evaluate the model.
10. Display the confusion matrix.
11. Predict a sample transaction.
12. Display important features.
13. Generate visualization files.

## Requirements

The project requires the following Python libraries:

```text
pandas
numpy
matplotlib
scikit-learn
```

These dependencies are also listed in `requirements.txt`.

## Model Evaluation

The model performance is evaluated using multiple metrics rather than relying only on accuracy.

This is important for fraud detection because fraudulent transactions can represent a smaller portion of the overall dataset. In such cases, precision and recall provide additional information about the model's ability to detect fraudulent transactions while limiting false alerts.

## Limitations

This project has the following limitations:

* The dataset is synthetic.
* Logistic Regression is a relatively simple classification algorithm.
* The number of available transaction features is limited.
* The project does not process real-time transactions.
* The model has not been validated against real-world fraud patterns.
* The current implementation is intended for educational purposes and should not be considered a production fraud detection system.

## Future Improvements

Possible improvements include:

* Comparing Logistic Regression with Random Forest, XGBoost, and other classification algorithms.
* Performing hyperparameter tuning.
* Applying cross-validation.
* Using SMOTE or other techniques for handling class imbalance.
* Adding ROC-AUC and Precision-Recall curves.
* Performing more extensive exploratory data analysis.
* Adding behavioral and historical transaction features.
* Optimizing the classification threshold.
* Developing a web interface for real-time predictions.
* Deploying the trained model as an API.
* Implementing model monitoring and periodic retraining.

## Conclusion

This project demonstrates the application of machine learning to credit card fraud detection using Logistic Regression.

The workflow covers the major stages of a machine learning classification project, including data preprocessing, feature scaling, class balancing, model training, evaluation, fraud probability prediction, and feature analysis.

Although the project uses a synthetic dataset and is intended for educational purposes, it provides a practical foundation for understanding how machine learning can be applied to transaction fraud detection.

## License

This project is intended for educational and learning purposes.
