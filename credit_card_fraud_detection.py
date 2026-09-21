# Credit Card Fraud Detection
# Tools: Python, Pandas, Matplotlib, Scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# Load transaction data
data = pd.read_csv("credit_card_transactions.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset shape:", data.shape)

print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Clean missing numerical values using median
numeric_columns = data.select_dtypes(include=["number"]).columns
for column in numeric_columns:
    if data[column].isnull().any():
        data[column] = data[column].fillna(data[column].median())

print("\nMissing values after cleaning:", data.isnull().sum().sum())

print("\nTransaction class distribution:")
print(data["fraud"].value_counts())
print("\nFraud percentage:", f"{data['fraud'].mean() * 100:.2f}%")

# Features and target
X = data.drop(columns=["transaction_id", "fraud"])
y = data["fraud"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build model with feature scaling
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=["Normal", "Fraud"],
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Predict a new transaction
new_transaction = pd.DataFrame([{
    "amount": 950.00,
    "hour": 2,
    "distance_from_home_km": 120,
    "merchant_risk_score": 0.90,
    "transactions_last_24h": 9,
    "card_present": 0,
    "international": 1,
    "online_transaction": 1
}])

prediction = model.predict(new_transaction)[0]
fraud_probability = model.predict_proba(new_transaction)[0][1]

print("\nNew Transaction Prediction:",
      "Potential Fraud" if prediction == 1 else "Normal Transaction")
print(f"Fraud Probability: {fraud_probability * 100:.2f}%")

# Feature importance from logistic regression coefficients
feature_names = X.columns
coefficients = model.named_steps["classifier"].coef_[0]

importance = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefficients,
    "absolute_importance": abs(coefficients)
}).sort_values("absolute_importance", ascending=False)

print("\nImportant Transaction Features:")
print(importance[["feature", "coefficient"]])

# Visualization 1: transaction amount vs merchant risk
plt.figure(figsize=(8, 5))
plt.scatter(
    data["amount"],
    data["merchant_risk_score"],
    c=data["fraud"],
    alpha=0.5
)
plt.xlabel("Transaction Amount")
plt.ylabel("Merchant Risk Score")
plt.title("Normal vs Fraudulent Transactions")
plt.tight_layout()
plt.savefig("fraud_transaction_analysis.png", dpi=150)
plt.show()

# Visualization 2: fraud count
plt.figure(figsize=(7, 5))
data["fraud"].value_counts().sort_index().plot(kind="bar")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.title("Normal and Fraudulent Transaction Counts")
plt.xticks([0, 1], ["Normal", "Fraud"], rotation=0)
plt.tight_layout()
plt.savefig("fraud_class_distribution.png", dpi=150)
plt.show()
