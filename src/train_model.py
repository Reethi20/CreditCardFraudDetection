import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)
import joblib
import os

print("----- SCRIPT STARTED -----")

# ===============================
# 1. LOAD DATA IN CHUNKS
# ===============================
chunk_size = 50000

try:
    chunks = pd.read_csv("data/creditcard.csv", chunksize=chunk_size)
    print("CSV loaded in chunks.")
except Exception as e:
    print("Error loading CSV:", e)
    exit()

fraud_list = []
non_fraud_list = []

print("Processing chunks...")
for i, chunk in enumerate(chunks, start=1):
    print(f"Processing chunk {i}")
    fraud_list.append(chunk[chunk["Class"] == 1])
    non_fraud_list.append(chunk[chunk["Class"] == 0])

fraud_df = pd.concat(fraud_list)
non_fraud_df = pd.concat(non_fraud_list)

print(f"Fraud transactions: {len(fraud_df)}")
print(f"Non-fraud transactions: {len(non_fraud_df)}")

# ===============================
# 2. UNDERSAMPLING
# ===============================
fraud_count = len(fraud_df)

non_fraud_undersampled = non_fraud_df.sample(
    n=fraud_count,
    random_state=42
)

df = pd.concat([fraud_df, non_fraud_undersampled])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset balanced and shuffled.")

# ===============================
# 3. PREPROCESSING
# ===============================
amount_scaler = StandardScaler()
time_scaler = StandardScaler()

df["scaled_amount"] = amount_scaler.fit_transform(
    df["Amount"].values.reshape(-1, 1)
)

df["scaled_time"] = time_scaler.fit_transform(
    df["Time"].values.reshape(-1, 1)
)

df.drop(["Amount", "Time"], axis=1, inplace=True)

X = df.drop("Class", axis=1)
y = df["Class"]

print("Preprocessing completed.")

# ===============================
# 4. TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Data split completed.")

# ===============================
# 5. MODEL TRAINING
# ===============================
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed.")

# ===============================
# 6. SAVE MODEL & SCALERS
# ===============================
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/fraud_detection_model.joblib")
joblib.dump(amount_scaler, "models/amount_scaler.joblib")
joblib.dump(time_scaler, "models/time_scaler.joblib")

print("Model and scalers saved successfully.")

# ===============================
# 7. EVALUATION
# ===============================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n----- CLASSIFICATION REPORT -----")
print(classification_report(y_test, y_pred))

print("----- CONFUSION MATRIX -----")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

print("----- SCRIPT FINISHED -----")
