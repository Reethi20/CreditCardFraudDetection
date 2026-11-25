import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

print("--- Script Started ---")

# --- Load Data in Chunks ---
chunk_size = 50000
try:
    chunks = pd.read_csv('data/creditcard.csv', chunksize=chunk_size)
    print("CSV chunks created.")
except Exception as e:
    print(f"Error creating chunks: {e}")
    exit()


fraud_df = pd.DataFrame()
non_fraud_df = pd.DataFrame()

print('Processing data in chunks...')
i = 0
for chunk in chunks:
    i += 1
    print(f"Processing chunk {i}...")
    fraud_chunk = chunk[chunk['Class'] == 1]
    non_fraud_chunk = chunk[chunk['Class'] == 0]
    fraud_df = pd.concat([fraud_df, fraud_chunk])
    non_fraud_df = pd.concat([non_fraud_df, non_fraud_chunk])

print('Finished processing chunks.')
print(f"Found {len(fraud_df)} fraud transactions and {len(non_fraud_df)} non-fraud transactions.")

# Undersample the non-fraudulent data
f_count = len(fraud_df)
non_fraud_undersampled = non_fraud_df.sample(n=f_count, random_state=42)
print("Undersampling complete.")

# Combine to create the final balanced DataFrame
df = pd.concat([fraud_df, non_fraud_undersampled])
print("Combined fraud and non-fraud samples.")

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print("Dataset shuffled.")

# --- Preprocessing ---
print("Starting preprocessing...")
scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)
print("Preprocessing complete.")

X = df.drop('Class', axis=1)
y = df['Class']

# --- Model Training ---
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print("Data split. Starting model training...")

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

# --- Save the Model ---
print("Saving model and scaler...")
model_path = 'models/fraud_detection_model.joblib'
scaler_path = 'models/scaler.joblib'

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

if os.path.exists(model_path) and os.path.exists(scaler_path):
    print('Model and scaler saved successfully.')
else:
    print('Error: Model or scaler saving failed.')


# --- Evaluation ---
print("Evaluating model...")
y_pred = model.predict(X_test)

print('--- Classification Report ---')
print(classification_report(y_test, y_pred))

print('--- Confusion Matrix ---')
print(confusion_matrix(y_test, y_pred))

print("--- Script Finished ---")
