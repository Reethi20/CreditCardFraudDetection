import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Fraud Detection", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.write("Upload a transaction CSV file to detect fraud.")

# ======================
# Upload CSV
# ======================
uploaded_file = st.file_uploader(
    "Upload CSV file (must contain Time, Amount, V1–V28)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.dataframe(df.head())

    if st.button("Predict Fraud"):
        results = []

        for _, row in df.iterrows():
            payload = row.to_dict()

            response = requests.post(
                "http://127.0.0.1:5000/predict",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()
                results.append(result["prediction"])
            else:
                results.append("Error")

        df["Prediction"] = results

        st.write("### Prediction Results")
        st.dataframe(df)

        fraud_count = df["Prediction"].value_counts().get("Fraud", 0)
        st.success(f"Fraud Transactions Detected: {fraud_count}")
