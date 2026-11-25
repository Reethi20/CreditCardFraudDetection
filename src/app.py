from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# ===============================
# Load model & scalers
# ===============================
model = joblib.load("models/fraud_detection_model.joblib")
amount_scaler = joblib.load("models/amount_scaler.joblib")
time_scaler = joblib.load("models/time_scaler.joblib")

# ===============================
# Health check route
# ===============================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Credit Card Fraud Detection API is running"
    })

# ===============================
# Prediction route
# ===============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # Check required fields
        if "Time" not in df.columns or "Amount" not in df.columns:
            return jsonify({
                "error": "JSON must contain 'Time' and 'Amount' fields"
            }), 400

        # Preprocessing (SAME AS TRAINING)
        df["scaled_amount"] = amount_scaler.transform(
            df["Amount"].values.reshape(-1, 1)
        )
        df["scaled_time"] = time_scaler.transform(
            df["Time"].values.reshape(-1, 1)
        )

        df.drop(["Time", "Amount"], axis=1, inplace=True)

        # Align feature order
        df = df.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

        # Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        result = {
            "prediction": "Fraud" if prediction == 1 else "Not Fraud",
            "fraud_probability": round(float(probability), 4)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
