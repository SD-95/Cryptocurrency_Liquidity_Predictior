from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__)

# Load models and encoders
rf_model = joblib.load("random_forest_model.pkl")
meta_model = joblib.load("meta_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")
lstm_model = load_model("lstm_model.h5")

# Expected final features for prediction
feature_names = [
    "1h", "24h", "7d",
    "price_lag1", "volume_lag1", "mktcap_lag1",
    "price_2d_avg", "volume_2d_avg",
    "vol_to_mcap", "vol_price_ratio"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("Received Input:", data)

        # Extract raw input features from HTML form
        price = float(data.get("price", 0))
        price_1h = float(data.get("price_1h", 0))
        price_24h = float(data.get("price_24h", 0))
        price_7d = float(data.get("price_7d", 0))
        volume_24h = float(data.get("volume_24h", 0))
        market_cap = float(data.get("market_cap", 0))

        # Simulate required lag and rolling features (assume lag = current - change)
        price_lag1 = price / (1 + price_1h / 100) if price_1h != -100 else 0
        volume_lag1 = volume_24h  # If you have no real lag data, assume same
        mktcap_lag1 = market_cap  # Same logic

        price_2d_avg = (price + price_lag1) / 2
        volume_2d_avg = (volume_24h + volume_lag1) / 2

        vol_to_mcap = volume_24h / market_cap if market_cap else 0
        vol_price_ratio = volume_24h / price if price else 0

        # Construct final feature vector
        final_features = [
            price_1h, price_24h, price_7d,
            price_lag1, volume_lag1, mktcap_lag1,
            price_2d_avg, volume_2d_avg,
            vol_to_mcap, vol_price_ratio
        ]

        # Reshape and scale
        input_arr = np.array(final_features).reshape(1, -1)
        input_scaled = scaler.transform(input_arr)

        # LSTM input reshape
        input_seq = input_scaled.reshape((1, 1, input_scaled.shape[1]))

        # Get model predictions
        rf_probs = rf_model.predict_proba(input_scaled)
        lstm_probs = lstm_model.predict(input_seq)

        stacked_input = np.hstack((rf_probs, lstm_probs))
        final_pred_class = meta_model.predict(stacked_input)[0]
        final_pred_proba = meta_model.predict_proba(stacked_input)[0][final_pred_class]

        liquidity_level = le.inverse_transform([final_pred_class])[0]

        # Generate advice
        if liquidity_level == "High":
            advice = "Buy"
        elif liquidity_level == "Medium":
            advice = "Hold"
        else:
            advice = "Avoid"

        return jsonify({
            "liquidity_level": liquidity_level,
            "confidence_score": round(final_pred_proba * 100, 2),
            "investment_advice": advice
        })

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT env variable
    app.run(host="0.0.0.0", port=port, debug=True)