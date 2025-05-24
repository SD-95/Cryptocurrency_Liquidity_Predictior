from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os

# Suppress TensorFlow oneDNN optimization logs (optional)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)

try:
    rf_model = joblib.load("random_forest_model.pkl")
    meta_model = joblib.load("meta_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    lstm_model = load_model("lstm_model.h5")
except Exception as e:
    print(f"❌ Model loading failed: {e}")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    advice = None

    if request.method == "POST":
        try:
            price = float(request.form.get("price", 0))
            price_1h = float(request.form.get("price_1h", 0))
            price_24h = float(request.form.get("price_24h", 0))
            price_7d = float(request.form.get("price_7d", 0))
            volume_24h = float(request.form.get("volume_24h", 0))
            market_cap = float(request.form.get("market_cap", 0))

            # Feature engineering
            price_lag1 = price / (1 + price_1h / 100) if price_1h != -100 else 0
            volume_lag1 = volume_24h
            mktcap_lag1 = market_cap

            price_2d_avg = (price + price_lag1) / 2
            volume_2d_avg = (volume_24h + volume_lag1) / 2

            vol_to_mcap = volume_24h / market_cap if market_cap else 0
            vol_price_ratio = volume_24h / price if price else 0

            final_features = [
                price_1h, price_24h, price_7d,
                price_lag1, volume_lag1, mktcap_lag1,
                price_2d_avg, volume_2d_avg,
                vol_to_mcap, vol_price_ratio
            ]

            input_arr = np.array(final_features).reshape(1, -1)
            input_scaled = scaler.transform(input_arr)
            input_seq = input_scaled.reshape((1, 1, input_scaled.shape[1]))

            rf_probs = rf_model.predict_proba(input_scaled)
            lstm_probs = lstm_model.predict(input_seq)

            stacked_input = np.hstack((rf_probs, lstm_probs))
            final_pred_class = meta_model.predict(stacked_input)[0]
            final_pred_proba = meta_model.predict_proba(stacked_input)[0][final_pred_class]

            liquidity_level = le.inverse_transform([final_pred_class])[0]

            advice_map = {
                "High": "Buy",
                "Medium": "Hold",
                "Low": "Avoid"
            }
            advice = advice_map.get(liquidity_level, "Unknown")

            prediction = liquidity_level
            confidence = round(final_pred_proba * 100, 2)

        except Exception as e:
            prediction = f"Error: {e}"
            confidence = "-"
            advice = "-"

    return render_template(
        "index.html",
        prediction_text=prediction,
        confidence_score=confidence,
        investment_advice=advice
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)