from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
import os
import numpy as np
import logging
from datetime import datetime
from helpers import scale_lat_long
from joblib import load
from flask_cors import CORS
from pathlib import Path

# -------------------------
# Paths (Render-safe)
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgb_model.pkl"
SCALER_PATH = BASE_DIR / "lat_long_scaler.pkl"
PLOTS_DIR = BASE_DIR / "Plots"

# Configure logging for debugging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Load the trained XGBoost model
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    logging.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Load the scaler
try:
    scaler = load(SCALER_PATH)
    logging.info(f"Scaler loaded successfully from {SCALER_PATH}")
except Exception as e:
    logging.error(f"Error loading scaler: {e}")
    scaler = None

app = Flask(__name__)
CORS(app)

# Input columns expected by the model (from X_train_resampled)
feature_columns = [
    "ROAD_CLASS", "LATITUDE", "LONGITUDE", "ACCLOC", "VISIBILITY",
    "RDSFCOND", "INVTYPE", "INJURY", "VEHTYPE", "DRIVCOND", "PEDTYPE",
    "PEDACT", "PEDCOND", "PEDESTRIAN", "MOTORCYCLE", "TRUCK",
    "TRSN_CITY_VEH", "PASSENGER", "SPEEDING", "ALCOHOL", "YEAR",
    "TIME_BIN_NUM", "MONTH", "DAY_OF_WEEK"
]

# Hardcoded ordinal mappings (from training dataset)
category_mappings = {
    "ROAD_CLASS": {
        "Major Arterial": 0, "Minor Arterial": 1, "Collector": 2, "Local": 3
    },
    "ACCLOC": {
        "Intersection": 0, "Mid-Block": 1, "Other": 2
    },
    "VISIBILITY": {
        "Clear": 0, "Rain": 1, "Fog": 2
    },
    "RDSFCOND": {
        "Dry": 0, "Wet": 1, "Slush": 2, "Ice": 3
    },
    "INVTYPE": {
        "Driver": 0, "Passenger": 1, "Pedestrian": 2
    },
    "INJURY": {
        "None": 0, "Minor": 1, "Major": 2, "Fatal": 3
    },
    "VEHTYPE": {
        "Automobile": 0, "Automobile, Station Wagon": 0, "Truck": 1, "Motorcycle": 2
    },
    "DRIVCOND": {
        "Normal": 0, "Impaired": 1, "Unknown": 2
    },
    "PEDTYPE": {
        "N/A": 0, "Child": 1, "Adult": 2
    },
    "PEDACT": {
        "N/A": 0, "Crossing": 1
    },
    "PEDCOND": {
        "N/A": 0, "Normal": 1, "Inattentive": 2
    }
}

@app.route("/predict", methods=["POST"])
def predict():
    # Guardrails: if artifacts failed to load, return a clear error
    if model is None:
        return jsonify({"error": "Model not loaded on server. Check server logs and file paths."}), 500
    if scaler is None:
        return jsonify({"error": "Scaler not loaded on server. Check server logs and file paths."}), 500

    data = request.get_json()
    logging.info(f"Incoming data: {data}")

    # Extract features from datetime if present
    if "datetime" in data:
        try:
            dt = datetime.fromisoformat(data["datetime"])
            data["YEAR"] = dt.year
            data["MONTH"] = dt.month
            data["DAY_OF_WEEK"] = dt.weekday()
            hour = dt.hour
            if hour < 6:
                data["TIME_BIN_NUM"] = 0
            elif hour < 12:
                data["TIME_BIN_NUM"] = 1
            elif hour < 18:
                data["TIME_BIN_NUM"] = 2
            else:
                data["TIME_BIN_NUM"] = 3
            del data["datetime"]
        except Exception as e:
            logging.error(f"Failed to parse datetime: {e}")
            return jsonify({"error": "Invalid datetime format. Must be ISO format."}), 400

    for col in feature_columns:
        if col not in data:
            return jsonify({"error": f"Missing input: {col}"}), 400

    for col, mapping in category_mappings.items():
        if col in data and isinstance(data[col], str):
            try:
                data[col] = mapping[data[col].strip()]
            except KeyError:
                return jsonify({
                    "error": f"Invalid value '{data[col]}' for {col}. Acceptable: {list(mapping.keys())}"
                }), 400

    try:
        user_lat_long = np.array([[data["LATITUDE"], data["LONGITUDE"]]])
        logging.info(f"Raw lat/long: {user_lat_long}")

        scaled_lat_long = scaler.transform(user_lat_long)
        data["LATITUDE"], data["LONGITUDE"] = scaled_lat_long[0]

        logging.info(f"Scaled lat/long: {scaled_lat_long[0]}")
    except Exception as e:
        logging.error(f"Error during scaling: {e}")
        return jsonify({"error": f"Failed to scale coordinates: {str(e)}"}), 500

    input_df = pd.DataFrame([data], columns=feature_columns)
    input_df = input_df.astype(float)

    prediction = model.predict(input_df)[0]
    label = "Fatal" if prediction == 1 else "Non-Fatal"

    logging.info(f"Prediction result: {label} ({int(prediction)})")

    return jsonify({"prediction": int(prediction), "label": label})

@app.route("/plots/<filename>", methods=["GET"])
def serve_plot(filename):
    # Render/Linux is case-sensitive: folder is "Plots", not "plots"
    return send_from_directory(str(PLOTS_DIR), filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
