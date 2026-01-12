import os
from pathlib import Path

import requests
import streamlit as st

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
st.set_page_config(page_title="KSI Fatality Prediction", layout="wide")

# IMPORTANT:
# - On Render (single service running both Streamlit + Flask in same container),
#   Flask is reachable from Streamlit at localhost:5000
# - If you later split services, set API_BASE_URL in Render env vars
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
PREDICT_URL = f"{API_BASE_URL}/predict"

st.title("KSI Fatality Prediction")

st.markdown(
    """
Welcome to the **KSI Fatality Predictor**.  
This tool estimates the likelihood of a **Fatal** vs **Non-Fatal** collision outcome based on
road conditions, vehicle type, visibility, and driver behavior.

Fill the fields below and click **Predict**.
"""
)

# ------------------------------------------------------------
# Dropdown options (must match backend category_mappings keys)
# ------------------------------------------------------------
dropdown_options = {
    "ROAD_CLASS": ["Major Arterial", "Minor Arterial", "Collector", "Local"],
    "ACCLOC": ["Intersection", "Mid-Block", "Other"],
    "VISIBILITY": ["Clear", "Rain", "Fog"],
    "RDSFCOND": ["Dry", "Wet", "Slush", "Ice"],
    "INVTYPE": ["Driver", "Passenger", "Pedestrian"],
    "INJURY": ["None", "Minor", "Major", "Fatal"],
    # FIX: must match keys used in your backend mapping
    # Backend supports:
    # "Automobile", "Automobile, Station Wagon", "Truck", "Motorcycle"
    "VEHTYPE": ["Automobile", "Automobile, Station Wagon", "Truck", "Motorcycle"],
    "DRIVCOND": ["Normal", "Impaired", "Unknown"],
    "PEDTYPE": ["N/A", "Child", "Adult"],
    "PEDACT": ["N/A", "Crossing"],
    "PEDCOND": ["N/A", "Normal", "Inattentive"],
}

binary_fields = [
    "PEDESTRIAN", "MOTORCYCLE", "TRUCK", "TRSN_CITY_VEH",
    "PASSENGER", "SPEEDING", "ALCOHOL",
]

# Numeric fields expected by backend
# LAT/LONG should be float inputs; rest ints
# (Your backend eventually casts everything to float anyway.)
# ------------------------------------------------------------
st.subheader("Inputs")

input_data = {}

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Categorical")
    for field, options in dropdown_options.items():
        input_data[field] = st.selectbox(field, options)

with col2:
    st.markdown("### Location / Time")
    input_data["LATITUDE"] = st.number_input("LATITUDE", value=43.6532, format="%.6f")
    input_data["LONGITUDE"] = st.number_input("LONGITUDE", value=-79.3832, format="%.6f")

    input_data["YEAR"] = st.number_input("YEAR", value=2024, step=1)
    input_data["MONTH"] = st.number_input("MONTH", value=1, step=1, min_value=1, max_value=12)
    input_data["DAY_OF_WEEK"] = st.number_input("DAY_OF_WEEK", value=0, step=1, min_value=0, max_value=6)

    # TIME_BIN_NUM: 0=Night(0-6), 1=Morning(6-12), 2=Afternoon(12-18), 3=Evening(18-24)
    input_data["TIME_BIN_NUM"] = st.selectbox("TIME_BIN_NUM", [0, 1, 2, 3], index=1)

st.markdown("### Binary flags")
bin_cols = st.columns(4)
for i, field in enumerate(binary_fields):
    with bin_cols[i % 4]:
        choice = st.selectbox(field, ["No", "Yes"], key=field)
        input_data[field] = 1 if choice == "Yes" else 0

# ------------------------------------------------------------
# Predict
# ------------------------------------------------------------
st.divider()
st.caption(f"API endpoint: {PREDICT_URL}")

if st.button("Predict", type="primary"):
    try:
        resp = requests.post(PREDICT_URL, json=input_data, timeout=20)

        # If backend returns non-JSON on error, guard it
        try:
            payload = resp.json()
        except Exception:
            payload = {"error": resp.text}

        if resp.ok:
            st.success(f"Prediction: **{payload['label']}** ({payload['prediction']})")
        else:
            st.error(f"Error: {payload.get('error', 'Unexpected error')}")
            st.write("Status:", resp.status_code)
            st.json(payload)

    except requests.exceptions.RequestException as e:
        st.error("Error: Unable to get prediction (network/backend issue).")
        st.write(str(e))
