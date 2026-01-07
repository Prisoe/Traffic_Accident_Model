import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path

st.set_page_config(page_title="HTML Form", layout="wide")
st.title("🚗 HTML-Driven Accident Prediction Form")

# Load the HTML content
HERE = Path(__file__).resolve().parent
with open(HERE / "Home.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# Display it in a Streamlit container
components.html(html_content, height=1800, scrolling=True)

































# import streamlit as st
# import requests
#
# st.title("KSI Fatality Prediction")
#
# st.markdown("""
# Welcome to the **KSI Fatality Predictor**.
# This tool helps you estimate the likelihood of a fatal motor vehicle collision in Toronto
# based on key factors such as road conditions, vehicle type, visibility, and driver behavior.
#
# Please use the dropdowns below to enter the details of the collision.
# Click **Predict** to see whether the model predicts a **Fatal** or **Non-Fatal** outcome.
# """)
#
# # Dropdown options for categorical fields
# dropdown_options = {
#     "ROAD_CLASS": ["Major Arterial", "Minor Arterial", "Collector", "Local"],
#     "ACCLOC": ["Intersection", "Mid-Block", "Other"],
#     "VISIBILITY": ["Clear", "Rain", "Fog"],
#     "RDSFCOND": ["Dry", "Wet", "Slush", "Ice"],
#     "INVTYPE": ["Driver", "Passenger", "Pedestrian"],
#     "INJURY": ["None", "Minor", "Major", "Fatal"],
#     "VEHTYPE": ["Automobile", "Station Wagon", "Truck", "Motorcycle"],
#     "DRIVCOND": ["Normal", "Impaired", "Unknown"],
#     "PEDTYPE": ["N/A", "Child", "Adult"],
#     "PEDACT": ["N/A", "Crossing"],
#     "PEDCOND": ["N/A", "Normal", "Inattentive"]
# }
#
# # Binary fields that should appear as Yes/No
# binary_fields = [
#     "PEDESTRIAN", "MOTORCYCLE", "TRUCK", "TRSN_CITY_VEH",
#     "PASSENGER", "SPEEDING", "ALCOHOL"
# ]
#
# # Standard numeric inputs
# numeric_fields = ["LATITUDE", "LONGITUDE", "YEAR", "TIME_BIN_NUM", "MONTH", "DAY_OF_WEEK"]
#
# # Collect inputs
# input_data = {}
#
# # Categorical dropdowns
# for field, options in dropdown_options.items():
#     input_data[field] = st.selectbox(field, options)
#
# # Yes/No binary dropdowns
# for field in binary_fields:
#     binary_input = st.selectbox(field, ["No", "Yes"])
#     input_data[field] = 1 if binary_input == "Yes" else 0
#
# # Standard numeric fields
# for field in numeric_fields:
#     input_data[field] = st.number_input(field, step=1.0 if field in ["LATITUDE", "LONGITUDE"] else 1)
#
# # Submit to Flask
# if st.button("Predict"):
#     response = requests.post("http://127.0.0.1:5000/predict", json=input_data)
#     if response.ok:
#         st.success(f"Prediction: {response.json()['label']} ({response.json()['prediction']})")
#     else:
#         st.error(f"Error: {response.json().get('error', 'Unexpected error')}")
