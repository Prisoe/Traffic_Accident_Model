import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Visual Insights", layout="wide")
st.title("KSI Collision Data Visualizations")

st.markdown("""
Explore visual patterns in the Toronto KSI dataset that shaped the way we trained our predictive model. 
These plots highlight clusters, categories, time patterns, and conditions contributing to accident severity.
""")

plots = [
    ("Figure_1.png", "Clustered Accident Locations (Etobicoke to Downtown)"),
    ("Figure_1_.png", "Yearly Trend of Accidents (2006–2023)"),
    ("Figure_2.png", "Accident Concentrations by Category"),
    ("Figure_3.png", "Heatmap: Severity & Category by Region"),
    ("Figure_4.png", "Severity by Time of Day & Injury Level"),
    ("Figure_5.png", "Severity by Light, Road, and Visibility Conditions"),
    ("Figure_6.png", "Distribution of Accidents by Hour of the Day"),
    ("Figure_7.png", "Distribution of Injury Severity"),
    ("Figure_8.png", "Distribution of Accident Classifications (ACCLASS)"),
    ("Figure_9.png", "Co-occurrence Matrix of Conditions and Factors"),
    ("Figure_11.png", "Normalized Trend of Entities Involved in Accidents (2006–2023)"),
    ("Figure_12.png", "Heatmap of Accident Reasons Over Years (2006–2023)"),
    ("Figure_13.png", "Confusion Matrix for Random Forest Model"),
    ("Figure_14.png", "ROC Curve for Random Forest Model")
]

# Base URL for plot endpoint
base_url = "http://127.0.0.1:5000/plots"

# Display each plot with description
for filename, caption in plots:
    st.markdown(f"### {caption}")
    try:
        response = requests.get(f"{base_url}/{filename}")
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            st.image(image, use_container_width=True)
        else:
            st.warning(f"Plot {filename} not found on server.")
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")

    st.markdown("---")
