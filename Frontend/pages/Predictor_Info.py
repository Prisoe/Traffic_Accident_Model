import streamlit as st

st.set_page_config(page_title="Predictor Reference Guide")
st.title("Understanding the Predictors")

st.markdown("""
Each of the predictors used in our model captures important aspects of traffic collisions in Toronto.
This page breaks them down so you know **what they mean**, **how they were encoded**, and **why they matter**.
""")

predictors = {
    "ROAD_CLASS": "Classification of the road (e.g., Major Arterial, Local), indicating traffic flow and risk zones.",
    "LATITUDE & LONGITUDE": "Geographic location of the collision, used to identify spatial trends and clustering.",
    "ACCLOC": "Whether the accident happened at an intersection, mid-block, or other area.",
    "VISIBILITY": "Weather-related visibility conditions (Clear, Rain, Fog).",
    "RDSFCOND": "Road surface condition (Dry, Wet, Ice, etc.).",
    "INVTYPE": "Role of the individual involved in the accident (Driver, Passenger, Pedestrian).",
    "INJURY": "Injury level sustained (None, Minor, Major, Fatal).",
    "VEHTYPE": "Type of vehicle involved (e.g., Automobile, Truck, Motorcycle).",
    "DRIVCOND": "Driver's condition (Normal, Impaired, Unknown).",
    "PEDTYPE": "Type of pedestrian involved (Adult, Child, N/A).",
    "PEDACT": "Activity of the pedestrian at time of collision (Crossing, N/A).",
    "PEDCOND": "Condition of the pedestrian (Normal, Inattentive, etc.).",
    "PEDESTRIAN": "Binary: Was a pedestrian involved? (Yes/No).",
    "MOTORCYCLE": "Binary: Was a motorcycle involved? (Yes/No).",
    "TRUCK": "Binary: Was a truck involved? (Yes/No).",
    "TRSN_CITY_VEH": "Binary: Was a city vehicle involved?",
    "PASSENGER": "Binary: Was a passenger involved?",
    "SPEEDING": "Binary: Did speeding contribute to the collision?",
    "ALCOHOL": "Binary: Was alcohol a factor in the incident?",
    "YEAR": "Year of occurrence (e.g., 2022).",
    "TIME_BIN_NUM": "Numeric representation of time-of-day (e.g., Morning=0, Afternoon=1...).",
    "MONTH": "Numeric month (1 = Jan, 12 = Dec).",
    "DAY_OF_WEEK": "Day of the week (0 = Monday, 6 = Sunday)."
}

for predictor, explanation in predictors.items():
    with st.expander(predictor):
        st.write(explanation)
