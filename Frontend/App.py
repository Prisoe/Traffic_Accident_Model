import streamlit as st

st.set_page_config(page_title="Toronto KSI Dashboard")
st.title("🚦 Toronto KSI Dashboard")

st.markdown("""
Use the sidebar to:
- Predict accident severity
- Explore data visualizations
- Learn more about our model's predictors
""")

# pg = st.navigation([st.Page("pages/Home.py"), st.Page("pages/Plots.py"), st.Page("pages/Predictor_Info.py")])
# pg.run()