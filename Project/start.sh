#!/bin/bash
# Start Flask (port 5000) in background
python Backend/app.py &

# Start Streamlit (port 8501)
streamlit run Frontend/App.py --server.port 8501 --server.enableCORS false