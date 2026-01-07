#!/bin/bash
#!/bin/bash
python Backend/app.py &
streamlit run Frontend/App.py --server.port $PORT --server.address 0.0.0.0 --server.enableCORS false
