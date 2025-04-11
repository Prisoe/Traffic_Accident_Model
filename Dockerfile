# Use base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY . .

# Start both Streamlit and Flask using a process manager like supervisord
CMD ["bash", "start.sh"]
