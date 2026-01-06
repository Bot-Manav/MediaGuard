#!/bin/bash

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting Streamlit app..."
python -m streamlit run streamlit_app.py \
  --server.port=8000 \
  --server.address=0.0.0.0
