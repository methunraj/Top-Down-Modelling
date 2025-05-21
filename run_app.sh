#!/bin/bash

# Universal Market Forecasting Framework - Run Script
# This script starts the Streamlit application

# Ensure we're in the project directory
cd "$(dirname "$0")"

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing required packages..."
    pip install -r enhanced_requirements.txt
fi

# Run the Streamlit app
echo "Starting Universal Market Forecasting Framework..."
streamlit run streamlit_app.py