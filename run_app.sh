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

# Run the fix script to add update_settings method to MarketDistributor
echo "Adding update_settings method to MarketDistributor..."
python add_update_settings.py

# Run the Streamlit app
echo "Starting Universal Market Forecasting Framework..."
streamlit run streamlit_app.py