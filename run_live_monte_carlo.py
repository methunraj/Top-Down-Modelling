#!/usr/bin/env python3
"""
Run Live Monte Carlo Interface
Launch the interactive Streamlit application for real-time Monte Carlo monitoring
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the live interface
from src.streamlit.live_monte_carlo_interface import main

if __name__ == "__main__":
    main()