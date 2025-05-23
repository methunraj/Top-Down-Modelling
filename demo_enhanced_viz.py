#!/usr/bin/env python3
"""
Demo script for Enhanced Market Visualizations

This script demonstrates the new visualization capabilities including:
- Animated bar chart races
- Interactive world maps  
- Advanced country comparisons
- Executive dashboards
- Professional styling and themes
"""

import sys
import os
import pandas as pd
import numpy as np
import random
import streamlit as st

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.visualization.enhanced_visualizer import EnhancedMarketVisualizer
from src.streamlit.enhanced_visualization import render_enhanced_visualization_interface

def generate_demo_data():
    """Generate realistic demo market data"""
    
    # Define countries with realistic market characteristics
    countries_data = {
        'United States': {'base_size': 450e9, 'growth_rate': 0.06, 'volatility': 0.03},
        'China': {'base_size': 380e9, 'growth_rate': 0.12, 'volatility': 0.05},
        'Germany': {'base_size': 120e9, 'growth_rate': 0.04, 'volatility': 0.02},
        'Japan': {'base_size': 160e9, 'growth_rate': 0.02, 'volatility': 0.02},
        'United Kingdom': {'base_size': 90e9, 'growth_rate': 0.05, 'volatility': 0.03},
        'France': {'base_size': 85e9, 'growth_rate': 0.04, 'volatility': 0.02},
        'India': {'base_size': 75e9, 'growth_rate': 0.15, 'volatility': 0.08},
        'Italy': {'base_size': 60e9, 'growth_rate': 0.03, 'volatility': 0.02},
        'Brazil': {'base_size': 50e9, 'growth_rate': 0.08, 'volatility': 0.06},
        'Canada': {'base_size': 45e9, 'growth_rate': 0.05, 'volatility': 0.03},
        'South Korea': {'base_size': 40e9, 'growth_rate': 0.07, 'volatility': 0.04},
        'Spain': {'base_size': 35e9, 'growth_rate': 0.04, 'volatility': 0.03},
        'Australia': {'base_size': 30e9, 'growth_rate': 0.06, 'volatility': 0.03},
        'Mexico': {'base_size': 28e9, 'growth_rate': 0.09, 'volatility': 0.05},
        'Netherlands': {'base_size': 25e9, 'growth_rate': 0.04, 'volatility': 0.02},
        'Singapore': {'base_size': 20e9, 'growth_rate': 0.08, 'volatility': 0.04},
        'Indonesia': {'base_size': 18e9, 'growth_rate': 0.12, 'volatility': 0.07},
        'Sweden': {'base_size': 15e9, 'growth_rate': 0.05, 'volatility': 0.03},
        'Switzerland': {'base_size': 22e9, 'growth_rate': 0.03, 'volatility': 0.02},
        'Belgium': {'base_size': 12e9, 'growth_rate': 0.04, 'volatility': 0.02}
    }
    
    years = list(range(2020, 2029))  # Historical + forecast
    data = []
    
    for i, (country, specs) in enumerate(countries_data.items()):
        base_value = specs['base_size']
        annual_growth = specs['growth_rate']
        volatility = specs['volatility']
        
        for j, year in enumerate(years):
            # Calculate compound growth with some noise
            expected_value = base_value * (1 + annual_growth) ** j
            
            # Add market cycles and randomness
            cycle_factor = 1 + 0.1 * np.sin(j * 0.8)  # Market cycles
            noise_factor = 1 + random.gauss(0, volatility)
            
            # Special events (e.g., economic shocks)
            if year == 2020:  # Simulate COVID impact
                shock_factor = random.uniform(0.85, 0.95)
            elif year == 2021:  # Recovery
                shock_factor = random.uniform(1.05, 1.15)
            else:
                shock_factor = 1.0
            
            final_value = expected_value * cycle_factor * noise_factor * shock_factor
            
            data.append({
                'Country': country,
                'Year': year,
                'Value': max(final_value, 0),  # Ensure no negative values
                'idGeo': f"GEO_{i:02d}"
            })
    
    return pd.DataFrame(data)

class DemoConfigManager:
    """Minimal config manager for demo"""
    
    def get_project_info(self):
        return {
            'market_type': 'Technology Market',
            'project_name': 'Global Tech Market Analysis'
        }
    
    def get_output_directory(self):
        return os.path.join(os.path.dirname(__file__), 'demo_output')
    
    def get_column_mapping(self, mapping_type):
        return {
            'id_column': 'idGeo',
            'country_column': 'Country',
            'value_column': 'Value'
        }

def main():
    """Main demo application"""
    
    # Configure Streamlit page
    st.set_page_config(
        page_title="Enhanced Market Visualizations - Demo",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Create output directory
    os.makedirs('demo_output', exist_ok=True)
    
    # Initialize session state
    if 'distributed_market' not in st.session_state:
        st.session_state.distributed_market = generate_demo_data()
    
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = DemoConfigManager()
    
    # Sidebar controls
    st.sidebar.title("🎛️ Demo Controls")
    st.sidebar.markdown("---")
    
    # Data regeneration
    if st.sidebar.button("🔄 Regenerate Demo Data", help="Create new sample data"):
        st.session_state.distributed_market = generate_demo_data()
        st.rerun()
    
    # Show data info
    st.sidebar.markdown("### 📊 Current Dataset")
    data = st.session_state.distributed_market
    st.sidebar.metric("Countries", data['Country'].nunique())
    st.sidebar.metric("Years", f"{data['Year'].min()}-{data['Year'].max()}")
    st.sidebar.metric("Total Records", len(data))
    
    # Market overview
    latest_year = data['Year'].max()
    total_market = data[data['Year'] == latest_year]['Value'].sum()
    st.sidebar.metric("Total Market Size", f"${total_market/1e12:.1f}T")
    
    # Show sample data
    if st.sidebar.checkbox("Show Sample Data"):
        st.sidebar.dataframe(
            data.head(10)[['Country', 'Year', 'Value']], 
            use_container_width=True
        )
    
    # Main interface
    st.title("🚀 Enhanced Market Visualizations - Demo")
    st.markdown("**Professional Interactive Market Analytics Platform**")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Market Size", 
            f"${total_market/1e12:.1f}T",
            f"({latest_year})"
        )
    
    with col2:
        first_year = data['Year'].min()
        first_total = data[data['Year'] == first_year]['Value'].sum()
        growth = ((total_market/first_total) ** (1/(latest_year-first_year)) - 1) * 100
        st.metric(
            "CAGR", 
            f"{growth:.1f}%",
            f"({first_year}-{latest_year})"
        )
    
    with col3:
        top_country = data[data['Year'] == latest_year].nlargest(1, 'Value')['Country'].iloc[0]
        top_share = (data[data['Year'] == latest_year].nlargest(1, 'Value')['Value'].iloc[0] / total_market) * 100
        st.metric(
            "Market Leader", 
            top_country,
            f"{top_share:.1f}% share"
        )
    
    with col4:
        st.metric(
            "Countries", 
            data['Country'].nunique(),
            "Global coverage"
        )
    
    st.markdown("---")
    
    # Run the enhanced visualization interface
    render_enhanced_visualization_interface(st.session_state.config_manager)
    
    # Additional demo features
    st.markdown("---")
    st.markdown("### 🎯 Demo Features Showcase")
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        st.markdown("""
        **✨ New Visualization Features:**
        - 🏁 Animated bar chart races
        - 🌍 Interactive world maps
        - 📈 Modern area charts (non-stacked)
        - 🫧 Bubble evolution charts
        - 💧 Waterfall growth analysis
        - 🏆 Smart ranking tables
        - 📊 Executive dashboards
        """)
    
    with demo_col2:
        st.markdown("""
        **🎨 Enhanced User Experience:**
        - Professional color schemes
        - Interactive hover tooltips
        - Dynamic filtering controls
        - Country comparison tools
        - Mobile-responsive design
        - Export capabilities
        - Smart insights generation
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🚀 **Enhanced Market Visualizations** - "
        "Transforming market data into actionable insights with modern, "
        "interactive visualizations that engage users and drive decision-making."
    )

if __name__ == "__main__":
    main()