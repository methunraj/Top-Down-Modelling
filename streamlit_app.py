"""
Universal Market Forecasting Framework - Streamlit Application

This is the main entry point for the Streamlit web interface of the Universal Market
Forecasting Framework, providing an interactive interface for market forecasting and analysis.
"""

import os
import sys
import logging
import streamlit as st
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import framework components
from src.config.config_manager import ConfigurationManager
from src.market_analysis.market_analyzer import MarketAnalyzer
from src.distribution.market_distributor import MarketDistributor
from src.data_processing.data_loader import DataLoader

# Import Streamlit interface components
from src.streamlit.config_interface import (
    load_config_file,
    save_config_file,
    render_config_selector,
    render_config_interface
)
from src.streamlit.data_interface import (
    render_data_upload,
    create_test_data
)
from src.streamlit.test_data import generate_all_test_data
from src.streamlit.distribution_interface import (
    render_global_forecast_interface,
    render_distribution_interface
)
from src.streamlit.enhanced_visualization import (
    render_enhanced_visualization_interface
)
from src.streamlit.calibration_interface import render_calibration_interface

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Universal Market Forecasting Framework",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application state initialization
def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'global_forecast' not in st.session_state:
        st.session_state.global_forecast = None
    if 'country_historical' not in st.session_state:
        st.session_state.country_historical = None
    if 'indicators' not in st.session_state:
        st.session_state.indicators = {}
    if 'distributed_market' not in st.session_state:
        st.session_state.distributed_market = None
    if 'active_page' not in st.session_state:
        st.session_state.active_page = "Home"

# Initialize session state
initialize_session_state()

# Sidebar navigation
def sidebar_navigation():
    """Create sidebar navigation"""
    st.sidebar.title("Navigation")
    
    # Main navigation
    pages = {
        "Home": "🏠 Home",
        "Data Input": "📥 Data Input",
        "Global Forecasting": "🌐 Global Forecasting",
        "Market Distribution": "🔄 Market Distribution",
        "Visualization": "🚀 Enhanced Visualization",
        "Auto-Calibration": "🔄 Auto-Calibration",
        "Configuration": "⚙️ Configuration", 
        "Export": "📤 Export"
    }
    
    # Navigation selection
    selection = st.sidebar.radio("Go to", list(pages.values()))
    
    # Update active page
    for page_id, page_name in pages.items():
        if selection == page_name:
            st.session_state.active_page = page_id
    
    # Display current status
    st.sidebar.divider()
    st.sidebar.subheader("Current Status")
    
    # Show data status
    data_status = {
        "Configuration": "✅ Loaded" if st.session_state.config else "❌ Not loaded",
        "Global Forecast": "✅ Loaded" if st.session_state.global_forecast is not None else "❌ Not loaded",
        "Country Data": "✅ Loaded" if st.session_state.country_historical is not None else "❌ Not loaded",
        "Indicators": f"✅ {len(st.session_state.indicators)} loaded" if st.session_state.indicators else "❌ None loaded",
        "Results": "✅ Available" if st.session_state.distributed_market is not None else "❌ Not available"
    }
    
    for item, status in data_status.items():
        st.sidebar.text(f"{item}: {status}")

# Home page
def render_home_page():
    """Render the home/dashboard page"""
    st.title("Universal Market Forecasting Framework")
    st.subheader("Market-Agnostic Top-Down Forecasting")

    # Introduction
    st.markdown("""
    The Universal Market Forecasting Framework provides sophisticated market forecasting and distribution
    capabilities for any market type. This tool allows you to:

    - Generate global market forecasts using multiple methods
    - Distribute global forecasts across countries using data-driven algorithms
    - Apply growth constraints and smoothing for realistic projections
    - Utilize advanced techniques like causal inference and gradient harmonization
    - Auto-calibrate models based on historical accuracy
    - Visualize market size, growth rates, and market share
    - Export results in multiple formats
    """)

    # Test data options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Load Test Data (with sample indicators)", key="load_test_data_with_indicators"):
            with st.spinner("Generating test data with sample indicators..."):
                test_data = generate_all_test_data(use_sample_indicators=True)

                # Store in session state
                st.session_state.global_forecast = test_data['global_forecast']
                st.session_state.country_historical = test_data['country_historical']

                # Handle indicators
                if 'indicators' not in st.session_state:
                    st.session_state.indicators = {}

                for name, df in test_data['indicators'].items():
                    # Determine indicator type based on column names or name patterns
                    indicator_type = 'rank' if any('rank' in col.lower() or 'readiness' in col.lower() 
                                                 for col in df.columns if col not in ['idGeo', 'Country', 'Year']) else 'value'
                    st.session_state.indicators[name] = {
                        'data': df,
                        'meta': {
                            'name': name,
                            'type': indicator_type,
                            'weight': 'auto'
                        }
                    }

                st.success("Test data loaded successfully with sample indicators! You can now explore the application.")
    
    with col2:
        if st.button("Load Test Data (no indicators)", key="load_test_data_no_indicators"):
            with st.spinner("Generating test data without indicators..."):
                test_data = generate_all_test_data(use_sample_indicators=False)

                # Store in session state
                st.session_state.global_forecast = test_data['global_forecast']
                st.session_state.country_historical = test_data['country_historical']
                st.session_state.indicators = {}  # Clear any existing indicators

                st.success("Test data loaded successfully without indicators! The framework will use only historical market patterns.")

    # Quick start - setup your forecast
    st.header("Quick Start")

    # Four columns for main workflow steps
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("1. Data Input")
        st.markdown("""
        - Upload global market data
        - Import country historical data
        - Add indicator data (optional)
        """)
        if st.button("Go to Data Input"):
            st.session_state.active_page = "Data Input"
            st.rerun()
    
    with col2:
        st.subheader("2. Generate Forecast")
        st.markdown("""
        - Choose forecasting methods
        - Configure market distribution
        - Apply growth constraints
        """)
        if st.button("Go to Forecasting"):
            st.session_state.active_page = "Global Forecasting"
            st.rerun()
    
    with col3:
        st.subheader("3. Auto-Calibrate")
        st.markdown("""
        - Evaluate forecast accuracy
        - Auto-calibrate models
        - Apply targeted adjustments
        """)
        if st.button("Go to Auto-Calibration"):
            st.session_state.active_page = "Auto-Calibration"
            st.rerun()
    
    with col4:
        st.subheader("4. View Results")
        st.markdown("""
        - Visualize market data
        - Analyze growth patterns
        - Export results
        """)
        if st.button("Go to Visualization"):
            st.session_state.active_page = "Visualization"
            st.rerun()
    
    # Display project overview if a configuration is loaded
    if st.session_state.config:
        st.header("Project Overview")
        
        try:
            # Fixed: Handle both lambda function and direct project info access
            if hasattr(st.session_state.config, 'get_project_info'):
                project_info = st.session_state.config.get_project_info()
            else:
                # Handle direct project info storage
                project_info = st.session_state.config if isinstance(st.session_state.config, dict) else {}
            
            st.info(f"Project: {project_info.get('name', 'Unnamed Project')}")
            st.text(f"Market Type: {project_info.get('market_type', 'Unspecified')}")
            st.text(f"Version: {project_info.get('version', '1.0')}")
            
            # Show a summary of loaded data if available
            if st.session_state.distributed_market is not None:
                st.subheader("Market Forecast Summary")
                
                # Extract key statistics
                market_df = st.session_state.distributed_market
                total_countries = market_df['Country'].nunique()
                years = sorted(market_df['Year'].unique())
                first_year = min(years)
                last_year = max(years)
                
                # Display in a clean format
                st.markdown(f"""
                - Total Countries: **{total_countries}**
                - Forecast Period: **{first_year} - {last_year}**
                - Total Years: **{len(years)}**
                """)
                
                # Show a quick preview chart
                st.subheader("Market Size Preview")
                
                # Group by year and calculate total
                yearly_total = market_df.groupby('Year')['Value'].sum().reset_index()
                
                # Simple line chart of total market
                st.line_chart(yearly_total.set_index('Year'))
        
        except Exception as e:
            st.error(f"Error displaying project overview: {str(e)}")

# Data input page
def render_data_input_page():
    """Render the data input page"""
    st.title("Data Input")
    
    # Create tabs for different types of data input
    tab1, tab2, tab3 = st.tabs(["Global Market Data", "Country Historical Data", "Indicators"])
    
    # Tab 1: Global Market Data
    with tab1:
        st.header("Global Market Forecast Data")
        st.markdown("""
        Upload your global market forecast data in Excel or CSV format.
        This should contain yearly market size values for the entire market.
        """)
        
        # File uploader for global market data
        uploaded_file = st.file_uploader("Upload Global Market Data", type=["xlsx", "csv"], key="global_data")
        
        if uploaded_file:
            try:
                # Attempt to read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Display the data
                st.subheader("Preview")
                st.dataframe(df.head(10))
                
                # Column mapping
                st.subheader("Column Mapping")
                
                # Let user select the appropriate columns
                year_col = st.selectbox("Year Column", options=df.columns.tolist(), key="global_year")
                value_col = st.selectbox("Value Column", options=df.columns.tolist(), key="global_value")
                
                # Type column is optional
                include_type = st.checkbox("Include Type Column (Historical/Forecast)", value=False)
                if include_type:
                    type_col = st.selectbox("Type Column", options=df.columns.tolist(), key="global_type")
                
                # Save button
                if st.button("Save Global Market Data"):
                    st.session_state.global_forecast = df
                    
                    # Also save column mapping if we have a config
                    if st.session_state.config:
                        # TODO: Update config with column mapping
                        pass
                    
                    st.success("Global market data saved successfully!")
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Tab 2: Country Historical Data
    with tab2:
        st.header("Country Historical Data")
        st.markdown("""
        Upload your country historical data in Excel or CSV format.
        This should contain historical market values by country.
        """)
        
        # File uploader for country data
        uploaded_file = st.file_uploader("Upload Country Data", type=["xlsx", "csv"], key="country_data")
        
        if uploaded_file:
            try:
                # Attempt to read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Display the data
                st.subheader("Preview")
                st.dataframe(df.head(10))
                
                # Data format selection
                data_format = st.radio("Data Format", options=["Wide", "Long"], horizontal=True)
                
                # Column mapping
                st.subheader("Column Mapping")
                
                # Common columns regardless of format
                id_col = st.selectbox("Country ID Column", options=df.columns.tolist(), key="country_id")
                name_col = st.selectbox("Country Name Column", options=df.columns.tolist(), key="country_name")
                
                # Format-specific mapping
                if data_format == "Long":
                    year_col = st.selectbox("Year Column", options=df.columns.tolist(), key="country_year")
                    value_col = st.selectbox("Value Column", options=df.columns.tolist(), key="country_value")
                else:  # Wide format
                    st.info("For wide format, yearly data should be in columns named with years (e.g., '2020', '2021')")
                
                # Save button
                if st.button("Save Country Data"):
                    st.session_state.country_historical = df
                    
                    # Also save column mapping if we have a config
                    if st.session_state.config:
                        # TODO: Update config with column mapping
                        pass
                    
                    st.success("Country historical data saved successfully!")
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Tab 3: Indicators
    with tab3:
        st.header("Market Indicators")
        st.markdown("""
        Upload indicator data that can influence market distribution.
        Examples include GDP, population, technology adoption indices, etc.
        """)
        
        # Create a new indicator
        st.subheader("Add New Indicator")
        indicator_name = st.text_input("Indicator Name")
        
        # File uploader for indicator data
        uploaded_file = st.file_uploader("Upload Indicator Data", type=["xlsx", "csv"], key="indicator_data")
        
        if uploaded_file and indicator_name:
            try:
                # Attempt to read the file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Display the data
                st.subheader("Preview")
                st.dataframe(df.head(10))
                
                # Indicator type
                indicator_type = st.radio("Indicator Type", options=["Value", "Rank"], horizontal=True)
                
                # Weight configuration
                weight_config = st.radio("Weight Calculation", options=["Auto", "Manual"], horizontal=True)
                
                if weight_config == "Manual":
                    weight_value = st.slider("Indicator Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                
                # Column mapping
                st.subheader("Column Mapping")
                id_col = st.selectbox("Country ID Column", options=df.columns.tolist(), key=f"ind_{indicator_name}_id")
                
                # Save indicator
                if st.button("Add Indicator"):
                    # Save indicator to session state
                    st.session_state.indicators[indicator_name] = {
                        "data": df,
                        "type": indicator_type.lower(),
                        "weight": "auto" if weight_config == "Auto" else weight_value,
                        "id_column": id_col
                    }
                    
                    st.success(f"Indicator '{indicator_name}' added successfully!")
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Display currently loaded indicators
        if st.session_state.indicators:
            st.subheader("Loaded Indicators")
            
            for name, details in st.session_state.indicators.items():
                with st.expander(f"{name} ({details['type']})"):
                    st.dataframe(details["data"].head(5))
                    st.text(f"Weight: {details['weight']}")
                    
                    # Option to remove
                    if st.button("Remove", key=f"remove_{name}"):
                        del st.session_state.indicators[name]
                        st.success(f"Indicator '{name}' removed.")
                        st.rerun()

# Global forecasting page
def render_global_forecasting_page():
    """Render the global forecasting page"""
    st.title("Global Market Forecasting")
    
    # Check if we have the required data
    if st.session_state.global_forecast is None:
        st.warning("Please upload global market data first!")
        if st.button("Go to Data Input"):
            st.session_state.active_page = "Data Input"
            st.rerun()
        return
    
    # Split the page into two columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("Historical Data Visualization")
        
        # Display a chart of the historical global market data
        st.subheader("Global Market Trend")
        
        # In a real implementation, we would extract and plot the actual data
        # For now, just show a placeholder
        data = st.session_state.global_forecast
        st.line_chart(data)
        
        st.subheader("Market Growth Rate")
        # In a real implementation, calculate and show growth rates
        
    with col2:
        st.header("Forecasting Method")
        
        # Create a list of forecasting methods to choose from
        forecasting_methods = {
            "Statistical Methods": {
                "cagr": "Compound Annual Growth Rate (CAGR)",
                "moving_avg": "Moving Average",
                "exp_smoothing": "Exponential Smoothing",
                "arima": "ARIMA/SARIMA"
            },
            "Machine Learning Methods": {
                "prophet": "Prophet",
                "xgboost": "XGBoost",
                "lstm": "LSTM Neural Network"
            },
            "Technology Market Methods": {
                "bass": "Bass Diffusion Model",
                "gompertz": "Gompertz Curve",
                "tech_s_curve": "Technology S-Curve"
            },
            "Ensemble Methods": {
                "avg_ensemble": "Average Ensemble",
                "weighted_ensemble": "Weighted Ensemble"
            }
        }
        
        # Method selection
        method_category = st.selectbox("Method Category", options=list(forecasting_methods.keys()))
        method = st.selectbox("Forecasting Method", options=list(forecasting_methods[method_category].values()))
        
        # Parameters section - this would be dynamically generated based on the selected method
        st.subheader("Method Parameters")
        
        # Example parameters for different methods
        if "CAGR" in method:
            st.slider("Growth Rate (%)", min_value=-20.0, max_value=100.0, value=5.0, step=0.1)
            st.number_input("Base Year", min_value=2010, max_value=2023, value=2022)
        elif "Moving Average" in method:
            st.slider("Window Size", min_value=1, max_value=10, value=3)
        elif "Exponential Smoothing" in method:
            st.slider("Alpha", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            st.slider("Beta", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
            st.slider("Gamma", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        elif "ARIMA" in method:
            st.number_input("p (AR order)", min_value=0, max_value=5, value=1)
            st.number_input("d (Differencing)", min_value=0, max_value=2, value=1)
            st.number_input("q (MA order)", min_value=0, max_value=5, value=1)
        elif "Prophet" in method:
            st.slider("Changepoint Prior Scale", min_value=0.001, max_value=0.5, value=0.05, step=0.001)
            st.checkbox("Include Holiday Effects", value=False)
        elif "Diffusion" in method or "S-Curve" in method:
            st.slider("Innovation Factor", min_value=0.001, max_value=0.05, value=0.01, step=0.001)
            st.slider("Imitation Factor", min_value=0.1, max_value=0.5, value=0.3, step=0.01)
            st.number_input("Market Potential", min_value=1000, value=10000)
        
        # Forecast horizon
        st.subheader("Forecast Horizon")
        horizon = st.slider("Years to Forecast", min_value=1, max_value=20, value=5)
        
        # Generate forecast button
        if st.button("Generate Forecast"):
            # In a real implementation, this would call the appropriate forecasting method
            # For now, just show a success message
            st.success("Forecast generated successfully!")
    
    # Lower section for results
    st.header("Forecast Results")
    
    # In a real implementation, display forecasting results
    # For now, just a placeholder
    
    # Example tabs for different forecast views
    tab1, tab2, tab3 = st.tabs(["Forecast", "Evaluation Metrics", "Comparison"])
    
    with tab1:
        st.subheader("Market Forecast")
        st.text("The forecast will be shown here")
        
        # This would be a chart of the forecast
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['Historical', 'Forecast', 'Confidence Interval']
        )
        st.line_chart(chart_data)
    
    with tab2:
        st.subheader("Forecast Evaluation")
        
        # Example metrics table
        metrics_data = {
            "Metric": ["MAPE", "RMSE", "MAE", "Theil's U"],
            "Value": ["5.2%", "1245.3", "987.1", "0.15"]
        }
        st.table(pd.DataFrame(metrics_data))
    
    with tab3:
        st.subheader("Method Comparison")
        st.text("Compare results from different forecasting methods")
        
        # This would be a chart comparing multiple methods
        chart_data = pd.DataFrame(
            np.random.randn(20, 4),
            columns=['Historical', 'CAGR', 'ARIMA', 'Prophet']
        )
        st.line_chart(chart_data)

# Market distribution page
def render_market_distribution_page():
    """Render the market distribution page"""
    st.title("Market Distribution")
    
    # Check if we have the required data
    if st.session_state.global_forecast is None or st.session_state.country_historical is None:
        st.warning("Please upload global forecast data and country historical data first!")
        if st.button("Go to Data Input"):
            st.session_state.active_page = "Data Input"
            st.rerun()
        return
    
    # Market distribution parameters
    st.header("Distribution Parameters")
    
    # Create tabs for different aspects of distribution
    tab1, tab2, tab3, tab4 = st.tabs(["Tier Configuration", "Growth Constraints", "Indicators", "Smoothing"])
    
    # Tab 1: Tier Configuration
    with tab1:
        st.subheader("Market Tier Configuration")
        st.markdown("""
        Market tiers group countries based on their market share.
        This affects how growth rates are calculated and constrained.
        """)
        
        # Tier determination method
        tier_method = st.radio("Tier Determination Method", options=["Automatic", "Manual"], horizontal=True)
        
        if tier_method == "Automatic":
            # K-means parameters
            st.subheader("K-means Clustering Parameters")
            min_clusters = st.slider("Minimum Clusters", min_value=2, max_value=5, value=3)
            max_clusters = st.slider("Maximum Clusters", min_value=3, max_value=10, value=8)
            
            st.info("The system will use silhouette scores to determine the optimal number of clusters within this range.")
            
        else:  # Manual
            st.subheader("Manual Tier Thresholds")
            st.markdown("Define market share thresholds for each tier.")
            
            # Manual tier thresholds
            tier1_threshold = st.slider("Tier 1 Threshold (%)", min_value=1.0, max_value=30.0, value=10.0, step=0.1)
            tier2_threshold = st.slider("Tier 2 Threshold (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            tier3_threshold = st.slider("Tier 3 Threshold (%)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            
            # Manual tier assignment
            st.subheader("Manual Country Assignment")
            st.markdown("Optionally, override tier assignment for specific countries.")
            
            # In a real implementation, we would list countries here for manual assignment
            # For example:
            st.text("Country: United States, Current tier: 1")
    
    # Tab 2: Growth Constraints
    with tab2:
        st.subheader("Growth Constraints")
        st.markdown("""
        Growth constraints ensure realistic growth patterns in the forecast.
        Different market tiers can have different constraints.
        """)
        
        # Determination method
        constraint_method = st.radio("Constraint Determination", options=["Automatic", "Manual"], horizontal=True)
        
        if constraint_method == "Manual":
            # Manual constraints
            st.subheader("Manual Growth Constraints")
            
            # Tier 1 (Market Leaders)
            st.markdown("#### Tier 1 (Market Leaders)")
            tier1_max = st.slider("Maximum Growth Rate (%)", min_value=10.0, max_value=80.0, value=35.0, step=1.0, key="tier1_max")
            tier1_min = st.slider("Minimum Growth Rate (%)", min_value=-40.0, max_value=0.0, value=-15.0, step=1.0, key="tier1_min")
            
            # Tier 2 (Established Markets)
            st.markdown("#### Tier 2 (Established Markets)")
            tier2_max = st.slider("Maximum Growth Rate (%)", min_value=15.0, max_value=100.0, value=40.0, step=1.0, key="tier2_max")
            tier2_min = st.slider("Minimum Growth Rate (%)", min_value=-50.0, max_value=0.0, value=-20.0, step=1.0, key="tier2_min")
            
            # Tier 3 (Emerging Markets)
            st.markdown("#### Tier 3 (Emerging Markets)")
            tier3_max = st.slider("Maximum Growth Rate (%)", min_value=20.0, max_value=120.0, value=45.0, step=1.0, key="tier3_max")
            tier3_min = st.slider("Minimum Growth Rate (%)", min_value=-60.0, max_value=0.0, value=-25.0, step=1.0, key="tier3_min")
            
            # Apply scaling by market size
            st.checkbox("Apply Scaling by Market Size", value=True, help="Allow smaller markets to grow faster")
        
        else:  # Automatic
            st.info("Growth constraints will be calculated automatically based on historical data.")
            
            # Constraint factor
            st.slider("Constraint Factor", min_value=0.5, max_value=1.5, value=1.0, step=0.1, 
                     help="Values above 1.0 allow wider growth ranges, values below 1.0 enforce tighter constraints")
    
    # Tab 3: Indicators
    with tab3:
        st.subheader("Indicator Configuration")
        
        if not st.session_state.indicators:
            st.warning("No indicators added. Indicators can enhance distribution accuracy.")
            if st.button("Add Indicators"):
                st.session_state.active_page = "Data Input"
                st.rerun()
        else:
            st.markdown("""
            Configure how indicators influence market distribution.
            Adjust weights and transformations for each indicator.
            """)
            
            # Weight transformation
            st.subheader("Weight Transformation")
            transformation = st.selectbox("Transformation Method", 
                                         options=["Logarithmic", "Squared", "Sigmoid", "Linear"])
            
            # Significance method
            significance = st.selectbox("Significance Method", options=["Continuous", "Stepped"])
            
            # Individual indicator settings
            st.subheader("Indicator Weights")
            
            for name in st.session_state.indicators:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{name}**")
                with col2:
                    if st.session_state.indicators[name]["weight"] == "auto":
                        st.text("Auto")
                    else:
                        new_weight = st.number_input(f"Weight", 
                                                   min_value=0.0, 
                                                   max_value=1.0, 
                                                   value=float(st.session_state.indicators[name]["weight"]),
                                                   step=0.01,
                                                   key=f"weight_{name}")
                        # Update weight in session state
                        st.session_state.indicators[name]["weight"] = new_weight
    
    # Tab 4: Smoothing
    with tab4:
        st.subheader("Smoothing Parameters")
        st.markdown("""
        Smoothing ensures realistic growth patterns by reducing volatility in the forecast.
        Different market tiers can have different smoothing parameters.
        """)
        
        # Enable smoothing
        enable_smoothing = st.checkbox("Enable Smoothing", value=True)
        
        if enable_smoothing:
            # Tier-specific smoothing
            st.subheader("Tier-Specific Smoothing Parameters")
            
            # Tier 1 (Market Leaders)
            st.markdown("#### Tier 1 (Market Leaders)")
            tier1_window = st.slider("Window Size", min_value=1, max_value=7, value=3, key="smooth_t1_window")
            tier1_convergence = st.slider("Target Growth Rate (%)", min_value=5.0, max_value=25.0, value=15.0, step=0.5, key="smooth_t1_target")
            
            # Tier 2 (Established Markets)
            st.markdown("#### Tier 2 (Established Markets)")
            tier2_window = st.slider("Window Size", min_value=1, max_value=7, value=3, key="smooth_t2_window")
            tier2_convergence = st.slider("Target Growth Rate (%)", min_value=10.0, max_value=30.0, value=20.0, step=0.5, key="smooth_t2_target")
            
            # Tier 3 (Emerging Markets)
            st.markdown("#### Tier 3 (Emerging Markets)")
            tier3_window = st.slider("Window Size", min_value=1, max_value=9, value=5, key="smooth_t3_window")
            tier3_convergence = st.slider("Target Growth Rate (%)", min_value=15.0, max_value=35.0, value=25.0, step=0.5, key="smooth_t3_target")
            
            # Convergence rate
            convergence_rate = st.slider("Convergence Rate", min_value=0.1, max_value=0.5, value=0.25, step=0.01,
                                        help="How quickly growth rates converge to target values")
    
    # Redistribution settings
    st.header("Redistribution Settings")
    
    # Enable redistribution from specific year
    enable_redistribution = st.checkbox("Enable Redistribution from Specific Year", value=False,
                                      help="Preserve historical data exactly before a specific year")
    
    if enable_redistribution:
        redistribution_year = st.number_input("Redistribution Start Year", min_value=2010, max_value=2023, value=2020)
        st.info(f"Historical data before {redistribution_year} will be preserved exactly as-is.")
    
    # Run distribution
    if st.button("Run Market Distribution"):
        # Fixed: Add proper validation and processing
        if st.session_state.global_forecast is None or st.session_state.country_historical is None:
            st.error("Please ensure both global forecast and country historical data are loaded before running distribution.")
            return
        
        try:
            with st.spinner("Running market distribution..."):
                # In a real implementation, this would call the MarketDistributor
                # For demonstration, we'll create a processed version of the data
                distributed_result = st.session_state.country_historical.copy()
                
                # Add some basic validation and processing
                if 'Year' not in distributed_result.columns or 'Value' not in distributed_result.columns:
                    st.error("Country historical data must contain 'Year' and 'Value' columns")
                    return
                
                # Simple processing example - ensure non-negative values
                distributed_result['Value'] = distributed_result['Value'].clip(lower=0)
                
                st.session_state.distributed_market = distributed_result
                st.success("Market distribution completed successfully!")
                
        except Exception as e:
            st.error(f"Error during market distribution: {str(e)}")
            return
        
        # Show a summary of the result
        st.header("Distribution Result Summary")
        st.text("Total Countries: 20")
        st.text("Forecast Years: 2023-2027")
        
        # Option to view detailed results
        if st.button("View Detailed Results"):
            st.session_state.active_page = "Visualization"
            st.rerun()


# Configuration page
def render_configuration_page():
    """Render the configuration page"""
    st.title("Configuration")
    
    # Create tabs for different configuration sections
    tab1, tab2, tab3, tab4 = st.tabs(["Project Settings", "Data Sources", "Market Distribution", "Visualization"])
    
    # Tab 1: Project Settings
    with tab1:
        st.header("Project Settings")
        
        # Project information
        project_name = st.text_input("Project Name", value="Universal Market Forecast")
        market_type = st.text_input("Market Type", value="Technology Market")
        version = st.text_input("Version", value="1.0")
        description = st.text_area("Description", value="Universal market forecasting project")
    
    # Tab 2: Data Sources
    with tab2:
        st.header("Data Source Configuration")
        
        # Global forecast settings
        st.subheader("Global Forecast Settings")
        global_path = st.text_input("Global Forecast Path", value="data/global_forecast.xlsx")
        global_sheet = st.text_input("Sheet Name", value="Sheet1")
        
        # Column mapping
        st.subheader("Column Mapping")
        
        # Global forecast mapping
        st.markdown("**Global Forecast Mapping**")
        col1, col2, col3 = st.columns(3)
        with col1:
            global_year = st.text_input("Year Column", value="Year")
        with col2:
            global_value = st.text_input("Value Column", value="Value")
        with col3:
            global_type = st.text_input("Type Column", value="Type")
        
        # Country historical mapping
        st.markdown("**Country Historical Mapping**")
        col1, col2, col3 = st.columns(3)
        with col1:
            country_id = st.text_input("ID Column", value="idGeo")
        with col2:
            country_name = st.text_input("Country Column", value="Country")
        with col3:
            country_vertical = st.text_input("Vertical Column", value="nameVertical")
    
    # Tab 3: Market Distribution
    with tab3:
        st.header("Market Distribution Settings")
        
        # Tier determination
        st.subheader("Tier Determination")
        tier_method = st.radio("Tier Determination Method", options=["auto", "manual"], horizontal=True)
        
        if tier_method == "manual":
            st.markdown("**Manual Tier Settings**")
            st.text_area("Tier Configuration (YAML format)", value="""
tier_1:
  share_threshold: 5.0
  description: "Market Leaders"
tier_2:
  share_threshold: 1.0
  description: "Established Markets"
tier_3:
  share_threshold: 0.1
  description: "Emerging Markets"
            """)
        
        # Growth constraints
        st.subheader("Growth Constraints")
        constraint_method = st.radio("Constraint Determination Method", options=["auto", "manual"], horizontal=True)
        
        if constraint_method == "manual":
            col1, col2 = st.columns(2)
            with col1:
                max_growth = st.number_input("Maximum Growth Rate (%)", min_value=10.0, max_value=100.0, value=60.0)
            with col2:
                min_growth = st.number_input("Minimum Growth Rate (%)", min_value=-50.0, max_value=0.0, value=-30.0)
            
            scaling = st.checkbox("Apply Scaling by Market Size", value=True)
        
        # Smoothing settings
        st.subheader("Smoothing Settings")
        smoothing_enabled = st.checkbox("Enable Smoothing", value=True)
        
        if smoothing_enabled:
            st.text_area("Smoothing Configuration (YAML format)", value="""
tier_smoothing:
  tier_1:
    window: 3
    min_periods: 1
    max_growth: 35
    min_growth: -15
    target_growth: 15
  tier_2:
    window: 3
    min_periods: 1
    max_growth: 40
    min_growth: -20
    target_growth: 20
  tier_3:
    window: 5
    min_periods: 1
    max_growth: 45
    min_growth: -25
    target_growth: 25
convergence_rate: 0.25
            """)
    
    # Tab 4: Visualization
    with tab4:
        st.header("Visualization Settings")
        
        # Output settings
        st.subheader("Output Settings")
        output_path = st.text_input("Output Directory", value="data/output/")
        
        # Output formats
        formats = st.multiselect("Output Formats", options=["xlsx", "csv", "json"], default=["xlsx", "csv"])
        
        # Visualization types
        st.subheader("Visualization Types")
        
        # Market size visualization
        st.markdown("**Market Size Visualization**")
        market_size_title = st.text_input("Title Template", value="${market_type} Market Size by Country")
        market_size_top_n = st.number_input("Top N Countries", min_value=5, max_value=20, value=10)
        
        # Growth rate visualization
        st.markdown("**Growth Rate Visualization**")
        growth_title = st.text_input("Title Template", value="${market_type} Growth Rate Analysis", key="growth_title")
        show_yoy = st.checkbox("Show Year-over-Year Growth", value=True)
        show_cagr = st.checkbox("Show CAGR", value=True)
    
    # Save configuration button
    if st.button("Save Configuration"):
        # In a real implementation, this would create a ConfigurationManager and save the config
        # For now, just show a success message
        st.success("Configuration saved successfully!")
        
        # Create a dummy config in session state
        st.session_state.config = {}
        
        # Populate with entered values (in a real implementation, this would be a ConfigurationManager)
        project_info = {
            "name": project_name,
            "market_type": market_type,
            "version": version,
            "description": description
        }
        
        # Fixed: Store project info directly instead of lambda function to avoid serialization issues
        st.session_state.config = project_info
        st.session_state.config_type = "project_info"

# Export page
def render_export_page():
    """Render the export page"""
    st.title("Export Results")
    
    # Fixed: Handle import errors gracefully
    try:
        from src.streamlit.export_handler import export_market_data, export_visualizations, export_report
    except ImportError as e:
        st.error(f"Export functionality not available: {e}")
        st.info("Please ensure all required export modules are installed.")
        return

    # Check if we have distributed market data
    if st.session_state.distributed_market is None:
        st.warning("No market data available for export.")

        # Show options based on what data we have
        if st.session_state.global_forecast is not None:
            if st.button("Go to Global Forecasting"):
                st.session_state.active_page = "Global Forecasting"
                st.rerun()

        if st.session_state.global_forecast is not None and st.session_state.country_historical is not None:
            if st.button("Go to Market Distribution"):
                st.session_state.active_page = "Market Distribution"
                st.rerun()

        return

    # Data export options
    st.header("Data Export Options")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Export Format")
        export_formats = st.multiselect("Select Formats",
                                       options=["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"],
                                       default=["Excel (.xlsx)"])

        # Excel-specific options
        excel_options = {}
        if "Excel (.xlsx)" in export_formats:
            excel_options["create_summary"] = st.checkbox("Create Summary Sheet", value=True)
            excel_options["include_charts"] = st.checkbox("Include Charts", value=True)
            excel_options["apply_formatting"] = st.checkbox("Apply Formatting", value=True)

    with col2:
        st.subheader("Data Selection")

        # What to include
        include_market_values = st.checkbox("Include Market Values", value=True)
        include_market_shares = st.checkbox("Include Market Shares", value=True)
        include_growth_rates = st.checkbox("Include Growth Rates", value=True)
        include_metadata = st.checkbox("Include Forecast Metadata", value=True)

        # Year range
        year_range = st.slider("Year Range", min_value=2020, max_value=2030, value=(2020, 2030))

    # Visualization export
    st.header("Visualization Export")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Chart Selection")
        chart_types = []
        if st.checkbox("Market Size Chart", value=True):
            chart_types.append("Market Size Chart")
        if st.checkbox("Growth Rate Chart", value=True):
            chart_types.append("Growth Rate Chart")
        if st.checkbox("Market Share Chart", value=True):
            chart_types.append("Market Share Chart")
        if st.checkbox("Regional Analysis Chart", value=False):
            chart_types.append("Regional Analysis Chart")

    with col2:
        st.subheader("Export Format")
        chart_format = st.radio("Image Format", options=["PNG", "PDF", "SVG", "HTML"], horizontal=True)

        chart_dpi = 300
        if chart_format in ["PNG", "PDF"]:
            chart_dpi = st.slider("Resolution (DPI)", min_value=72, max_value=600, value=300, step=72)

    # Report generation
    st.header("Report Generation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Report Format")
        report_format = st.radio("Format", options=["PDF", "PowerPoint", "HTML"], horizontal=True)

        report_options = {}
        if report_format == "PowerPoint":
            report_options["include_speaker_notes"] = st.checkbox("Include Speaker Notes", value=False)
        elif report_format == "PDF":
            report_options["include_toc"] = st.checkbox("Include Table of Contents", value=True)

    with col2:
        st.subheader("Report Content")
        report_content = {}
        report_content["Executive Summary"] = st.checkbox("Executive Summary", value=True)
        report_content["Methodology Description"] = st.checkbox("Methodology Description", value=True)
        report_content["Detailed Market Analysis"] = st.checkbox("Detailed Market Analysis", value=True)
        report_content["Country Profiles"] = st.checkbox("Country Profiles", value=False)
        report_content["Appendix with Raw Data"] = st.checkbox("Appendix with Raw Data", value=False)

    # Export button
    output_dir = st.text_input("Output Directory", value="exports/")

    if st.button("Export Now"):
        with st.spinner("Exporting data and visualizations..."):
            try:
                # Ensure output directory exists
                import os
                os.makedirs(output_dir, exist_ok=True)

                # Get the distributed market data
                distributed_market = st.session_state.distributed_market

                # Export data files
                data_files = export_market_data(
                    distributed_market=distributed_market,
                    output_dir=output_dir,
                    export_formats=export_formats,
                    include_market_values=include_market_values,
                    include_market_shares=include_market_shares,
                    include_growth_rates=include_growth_rates,
                    include_metadata=include_metadata,
                    year_range=year_range
                )

                # Export visualizations
                if chart_types:
                    chart_files = export_visualizations(
                        distributed_market=distributed_market,
                        output_dir=output_dir,
                        chart_types=chart_types,
                        chart_format=chart_format,
                        dpi=chart_dpi
                    )
                else:
                    chart_files = {}

                # Export report
                report_files = export_report(
                    distributed_market=distributed_market,
                    output_dir=output_dir,
                    report_format=report_format,
                    report_content=report_content
                )

                # Combine all exported files
                all_files = {**data_files, **chart_files, **report_files}

                # Show success message and file paths
                st.success("Export completed successfully!")

                # Show exported file paths
                for file_type, file_path in all_files.items():
                    st.code(f"Exported to: {file_path}")

            except Exception as e:
                st.error(f"Error during export: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Render the appropriate page based on active_page
def render_current_page():
    """Render the currently active page"""
    # Get or create ConfigurationManager
    if 'config_manager' not in st.session_state:
        # Look for a configuration file
        config_path = 'config/market_config.yaml'
        if os.path.exists(config_path):
            try:
                config_manager = load_config_file(config_path)
                st.session_state.config_manager = config_manager
                st.session_state.config_path = config_path
            except Exception as e:
                st.error(f"Error loading configuration: {str(e)}")
                # Create empty config manager
                config_manager = ConfigurationManager()
                st.session_state.config_manager = config_manager
        else:
            # Create empty config manager
            config_manager = ConfigurationManager()
            st.session_state.config_manager = config_manager
    else:
        config_manager = st.session_state.config_manager

    # Render the active page
    if st.session_state.active_page == "Home":
        render_home_page()
    elif st.session_state.active_page == "Data Input":
        data = render_data_upload(config_manager)

        # Store data in session state if returned
        if 'global_forecast' in data and data['global_forecast'] is not None:
            st.session_state.global_forecast = data['global_forecast']

        if 'country_historical' in data and data['country_historical'] is not None:
            st.session_state.country_historical = data['country_historical']

        if 'indicators' in data and data['indicators']:
            if 'indicators' not in st.session_state:
                st.session_state.indicators = {}

            for name, indicator in data['indicators'].items():
                st.session_state.indicators[name] = indicator
    elif st.session_state.active_page == "Global Forecasting":
        forecast_config = render_global_forecast_interface()

        # Store forecast data in session state if returned
        if forecast_config is not None and 'result' in forecast_config and forecast_config['result'] is not None:
            st.session_state.global_forecast = forecast_config['result']
    elif st.session_state.active_page == "Market Distribution":
        # Create market distributor if needed
        if 'market_analyzer' not in st.session_state or st.session_state.market_analyzer is None:
            try:
                from src.market_analysis.market_analyzer import MarketAnalyzer
                
                # Get config path
                config_path = getattr(st.session_state, 'config_path', None)
                
                # If we have a valid config path, use it to create MarketAnalyzer
                if config_path and os.path.exists(config_path):
                    market_analyzer = MarketAnalyzer(config_path)
                    st.session_state.market_analyzer = market_analyzer
                else:
                    # Create a temporary config file from the current config_manager
                    temp_config_path = 'config/temp_config.yaml'
                    try:
                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
                        # Save current config to temp file
                        if hasattr(config_manager, 'config'):
                            config_manager.save_config(temp_config_path)
                            # Create MarketAnalyzer with the temp config
                            market_analyzer = MarketAnalyzer(temp_config_path)
                            st.session_state.market_analyzer = market_analyzer
                        else:
                            st.error("No valid configuration available")
                            market_analyzer = None
                    except Exception as e:
                        st.error(f"Error saving temporary configuration: {str(e)}")
                        market_analyzer = None
            except Exception as e:
                st.error(f"Error creating market analyzer: {str(e)}")
                market_analyzer = None
        else:
            market_analyzer = st.session_state.market_analyzer
            
        # Get market distributor from market analyzer
        market_distributor = None
        if market_analyzer and hasattr(market_analyzer, 'market_distributor'):
            market_distributor = market_analyzer.market_distributor
            
        # Render distribution interface
        distribution_config = render_distribution_interface(market_distributor, config_manager)
    elif st.session_state.active_page == "Visualization":
        render_enhanced_visualization_interface(config_manager)
    elif st.session_state.active_page == "Auto-Calibration":
        # Create MarketAnalyzer instance if needed
        if 'market_analyzer' not in st.session_state or st.session_state.market_analyzer is None:
            try:
                from src.market_analysis.market_analyzer import MarketAnalyzer
                
                # Get config path
                config_path = getattr(st.session_state, 'config_path', None)
                
                # If we have a valid config path, use it to create MarketAnalyzer
                if config_path and os.path.exists(config_path):
                    market_analyzer = MarketAnalyzer(config_path)
                    st.session_state.market_analyzer = market_analyzer
                else:
                    # Create a temporary config file from the current config_manager
                    temp_config_path = 'config/temp_config.yaml'
                    try:
                        # Save current config to temp file
                        if hasattr(config_manager, 'config'):
                            # Ensure the directory exists
                            os.makedirs(os.path.dirname(temp_config_path), exist_ok=True)
                            config_manager.save_config(temp_config_path)
                            
                            # Create MarketAnalyzer with the temp config
                            market_analyzer = MarketAnalyzer(temp_config_path)
                            st.session_state.market_analyzer = market_analyzer
                        else:
                            st.error("No valid configuration available")
                            market_analyzer = None
                    except Exception as e:
                        st.error(f"Error saving temporary configuration: {str(e)}")
                        market_analyzer = None
            except Exception as e:
                st.error(f"Error creating market analyzer: {str(e)}")
                market_analyzer = None
        else:
            market_analyzer = st.session_state.market_analyzer
        
        # Render calibration interface
        calibration_config = render_calibration_interface(config_manager, market_analyzer)
        
        # Update configuration if returned
        if calibration_config:
            # Update market_distribution.calibration in config
            current_config = config_manager.config if hasattr(config_manager, 'config') else {}
            
            # Ensure market_distribution key exists
            if 'market_distribution' not in current_config:
                current_config['market_distribution'] = {}
            
            # Update calibration settings
            current_config['market_distribution']['calibration'] = calibration_config
            
            # Update config manager
            config_manager.config = current_config
            st.session_state.config_manager = config_manager
            
            # Save to file if path exists
            if 'config_path' in st.session_state:
                try:
                    save_config_file(current_config, st.session_state.config_path)
                except Exception as e:
                    st.error(f"Error saving configuration: {str(e)}")
    elif st.session_state.active_page == "Configuration":
        # Get current configuration data
        if hasattr(config_manager, 'config'):
            config_data = config_manager.config
        else:
            config_data = None

        # Render configuration interface
        updated_config = render_config_interface(config_data)

        # Save configuration if different
        if updated_config != config_data:
            # Update config manager
            config_manager.config = updated_config
            st.session_state.config_manager = config_manager

            # Save to file if path exists
            if 'config_path' in st.session_state:
                try:
                    save_config_file(updated_config, st.session_state.config_path)
                    st.success(f"Configuration saved to {st.session_state.config_path}")
                except Exception as e:
                    st.error(f"Error saving configuration: {str(e)}")
    elif st.session_state.active_page == "Export":
        render_export_page()

# Main function
def main():
    """Main function to run the Streamlit application"""
    # Create sidebar navigation
    sidebar_navigation()

    # Render the current page
    render_current_page()

# Run the application
if __name__ == "__main__":
    main()