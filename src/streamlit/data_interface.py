"""
Streamlit Data Interface Module

This module provides components and utilities for data upload, validation,
and preparation through the Streamlit interface.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import io

from src.data_processing.data_loader import DataLoader
from src.config.config_manager import ConfigurationManager

# Configure logger
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_global_forecast(df: pd.DataFrame, config_manager: ConfigurationManager) -> Tuple[bool, str]:
    """
    Validate global forecast data against expected format.
    
    Args:
        df: DataFrame containing global forecast data
        config_manager: Configuration manager instance
        
    Returns:
        Tuple containing (is_valid, error_message)
    """
    # Get column mapping
    try:
        global_mapping = config_manager.get_column_mapping('global_forecast')
        year_col = global_mapping.get('year_column', 'Year')
        value_col = global_mapping.get('value_column', 'Value')
        type_col = global_mapping.get('type_column', 'Type')
    except Exception as e:
        return False, f"Error getting column mapping: {str(e)}"
    
    # Check that required columns exist
    missing_columns = []
    for col, col_name in [(year_col, "Year"), (value_col, "Value"), (type_col, "Type")]:
        if col not in df.columns:
            missing_columns.append(f"{col_name} column '{col}'")
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check that year column contains valid years
    try:
        years = df[year_col].astype(int)
        if not all(1900 <= year <= 2100 for year in years):
            return False, f"Year column contains values outside valid range (1900-2100)"
    except Exception:
        return False, f"Year column does not contain valid years"
    
    # Check that value column contains numeric values
    try:
        values = df[value_col].astype(float)
        if values.isna().any():
            return False, f"Value column contains non-numeric or missing values"
    except Exception:
        return False, f"Value column does not contain valid numeric values"
    
    # Check that type column contains expected values
    expected_types = {'Historical', 'Forecast'}
    actual_types = set(df[type_col].unique())
    if not actual_types.issubset(expected_types) and not expected_types.issubset(actual_types):
        return False, f"Type column should contain 'Historical' and 'Forecast' values"
    
    return True, ""


def validate_country_historical(df: pd.DataFrame, config_manager: ConfigurationManager) -> Tuple[bool, str]:
    """
    Validate country historical data against expected format.
    
    Args:
        df: DataFrame containing country historical data
        config_manager: Configuration manager instance
        
    Returns:
        Tuple containing (is_valid, error_message)
    """
    # Get column mapping
    try:
        column_mapping = config_manager.get_column_mapping('country_historical')
        id_col = column_mapping.get('id_column', 'idGeo')
        country_col = column_mapping.get('country_column', 'Country')
    except Exception as e:
        return False, f"Error getting column mapping: {str(e)}"
    
    # Check that required columns exist
    missing_columns = []
    for col, col_name in [(id_col, "ID"), (country_col, "Country")]:
        if col not in df.columns:
            missing_columns.append(f"{col_name} column '{col}'")
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check format based on data_sources settings
    try:
        data_sources = config_manager.get_config_section('data_sources')
        data_format = data_sources.get('country_historical', {}).get('format', 'wide')
    except Exception:
        # Default to wide format if not specified
        data_format = 'wide'
    
    if data_format == 'wide':
        # Check for year columns
        year_cols = [col for col in df.columns if str(col).isdigit() or 
                     (isinstance(col, str) and col.isdigit())]
        
        if not year_cols:
            return False, f"No year columns found in wide format data"
        
        # Check that year columns contain numeric values
        for year_col in year_cols:
            try:
                values = df[year_col].astype(float)
                if values.isna().all():
                    return False, f"Year column '{year_col}' contains only missing values"
            except Exception:
                return False, f"Year column '{year_col}' does not contain valid numeric values"
    else:  # long format
        # Check for year and value columns
        year_col = 'Year'  # Default, should be configurable
        value_col = 'Value'  # Default, should be configurable
        
        if year_col not in df.columns:
            return False, f"Missing Year column '{year_col}' for long format data"
        
        if value_col not in df.columns:
            return False, f"Missing Value column '{value_col}' for long format data"
        
        # Check that year column contains valid years
        try:
            years = df[year_col].astype(int)
            if not all(1900 <= year <= 2100 for year in years):
                return False, f"Year column contains values outside valid range (1900-2100)"
        except Exception:
            return False, f"Year column does not contain valid years"
        
        # Check that value column contains numeric values
        try:
            values = df[value_col].astype(float)
            if values.isna().all():
                return False, f"Value column contains only missing values"
        except Exception:
            return False, f"Value column does not contain valid numeric values"
    
    return True, ""


def validate_indicator_data(df: pd.DataFrame, indicator_type: str) -> Tuple[bool, str]:
    """
    Validate indicator data against expected format.
    
    Args:
        df: DataFrame containing indicator data
        indicator_type: Type of indicator ('value' or 'rank')
        
    Returns:
        Tuple containing (is_valid, error_message)
    """
    # Check that ID and Country columns exist (these are commonly expected)
    id_col = 'idGeo'  # Default, should be configurable
    country_col = 'Country'  # Default, should be configurable
    
    missing_columns = []
    for col, col_name in [(id_col, "ID"), (country_col, "Country")]:
        if col not in df.columns:
            missing_columns.append(f"{col_name} column '{col}'")
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Check for numeric data columns
    data_cols = [col for col in df.columns if col not in [id_col, country_col]]
    
    if not data_cols:
        return False, f"No data columns found beyond ID and Country columns"
    
    # Check data values based on indicator type
    if indicator_type == 'value':
        # For value indicators, data should be numeric
        for col in data_cols:
            try:
                values = df[col].astype(float)
                if values.isna().all():
                    return False, f"Data column '{col}' contains only missing values"
            except Exception:
                return False, f"Data column '{col}' does not contain valid numeric values"
    else:  # rank indicator
        # For rank indicators, data should be integers or can be ordinals
        # We'll be permissive here and just check that we have some values
        for col in data_cols:
            if df[col].isna().all():
                return False, f"Data column '{col}' contains only missing values"
    
    return True, ""


def convert_wide_to_long(df: pd.DataFrame, config_manager: ConfigurationManager) -> pd.DataFrame:
    """
    Convert wide format country data to long format.
    
    Args:
        df: DataFrame in wide format
        config_manager: Configuration manager instance
        
    Returns:
        DataFrame in long format
    """
    # Get column mapping
    try:
        column_mapping = config_manager.get_column_mapping('country_historical')
        id_col = column_mapping.get('id_column', 'idGeo')
        country_col = column_mapping.get('country_column', 'Country')
    except Exception as e:
        logger.error(f"Error getting column mapping: {str(e)}")
        id_col = 'idGeo'
        country_col = 'Country'
    
    # Identify non-year columns (metadata columns)
    year_cols = [col for col in df.columns if str(col).isdigit() or 
                (isinstance(col, str) and col.isdigit())]
    
    meta_cols = [col for col in df.columns if col not in year_cols]
    
    # Convert to long format
    long_df = pd.melt(
        df,
        id_vars=meta_cols,
        value_vars=year_cols,
        var_name='Year',
        value_name='Value'
    )
    
    # Ensure Year is numeric
    long_df['Year'] = pd.to_numeric(long_df['Year'])
    
    return long_df


def convert_long_to_wide(df: pd.DataFrame, config_manager: ConfigurationManager) -> pd.DataFrame:
    """
    Convert long format country data to wide format.
    
    Args:
        df: DataFrame in long format
        config_manager: Configuration manager instance
        
    Returns:
        DataFrame in wide format
    """
    # Get column mapping
    try:
        column_mapping = config_manager.get_column_mapping('country_historical')
        id_col = column_mapping.get('id_column', 'idGeo')
        country_col = column_mapping.get('country_column', 'Country')
    except Exception as e:
        logger.error(f"Error getting column mapping: {str(e)}")
        id_col = 'idGeo'
        country_col = 'Country'
    
    # Identify year and value columns
    year_col = 'Year'  # Default, should be configurable
    value_col = 'Value'  # Default, should be configurable
    
    # Get metadata columns (all except year and value)
    meta_cols = [col for col in df.columns if col not in [year_col, value_col]]
    
    # Convert to wide format
    wide_df = df.pivot_table(
        index=meta_cols,
        columns=year_col,
        values=value_col
    ).reset_index()
    
    return wide_df


def render_data_upload(config_manager: ConfigurationManager) -> Dict[str, Any]:
    """
    Render data upload interface for global and country data.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Dictionary containing uploaded data
    """
    st.title("Data Upload")
    
    # Initialize data dictionary
    data = {
        'global_forecast': None,
        'country_historical': None,
        'indicators': {}
    }
    
    # Create tabs for different data sources
    tab1, tab2, tab3 = st.tabs(["Global Market Data", "Country Historical Data", "Indicators"])
    
    # Tab 1: Global Market Data
    with tab1:
        st.header("Global Market Forecast Data")
        st.markdown("""
        Upload your global market forecast data in Excel or CSV format.
        This should contain yearly market size values for the entire market.
        """)
        
        # Get expected path from configuration
        try:
            global_path = config_manager.get_config_section('data_sources').get('global_forecast', {}).get('path', '')
            st.info(f"Configured path: {global_path}")
        except Exception:
            global_path = ''
        
        # Get expected column mapping from configuration
        try:
            global_mapping = config_manager.get_column_mapping('global_forecast')
            year_col = global_mapping.get('year_column', 'Year')
            value_col = global_mapping.get('value_column', 'Value')
            type_col = global_mapping.get('type_column', 'Type')
            
            st.markdown(f"""
            Expected columns:
            - Year column: **{year_col}**
            - Value column: **{value_col}**
            - Type column: **{type_col}** (with values 'Historical' and 'Forecast')
            """)
        except Exception:
            st.warning("Could not retrieve column mapping from configuration")
        
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
                
                # Validate data
                is_valid, error_message = validate_global_forecast(df, config_manager)
                
                if is_valid:
                    st.success("Global market data validated successfully!")
                    
                    # Data summary
                    years = sorted(df[year_col].unique())
                    st.markdown(f"""
                    **Data Summary:**
                    - Years: {min(years)} to {max(years)}
                    - Total years: {len(years)}
                    - Historical years: {len(df[df[type_col] == 'Historical'])}
                    - Forecast years: {len(df[df[type_col] == 'Forecast'])}
                    """)
                    
                    # Simple line chart of global market
                    st.subheader("Global Market Trend")
                    chart_data = df.copy()
                    chart_data = chart_data.sort_values(by=year_col)
                    st.line_chart(data=chart_data, x=year_col, y=value_col)
                    
                    # Option to save
                    if st.button("Save Global Market Data"):
                        data['global_forecast'] = df
                        st.session_state.global_forecast = df
                        st.success("Global market data saved to session!")
                else:
                    st.error(f"Validation error: {error_message}")
                    
                    # Column mapper
                    st.subheader("Column Mapping")
                    st.warning("The data does not match the expected format. You can adjust the column mapping below.")
                    
                    year_col_map = st.selectbox("Year Column", options=df.columns.tolist(), index=0 if year_col not in df.columns else df.columns.tolist().index(year_col))
                    value_col_map = st.selectbox("Value Column", options=df.columns.tolist(), index=0 if value_col not in df.columns else df.columns.tolist().index(value_col))
                    type_col_map = st.selectbox("Type Column", options=df.columns.tolist(), index=0 if type_col not in df.columns else df.columns.tolist().index(type_col))
                    
                    # Option to update mapping and save
                    if st.button("Update Mapping and Save"):
                        # Create a new DataFrame with the correct column names
                        mapped_df = df.copy()
                        mapped_df = mapped_df.rename(columns={
                            year_col_map: year_col,
                            value_col_map: value_col,
                            type_col_map: type_col
                        })
                        
                        # Validate again
                        is_valid, error_message = validate_global_forecast(mapped_df, config_manager)
                        
                        if is_valid:
                            data['global_forecast'] = mapped_df
                            st.session_state.global_forecast = mapped_df
                            st.success("Global market data mapped and saved to session!")
                        else:
                            st.error(f"Validation error after mapping: {error_message}")
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Option to use existing data
        if st.session_state.get('global_forecast') is not None:
            st.info("Global market data is already loaded in session.")

            if st.button("Show Loaded Data", key="show_global_data"):
                st.dataframe(st.session_state.global_forecast.head(10))
    
    # Tab 2: Country Historical Data
    with tab2:
        st.header("Country Historical Data")
        st.markdown("""
        Upload your country historical data in Excel or CSV format.
        This should contain historical market values by country.
        """)
        
        # Get expected path from configuration
        try:
            country_path = config_manager.get_config_section('data_sources').get('country_historical', {}).get('path', '')
            country_format = config_manager.get_config_section('data_sources').get('country_historical', {}).get('format', 'wide')
            st.info(f"Configured path: {country_path} (Format: {country_format})")
        except Exception:
            country_path = ''
            country_format = 'wide'
        
        # Get expected column mapping from configuration
        try:
            country_mapping = config_manager.get_column_mapping('country_historical')
            id_col = country_mapping.get('id_column', 'idGeo')
            country_col = country_mapping.get('country_column', 'Country')
            
            st.markdown(f"""
            Expected columns:
            - ID column: **{id_col}**
            - Country column: **{country_col}**
            - Format: **{country_format}**
            """)
            
            if country_format == 'wide':
                st.markdown("Wide format: Each year should be a separate column")
            else:
                st.markdown("Long format: Should include 'Year' and 'Value' columns")
        except Exception:
            st.warning("Could not retrieve column mapping from configuration")
        
        # Data format selection
        data_format = st.radio("Data Format", options=["Wide", "Long"], 
                              index=0 if country_format.lower() == 'wide' else 1,
                              horizontal=True)
        
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
                
                # Convert format if needed
                if (data_format == "Wide" and country_format.lower() == 'long') or \
                   (data_format == "Long" and country_format.lower() == 'wide'):
                    st.warning(f"Data format ({data_format}) does not match configured format ({country_format}). Converting...")
                    
                    if data_format == "Wide" and country_format.lower() == 'long':
                        # Need to convert wide to long
                        df = convert_wide_to_long(df, config_manager)
                    else:
                        # Need to convert long to wide
                        df = convert_long_to_wide(df, config_manager)
                    
                    st.subheader("Converted Data Preview")
                    st.dataframe(df.head(10))
                
                # Validate data
                is_valid, error_message = validate_country_historical(df, config_manager)
                
                if is_valid:
                    st.success("Country historical data validated successfully!")
                    
                    # Data summary
                    if data_format == "Wide":
                        year_cols = [col for col in df.columns if str(col).isdigit() or 
                                     (isinstance(col, str) and col.isdigit())]
                        years = sorted([int(col) for col in year_cols])
                        countries = df[country_col].nunique()
                    else:  # Long format
                        year_col = 'Year'  # Default, should be configurable
                        years = sorted(df[year_col].unique())
                        countries = df[country_col].nunique()
                    
                    st.markdown(f"""
                    **Data Summary:**
                    - Countries: {countries}
                    - Years: {min(years)} to {max(years)}
                    - Total years: {len(years)}
                    """)
                    
                    # Option to save
                    if st.button("Save Country Historical Data"):
                        data['country_historical'] = df
                        st.session_state.country_historical = df
                        st.success("Country historical data saved to session!")
                else:
                    st.error(f"Validation error: {error_message}")
                    
                    # Column mapper
                    st.subheader("Column Mapping")
                    st.warning("The data does not match the expected format. You can adjust the column mapping below.")
                    
                    # Get current column mapping
                    try:
                        country_mapping = config_manager.get_column_mapping('country_historical')
                        id_col = country_mapping.get('id_column', 'idGeo')
                        country_col = country_mapping.get('country_column', 'Country')
                    except Exception:
                        id_col = 'idGeo'
                        country_col = 'Country'
                    
                    id_col_map = st.selectbox("ID Column", options=df.columns.tolist(), index=0 if id_col not in df.columns else df.columns.tolist().index(id_col))
                    country_col_map = st.selectbox("Country Column", options=df.columns.tolist(), index=0 if country_col not in df.columns else df.columns.tolist().index(country_col))
                    
                    if data_format == "Long":
                        year_col_map = st.selectbox("Year Column", options=df.columns.tolist(), index=0 if 'Year' not in df.columns else df.columns.tolist().index('Year'))
                        value_col_map = st.selectbox("Value Column", options=df.columns.tolist(), index=0 if 'Value' not in df.columns else df.columns.tolist().index('Value'))
                    
                    # Option to update mapping and save
                    if st.button("Update Mapping and Save"):
                        # Create a new DataFrame with the correct column names
                        mapped_df = df.copy()
                        
                        if data_format == "Wide":
                            mapped_df = mapped_df.rename(columns={
                                id_col_map: id_col,
                                country_col_map: country_col
                            })
                        else:  # Long format
                            mapped_df = mapped_df.rename(columns={
                                id_col_map: id_col,
                                country_col_map: country_col,
                                year_col_map: 'Year',
                                value_col_map: 'Value'
                            })
                        
                        # Validate again
                        is_valid, error_message = validate_country_historical(mapped_df, config_manager)
                        
                        if is_valid:
                            data['country_historical'] = mapped_df
                            st.session_state.country_historical = mapped_df
                            st.success("Country historical data mapped and saved to session!")
                        else:
                            st.error(f"Validation error after mapping: {error_message}")
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Option to use existing data
        if st.session_state.get('country_historical') is not None:
            st.info("Country historical data is already loaded in session.")

            if st.button("Show Loaded Data", key="show_country_data"):
                st.dataframe(st.session_state.country_historical.head(10))
    
    # Tab 3: Indicators
    with tab3:
        st.header("Market Indicators")
        st.markdown("""
        Upload indicator data that can influence market distribution.
        Examples include GDP, population, technology adoption indices, etc.
        """)
        
        # Get configured indicators from configuration
        try:
            configured_indicators = config_manager.get_config_section('data_sources').get('indicators', [])
            if configured_indicators:
                st.info(f"Configured indicators: {', '.join([ind.get('name', 'Unnamed') for ind in configured_indicators])}")
        except Exception:
            configured_indicators = []
        
        # Create a new indicator
        st.subheader("Add New Indicator")
        
        # Indicator name
        indicator_name = st.text_input("Indicator Name")
        
        # Indicator type
        indicator_type = st.radio("Indicator Type", options=["value", "rank"], horizontal=True)
        
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
                
                # Validate data
                is_valid, error_message = validate_indicator_data(df, indicator_type)
                
                if is_valid:
                    st.success(f"Indicator '{indicator_name}' validated successfully!")
                    
                    # Weight configuration
                    weight_config = st.radio("Weight Calculation", options=["Auto", "Manual"], horizontal=True)
                    
                    if weight_config == "Manual":
                        weight_value = st.slider("Indicator Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
                    else:
                        weight_value = "auto"
                    
                    # Option to save indicator
                    if st.button("Add Indicator"):
                        # Create indicator metadata
                        indicator_meta = {
                            "name": indicator_name,
                            "type": indicator_type,
                            "weight": weight_value
                        }
                        
                        # Store indicator data in session state
                        if 'indicators' not in st.session_state:
                            st.session_state.indicators = {}
                        
                        st.session_state.indicators[indicator_name] = {
                            "data": df,
                            "meta": indicator_meta
                        }
                        
                        # Also add to return data
                        data['indicators'][indicator_name] = {
                            "data": df,
                            "meta": indicator_meta
                        }
                        
                        st.success(f"Indicator '{indicator_name}' added successfully!")
                else:
                    st.error(f"Validation error: {error_message}")
            
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        # Display currently loaded indicators
        if hasattr(st.session_state, 'indicators') and st.session_state.indicators:
            st.subheader("Loaded Indicators")
            
            for name, details in st.session_state.indicators.items():
                with st.expander(f"{name} ({details['meta']['type']})"):
                    st.dataframe(details['data'].head(5))
                    st.text(f"Weight: {details['meta']['weight']}")
                    
                    # Option to remove
                    if st.button(f"Remove {name}"):
                        del st.session_state.indicators[name]
                        
                        # Also remove from return data if present
                        if name in data['indicators']:
                            del data['indicators'][name]
                        
                        st.success(f"Indicator '{name}' removed.")
                        st.rerun()
    
    return data


def create_test_data(config_manager: ConfigurationManager) -> Dict[str, Any]:
    """
    Create test data for demonstration purposes.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Dictionary containing test data
    """
    # Initialize data dictionary
    data = {
        'global_forecast': None,
        'country_historical': None,
        'indicators': {}
    }
    
    # Get column mapping
    try:
        global_mapping = config_manager.get_column_mapping('global_forecast')
        country_mapping = config_manager.get_column_mapping('country_historical')
        
        year_col = global_mapping.get('year_column', 'Year')
        value_col = global_mapping.get('value_column', 'Value')
        type_col = global_mapping.get('type_column', 'Type')
        
        id_col = country_mapping.get('id_column', 'idGeo')
        country_col = country_mapping.get('country_column', 'Country')
    except Exception:
        # Use defaults if mapping not available
        year_col = 'Year'
        value_col = 'Value'
        type_col = 'Type'
        id_col = 'idGeo'
        country_col = 'Country'
    
    # Create global forecast data
    years = list(range(2018, 2031))
    historical_years = [y for y in years if y <= 2023]
    forecast_years = [y for y in years if y > 2023]
    
    # Create historical data with realistic pattern
    historical_values = [100 + i * 15 + np.random.normal(0, 3) for i in range(len(historical_years))]
    
    # Create forecast data with growth trend
    base = historical_values[-1]
    growth_rate = 0.12  # 12% annual growth
    forecast_values = [base * ((1 + growth_rate) ** (i + 1)) for i in range(len(forecast_years))]
    
    # Combine into DataFrame
    global_data = []
    
    for year, value in zip(historical_years, historical_values):
        global_data.append({
            year_col: year,
            value_col: value,
            type_col: 'Historical'
        })
    
    for year, value in zip(forecast_years, forecast_values):
        global_data.append({
            year_col: year,
            value_col: value,
            type_col: 'Forecast'
        })
    
    global_df = pd.DataFrame(global_data)
    
    # Create country historical data
    countries = [
        {"id": 1, "name": "United States"},
        {"id": 2, "name": "China"},
        {"id": 3, "name": "Japan"},
        {"id": 4, "name": "Germany"},
        {"id": 5, "name": "United Kingdom"},
        {"id": 6, "name": "France"},
        {"id": 7, "name": "India"},
        {"id": 8, "name": "Canada"},
        {"id": 9, "name": "South Korea"},
        {"id": 10, "name": "Brazil"}
    ]
    
    # Distribution parameters
    shares = {
        1: 0.32,  # US
        2: 0.24,  # China
        3: 0.12,  # Japan
        4: 0.08,  # Germany
        5: 0.06,  # UK
        6: 0.05,  # France
        7: 0.04,  # India
        8: 0.03,  # Canada
        9: 0.03,  # South Korea
        10: 0.03  # Brazil
    }
    
    # Create country data in long format
    country_data = []
    
    for year in historical_years:
        total_value = global_df[global_df[year_col] == year][value_col].values[0]
        
        for country in countries:
            country_id = country["id"]
            
            # Apply some random variation to shares
            share_variation = np.random.normal(0, 0.01)
            effective_share = max(0.01, shares[country_id] + share_variation)
            
            # Calculate country value
            value = total_value * effective_share
            
            country_data.append({
                id_col: country_id,
                country_col: country["name"],
                'Year': year,
                'Value': value
            })
    
    country_df = pd.DataFrame(country_data)
    
    # Create GDP indicator (value type)
    gdp_data = []
    
    # GDP base values
    gdp_base = {
        1: 21400,  # US
        2: 16800,  # China
        3: 5200,   # Japan
        4: 4100,   # Germany
        5: 3200,   # UK
        6: 2800,   # France
        7: 3100,   # India
        8: 1900,   # Canada
        9: 1700,   # South Korea
        10: 2000   # Brazil
    }
    
    for year in range(2018, 2024):
        for country in countries:
            country_id = country["id"]
            
            # Apply growth to GDP
            growth_factor = 1 + (np.random.normal(0.03, 0.01))  # Around 3% growth with variation
            
            if year == 2018:
                gdp = gdp_base[country_id]
            else:
                prev_gdp = next(item['GDP'] for item in gdp_data if item['Year'] == year - 1 and item[id_col] == country_id)
                gdp = prev_gdp * growth_factor
            
            gdp_data.append({
                id_col: country_id,
                country_col: country["name"],
                'Year': year,
                'GDP': gdp
            })
    
    gdp_df = pd.DataFrame(gdp_data)
    
    # Create technology adoption indicator (rank type)
    tech_adoption_data = []
    
    # Technology adoption ranks (lower is better)
    tech_ranks = {
        1: 1,  # US
        2: 2,  # China
        3: 3,  # Japan
        4: 4,  # Germany
        5: 5,  # UK
        6: 6,  # France
        7: 8,  # India
        8: 7,  # Canada
        9: 9,  # South Korea
        10: 10  # Brazil
    }
    
    for year in range(2018, 2024):
        for country in countries:
            country_id = country["id"]
            
            # Apply some random variation but maintain general order
            rank_variation = np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
            effective_rank = max(1, min(10, tech_ranks[country_id] + rank_variation))
            
            tech_adoption_data.append({
                id_col: country_id,
                country_col: country["name"],
                'Year': year,
                'TechRank': effective_rank
            })
    
    tech_adoption_df = pd.DataFrame(tech_adoption_data)
    
    # Store data
    data['global_forecast'] = global_df
    data['country_historical'] = country_df
    data['indicators']['GDP'] = {
        "data": gdp_df,
        "meta": {
            "name": "GDP",
            "type": "value",
            "weight": "auto"
        }
    }
    data['indicators']['TechAdoption'] = {
        "data": tech_adoption_df,
        "meta": {
            "name": "TechAdoption",
            "type": "rank",
            "weight": "auto"
        }
    }
    
    return data