"""
Enhanced Visualization Interface Fix - Handles flexible column naming

This module provides utility functions to handle different column names for market values
in the distributed market data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

def detect_value_column(df: pd.DataFrame) -> str:
    """
    Detect the column name containing market values in the DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Name of the value column
    """
    # List of possible value column names
    possible_names = ['Value', 'value', 'market_value', 'Market_Value', 'market_size', 
                     'Market_Size', 'Values', 'values', 'Amount', 'amount', 'distributed_value',
                     'Distributed_Value', 'market', 'Market']
    
    # Check each possible name
    for col_name in possible_names:
        if col_name in df.columns:
            return col_name
    
    # If no standard name found, look for numeric columns that aren't Year or id columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove known non-value columns
    exclude_cols = ['Year', 'year', 'idGeo', 'id', 'ID', 'index', 'market_share', 
                   'Market_Share', 'growth_rate', 'Growth_Rate']
    
    for col in exclude_cols:
        if col in numeric_cols:
            numeric_cols.remove(col)
    
    # If we have numeric columns left, use the first one
    if numeric_cols:
        return numeric_cols[0]
    
    # Default fallback
    return 'Value'

def standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the DataFrame to ensure it has a 'Value' column.
    
    Args:
        df: DataFrame to standardize
        
    Returns:
        DataFrame with standardized 'Value' column
    """
    df_copy = df.copy()
    
    # Detect the actual value column
    value_col = detect_value_column(df)
    
    # If it's not already called 'Value', rename it
    if value_col != 'Value' and value_col in df_copy.columns:
        df_copy = df_copy.rename(columns={value_col: 'Value'})
    
    # Ensure the Value column exists and is numeric
    if 'Value' in df_copy.columns:
        try:
            # Convert to numeric, but check for issues
            before_conversion = df_copy['Value'].notna().sum()
            df_copy['Value'] = pd.to_numeric(df_copy['Value'], errors='coerce')
            after_conversion = df_copy['Value'].notna().sum()
            
            # If we lost data during conversion, log a warning
            if before_conversion > after_conversion:
                import streamlit as st
                st.warning(f"Warning: {before_conversion - after_conversion} values could not be converted to numeric in the Value column")
            
            # Check for NaN values but don't automatically fill with 0
            nan_count = df_copy['Value'].isna().sum()
            if nan_count > 0:
                import streamlit as st
                st.warning(f"Found {nan_count} NaN values in the Value column")
                # Only fill with 0 if we have some non-NaN values
                if df_copy['Value'].notna().any():
                    df_copy['Value'] = df_copy['Value'].fillna(0)
                else:
                    st.error("All values in the Value column are NaN!")
                
        except Exception as e:
            import streamlit as st
            st.error(f"Error processing Value column: {str(e)}")
    
    return df_copy

def get_country_column(config_manager) -> str:
    """
    Get the country column name from configuration.
    
    Args:
        config_manager: Configuration manager instance
        
    Returns:
        Name of the country column
    """
    try:
        country_mapping = config_manager.get_column_mapping('country_historical')
        return country_mapping.get('country_column', 'Country')
    except:
        return 'Country'

def prepare_visualization_data(distributed_market: pd.DataFrame, config_manager) -> Tuple[pd.DataFrame, str]:
    """
    Prepare data for visualization by standardizing column names.
    
    Args:
        distributed_market: Raw distributed market DataFrame
        config_manager: Configuration manager instance
        
    Returns:
        Tuple of (standardized DataFrame, country column name)
    """
    # Standardize the value column
    standardized_df = standardize_dataframe_columns(distributed_market)
    
    # Get country column name
    country_col = get_country_column(config_manager)
    
    # Ensure required columns exist
    required_cols = ['Year', country_col]
    missing_cols = [col for col in required_cols if col not in standardized_df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert Year to numeric if it's not already
    try:
        standardized_df['Year'] = pd.to_numeric(standardized_df['Year'], errors='coerce')
    except:
        pass
    
    # Remove any rows with null values in essential columns
    standardized_df = standardized_df.dropna(subset=['Year', country_col, 'Value'])
    
    return standardized_df, country_col

def fix_enhanced_visualization():
    """
    Apply fixes to the enhanced visualization module.
    This function modifies the enhanced_visualization module to use flexible column naming.
    """
    import src.streamlit.enhanced_visualization as ev
    
    # Store original functions
    original_executive_dashboard = ev.render_executive_dashboard
    original_market_size_viz = ev.render_enhanced_market_size_visualization
    original_world_map = ev.render_interactive_world_map
    original_main_interface = ev.render_enhanced_visualization_interface
    
    # Create wrapped versions
    def wrapped_executive_dashboard(distributed_market: pd.DataFrame, config_manager) -> None:
        standardized_df, _ = prepare_visualization_data(distributed_market, config_manager)
        return original_executive_dashboard(standardized_df, config_manager)
    
    def wrapped_market_size_viz(distributed_market: pd.DataFrame, config_manager) -> None:
        standardized_df, _ = prepare_visualization_data(distributed_market, config_manager)
        return original_market_size_viz(standardized_df, config_manager)
    
    def wrapped_world_map(distributed_market: pd.DataFrame, config_manager) -> None:
        standardized_df, _ = prepare_visualization_data(distributed_market, config_manager)
        return original_world_map(standardized_df, config_manager)
    
    def wrapped_main_interface(config_manager) -> None:
        # Get the distributed market data from session state
        if 'distributed_market' in ev.st.session_state and ev.st.session_state.distributed_market is not None:
            # Standardize it before storing back
            standardized_df, _ = prepare_visualization_data(ev.st.session_state.distributed_market, config_manager)
            ev.st.session_state.distributed_market = standardized_df
        
        return original_main_interface(config_manager)
    
    # Replace the functions
    ev.render_executive_dashboard = wrapped_executive_dashboard
    ev.render_enhanced_market_size_visualization = wrapped_market_size_viz
    ev.render_interactive_world_map = wrapped_world_map
    ev.render_enhanced_visualization_interface = wrapped_main_interface
    
    print("Enhanced visualization fixes applied successfully!")
